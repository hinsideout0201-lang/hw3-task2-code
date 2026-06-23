import argparse
import csv
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

root = Path(__file__).resolve().parents[1]

def setup_env():
    cache_dir = root / "cache"
    os.environ.setdefault("HF_HOME", str(cache_dir / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "hf_datasets"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hf_home" / "hub"))
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)


class CalvinDataset(Dataset):
    def __init__(self, data_root, splits, chunk_size=8, image_size=128, max_samples_per_split=20000):
        self.data_root = Path(data_root)
        self.splits = splits
        self.chunk_size = chunk_size
        self.image_size = image_size

        self.datasets = []
        self.action_datasets = []
        self.index_map = []

        for split in splits:
            split_root = self.data_root / split
            print(f"Loading {split} from {split_root}")

            ds = LeRobotDataset(repo_id=f"local/calvin-official-{split}", root=split_root)

            self.datasets.append(ds)
            self.action_datasets.append(ds.hf_dataset.select_columns(["actions", "state", "episode_index"]))

            usable_len = max(0, len(ds) - chunk_size - 1)
            if max_samples_per_split is not None:
                usable_len = min(usable_len, max_samples_per_split)

            ds_id = len(self.datasets) - 1
            for i in range(usable_len):
                self.index_map.append((ds_id, i))

            print(f"{split}: raw length={len(ds)}, used samples={usable_len}")

        print("Total used samples:", len(self.index_map))

    def __len__(self):
        return len(self.index_map)

    def _to_int(self, x):
        if torch.is_tensor(x):
            return int(x.item())
        return int(x)

    def _to_float(self, x):
        if torch.is_tensor(x):
            return x.float()
        return torch.tensor(x, dtype=torch.float32)

    def _prep_image(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if x.ndim == 3 and x.shape[-1] == 3:
            x = x.permute(2, 0, 1).contiguous()
        x = x.float() / 255.0
        x = F.interpolate(x.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False,).squeeze(0)
        return x

    def __getitem__(self, idx):
        ds_id, local_idx = self.index_map[idx]
        ds = self.datasets[ds_id]
        action_ds = self.action_datasets[ds_id]

        item = ds[local_idx]
        episode_id = self._to_int(item["episode_index"])

        image = self._prep_image(item["image"])
        wrist_image = self._prep_image(item["wrist_image"])
        state = self._to_float(item["state"])

        actions = []
        pads = []
        last_action = self._to_float(item["actions"])

        for k in range(self.chunk_size):
            j = local_idx + k

            if j >= len(ds):
                actions.append(last_action)
                pads.append(True)
                continue

            future = action_ds[j]
            future_episode = self._to_int(future["episode_index"])

            if future_episode != episode_id:
                actions.append(last_action)
                pads.append(True)
            else:
                action = self._to_float(future["actions"])
                actions.append(action)
                pads.append(False)
                last_action = action

        actions = torch.stack(actions, dim=0)
        action_is_pad = torch.tensor(pads, dtype=torch.bool)

        return {
            "observation.images.image": image,
            "observation.images.wrist_image": wrist_image,
            "observation.state": state,
            "action": actions,
            "action_is_pad": action_is_pad,
        }


def compute_state_action_stats(data_root, splits, chunk_size=8, max_samples_per_split=5000):
    data_root = Path(data_root)

    state_sum = torch.zeros(15)
    state_sq_sum = torch.zeros(15)
    state_count = 0

    action_sum = torch.zeros(7)
    action_sq_sum = torch.zeros(7)
    action_count = 0

    print("Computing state/action stats...")

    for split in splits:
        split_root = data_root / split
        ds = LeRobotDataset(
            repo_id=f"local/calvin-stats-{split}",
            root=split_root,
        )
        small_ds = ds.hf_dataset.select_columns(["state", "actions", "episode_index"])

        usable_len = max(0, len(ds) - chunk_size - 1)
        usable_len = min(usable_len, max_samples_per_split)

        for i in range(usable_len):
            item = small_ds[i]

            state = item["state"]
            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=torch.float32)
            else:
                state = state.float()

            state_sum += state
            state_sq_sum += state * state
            state_count += 1

            episode_id = int(item["episode_index"])

            last_action = item["actions"]
            if not torch.is_tensor(last_action):
                last_action = torch.tensor(last_action, dtype=torch.float32)
            else:
                last_action = last_action.float()

            for k in range(chunk_size):
                j = i + k
                if j >= len(ds):
                    continue

                future = small_ds[j]
                if int(future["episode_index"]) != episode_id:
                    continue

                action = future["actions"]
                if not torch.is_tensor(action):
                    action = torch.tensor(action, dtype=torch.float32)
                else:
                    action = action.float()

                action_sum += action
                action_sq_sum += action * action
                action_count += 1

    state_mean = state_sum / max(state_count, 1)
    state_var = state_sq_sum / max(state_count, 1) - state_mean * state_mean
    state_std = torch.sqrt(torch.clamp(state_var, min=1e-6))

    action_mean = action_sum / max(action_count, 1)
    action_var = action_sq_sum / max(action_count, 1) - action_mean * action_mean
    action_std = torch.sqrt(torch.clamp(action_var, min=1e-6))

    print("state_mean:", state_mean)
    print("state_std:", state_std)
    print("action_mean:", action_mean)
    print("action_std:", action_std)

    return state_mean, state_std, action_mean, action_std


def build_dataset_stats(image_size, state_mean, state_std, action_mean, action_std):
    return {
        "observation.images.image": {
            "mean": torch.zeros(3, image_size, image_size),
            "std": torch.ones(3, image_size, image_size),
        },
        "observation.images.wrist_image": {
            "mean": torch.zeros(3, image_size, image_size),
            "std": torch.ones(3, image_size, image_size),
        },
        "observation.state": {
            "mean": state_mean,
            "std": state_std,
        },
        "action": {
            "mean": action_mean,
            "std": action_std,
        },
    }


def build_policy(args, device, dataset_stats):
    config = ACTConfig(
        input_features={
            "observation.images.image": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, args.image, args.image),
            ),
            "observation.images.wrist_image": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, args.image, args.image),
            ),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(15,),
            ),
        },
        output_features={
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(7,),
            ),
        },
        chunk_size=args.chunk,
        n_action_steps=args.chunk,
        device=device,
        use_amp=False,

        vision_backbone="resnet18",
        pretrained_backbone_weights=None,

        dim_model=args.dim_model,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=1,

        use_vae=True,
        n_vae_encoder_layers=args.n_vae_encoder_layers,
        latent_dim=args.latent_dim,
        kl_weight=args.kl_weight,

        optimizer_lr=args.lr,
        optimizer_lr_backbone=args.lr_backbone,
        optimizer_weight_decay=args.weight_decay,
    )

    policy = ACTPolicy(config, dataset_stats=dataset_stats).to(device)
    return policy


def evaluate_action_l1(policy, loader, device, max_batches=None):
    with torch.no_grad():
        policy.eval()
    
        total_l1 = 0.0
        total_count = 0
    
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    
            obs_batch = {
                "observation.images.image": batch["observation.images.image"],
                "observation.images.wrist_image": batch["observation.images.wrist_image"],
                "observation.state": batch["observation.state"],
            }
    
            pred = policy.predict_action_chunk(obs_batch)
            target = batch["action"]
            valid = ~batch["action_is_pad"]
    
            l1_each = torch.abs(pred - target).mean(dim=-1)
            l1_valid = l1_each[valid]
    
            if l1_valid.numel() > 0:
                total_l1 += l1_valid.sum().item()
                total_count += l1_valid.numel()
    
    policy.train()
    return total_l1 / max(total_count, 1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", default=str(root / "data" / "calvin-lerobot"))
    parser.add_argument("--split", nargs="+", required=True)
    parser.add_argument("--eval-split", default="splitD")
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", default=str(root / "outputs"))
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--image", type=int, default=128)
    parser.add_argument("--trainnum", type=int, default=50000)
    parser.add_argument("--evalnum", type=int, default=3000)
    parser.add_argument("--stats", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--n-encoder-layers", type=int, default=2)
    parser.add_argument("--n-vae-encoder-layers", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log", type=int, default=20)
    parser.add_argument("--evalgap", type=int, default=200)
    parser.add_argument("--save", type=int, default=1000)

    args = parser.parse_args()

    setup_env()
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    out_dir = Path(args.output_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = CalvinDataset(
    data_root=args.data_root,
    splits=args.split,
    chunk_size=args.chunk,
    image_size=args.image,
    max_samples_per_split=args.trainnum
    )

    eval_set = CalvinDataset(
        data_root=args.data_root,
        splits=[args.eval_split],
        chunk_size=args.chunk,
        image_size=args.image,
        max_samples_per_split=args.evalnum
    )
    
    state_mean, state_std, action_mean, action_std = compute_state_action_stats(
        data_root=args.data_root,
        splits=args.split,
        chunk_size=args.chunk,
        max_samples_per_split=args.stats
    )
    
    dataset_stats = build_dataset_stats(
        image_size=args.image,
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std
    )

    policy = build_policy(args, device, dataset_stats)
    policy.train()

    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    log_path = out_dir / "train_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_l1", "train_kld", "train_total_loss", "eval_l1", "time_sec"])

    step = 0
    t0 = time.time()
    running_l1 = []
    running_kld = []
    running_total = []
    best_eval = float("inf")

    while step < args.steps:
        for batch in train_loader:
            step += 1

            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

            loss, loss_dict = policy(batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            running_l1.append(float(loss_dict.get("l1_loss", 0.0)))
            running_kld.append(float(loss_dict.get("kld_loss", 0.0)))
            running_total.append(float(loss.item()))

            eval_l1 = ""

            if step % args.evalgap == 0 or step == 1:
                eval_l1 = evaluate_action_l1(policy, eval_loader, device)

                if eval_l1 < best_eval:
                    best_eval = eval_l1
                    ckpt = {
                        "policy": policy.state_dict(),
                        "args": vars(args),
                        "step": step,
                        "best_eval_l1": best_eval,
                        "dataset_stats": dataset_stats,
                    }
                    torch.save(ckpt, out_dir / "checkpoint_best.pt")
                    print(f"Saved best checkpoint at step {step}, eval_l1={best_eval:.6f}")

            if step % args.log == 0 or step == 1:
                avg_l1 = sum(running_l1) / max(len(running_l1), 1)
                avg_kld = sum(running_kld) / max(len(running_kld), 1)
                avg_total = sum(running_total) / max(len(running_total), 1)
                elapsed = time.time() - t0

                running_l1 = []
                running_kld = []
                running_total = []

                print(
                    f"step={step:05d} "
                    f"train_l1={avg_l1:.6f} "
                    f"train_kld={avg_kld:.6f} "
                    f"train_total={avg_total:.6f} "
                    f"eval_l1={eval_l1 if eval_l1 != '' else 'NA'} "
                    f"time={elapsed / 60:.1f}min"
                )

                with log_path.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([step, avg_l1, avg_kld, avg_total, eval_l1, elapsed])

            if step % args.save == 0 or step == args.steps:
                ckpt = {
                    "policy": policy.state_dict(),
                    "args": vars(args),
                    "step": step,
                    "dataset_stats": dataset_stats,
                }
                torch.save(ckpt, out_dir / f"checkpoint_step{step}.pt")
                torch.save(ckpt, out_dir / "checkpoint_last.pt")
                print(f"Saved checkpoint at step {step}")

            if step >= args.steps:
                break

    print("Training finished.")


if __name__ == "__main__":
    main()
