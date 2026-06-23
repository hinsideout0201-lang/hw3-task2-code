import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from trainCalvin import CalvinDataset, build_policy, setup_env, root


def evaluate(policy, loader, device, max_batches=None):
    policy.eval()
    total_l1, total_count = 0.0, 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            obs_batch = {"observation.images.image": batch["observation.images.image"], "observation.images.wrist_image": batch["observation.images.wrist_image"], "observation.state": batch["observation.state"]}
            pred = policy.predict_action_chunk(obs_batch)
            target = batch["action"]
            valid = batch["action_is_pad"].logical_not()

            l1valid = torch.abs(pred-target).mean(dim=-1)[valid]

            if l1valid.numel() > 0:
                total_l1 += l1valid.sum().item()
                total_count += l1valid.numel()

    return total_l1 / max(total_count, 1), total_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="splitD")
    parser.add_argument("--name", required=True)
    parser.add_argument("--data-root", default=str(root / "data" / "calvin-lerobot"))
    parser.add_argument("--out-dir", default=str(root / "results_10000"))
    parser.add_argument("--evalnum", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    setup_env()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = SimpleNamespace(**ckpt["args"])
    dataset_stats = ckpt["dataset_stats"]

    eval_set = CalvinDataset(
        data_root=args.data_root,
        splits=[args.split],
        chunk_size=train_args.chunk,
        image_size=train_args.image,
        max_samples_per_split=args.evalnum
    )

    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    policy = build_policy(train_args, device, dataset_stats)
    policy.load_state_dict(ckpt["policy"])
    policy.to(device)

    action_l1, valid_count = evaluate(
        policy=policy,
        loader=eval_loader,
        device=device,
        max_batches=args.max_batches
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.name}_eval.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "checkpoint", "eval_split", "evalnum", "valid_action_count", "action_l1"])
        writer.writerow([args.name, args.ckpt, args.split, args.evalnum, valid_count, action_l1])

    print("Evaluation finished.")

if __name__ == "__main__":
    main()
