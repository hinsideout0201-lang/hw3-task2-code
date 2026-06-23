import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


root = Path(__file__).resolve().parents[1]


def load_log(path, name):
    df = pd.read_csv(path)
    df["run"] = name
    df["eval_l1"] = pd.to_numeric(df["eval_l1"], errors="coerce")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--A", required=True)
    parser.add_argument("--ABC", required=True)
    parser.add_argument("--out-dir", default=str(root / "results"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a_log = load_log(args.A, "A-only")
    abc_log = load_log(args.ABC, "ABC-joint")
    df = pd.concat([a_log, abc_log], ignore_index=True)

    plt.figure()
    for name, group in df.groupby("run"):
        plt.plot(group["step"], group["train_l1"], label=name)
    plt.xlabel("Training Step")
    plt.ylabel("Train Action L1 Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "train_loss_curve.png", dpi=200)
    plt.close()

    plt.figure()
    for name, group in df.dropna(subset=["eval_l1"]).groupby("run"):
        plt.plot(group["step"], group["eval_l1"], marker="o", label=name)
    plt.xlabel("Training Step")
    plt.ylabel("Zero-shot Action L1 on splitD")
    plt.title("Validation Curve on Unseen Environment D")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "eval_l1_curve.png", dpi=200)
    plt.close()

    summary = []
    for name, group in df.groupby("run"):
        last_train = group["train_l1"].dropna().iloc[-1]
        eval_values = group["eval_l1"].dropna()
        last_eval = eval_values.iloc[-1] if len(eval_values) > 0 else None
        best_eval = eval_values.min() if len(eval_values) > 0 else None

        summary.append({
            "run": name,
            "last_train_l1": last_train,
            "last_eval_l1": last_eval,
            "best_eval_l1": best_eval
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)

    print(summary_df)
    print("Saved figures to:", out_dir)


if __name__ == "__main__":
    main()