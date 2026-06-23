import os
import sys
import csv
import subprocess
from pathlib import Path


root = Path(__file__).resolve().parents[1]
steps = 5000
batchsize = 2
chunk = 8
image = 128
trainnum = 50000
evalnum = 3000
stats = 5000
log = 50
evalgap = 500
save = 5000


def setup_env():
    cache_dir = root / "cache"
    os.environ.setdefault("HF_HOME", str(cache_dir / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "hf_datasets"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hf_home" / "hub"))
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)


def getstep(path):
    if not path.exists():
        return 0
    step = 0
    try:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("step"):
                    step = max(step, int(float(row["step"])))
    except Exception:
        return 0
    return step


def run(command, file):
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a") as f:
        process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            env=os.environ.copy(),)
        for line in process.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"failed: {return_code}")


def train(task, split):
    out_dir = root / "outputs" / task
    log_csv = out_dir / "train_log.csv"
    run_log = root / "outputs" / f"{task}_run.log"

    step = getstep(log_csv)
    if step >= steps:
        print(f"{task} already finished: step={step}, skip training.")
        return

    command = [
        sys.executable,
        "-u",
        "scripts/trainCalvin.py",
        "--split",
        *split,
        "--eval-split",
        "splitD",
        "--task",
        task,
        "--steps",
        str(steps),
        "--batch-size",
        str(batchsize),
        "--chunk",
        str(chunk),
        "--image",
        str(image),
        "--trainnum",
        str(trainnum),
        "--evalnum",
        str(evalnum),
        "--stats",
        str(stats),
        "--log",
        str(log),
        "--evalgap",
        str(evalgap),
        "--save",
        str(save),
    ]

    run(command, run_log)


def plot_results(A, ABC, result_dir):
    command = [sys.executable, "-u", "scripts/plotActResults.py", "--A", f"outputs/{A}/train_log.csv",
        "--ABC", f"outputs/{ABC}/train_log.csv", "--out-dir", result_dir]
    run_log = root / "outputs" / "plot_results.log"
    run(command, run_log)


def main():
    setup_env()
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)

    A = f"A_{steps}"
    ABC = f"ABC_{steps}"
    result_dir = f"results_{steps}"

    print(f"steps={steps}, batchsize={batchsize}, chunk={chunk}, image={image}")
    print("Start A-only training")
    train(A, ["splitA"])
    print("Start ABC-joint training")
    train(ABC, ["splitA", "splitB", "splitC"])
    print("Plot results")
    plot_results(A, ABC, result_dir)
    print("Done:", root / result_dir)


if __name__ == "__main__":
    main()
