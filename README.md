# HW3 ACT-CALVIN 实验代码说明

## 1. 仓库结构

```text
.
├── scripts/
│   ├── mainrun.py
│   ├── trainCalvin.py
│   ├── evalCalvin.py
│   └── plotActResults.py
└── results_5000/
    ├── train_loss_curve.png
    ├── eval_l1_curve.png
    ├── summary_metrics.csv
    ├── A_5000_best_D_eval.csv
    └── ABC_5000_best_D_eval.csv
```

各文件作用如下：

* `scripts/mainrun.py`：依次训练 A-only 和 ABC-joint 两组模型，并在训练结束后画图；
* `scripts/trainCalvin.py`：负责读取 CALVIN 数据、构造 ACTPolicy、训练模型、周期性评估并保存 checkpoint；
* `scripts/evalCalvin.py`：独立评测脚本，加载训练好的 checkpoint，在 splitD 上计算 Action L1；
* `scripts/plotActResults.py`：根据训练日志生成 loss 曲线、eval 曲线和指标汇总表；

## 2. 环境准备

实验基于 LeRobot、PyTorch、pandas 和 matplotlib。

进入项目根目录后，激活环境：

```bash
conda activate lerobot
```


## 3. 数据集准备

运行代码前，需要将CALVIN数据集放置到：

```text
data/calvin-lerobot/
```


## 4. 训练方式

在项目根目录下运行：

```bash
python scripts/mainrun.py
```


训练输出保存在：

```text
outputs/A_5000/
outputs/ABC_5000/
```


## 5. 独立评测方式

训练完成后，可以使用 `evalCalvin.py` 在 `splitD` 上重新评测模型。

评测 A-only 最优模型：

```bash
python scripts/evalCalvin.py \
  --ckpt outputs/A_5000/checkpoint_best.pt \
  --split splitD \
  --name A_5000_best_D \
  --evalnum 3000 \
  --batch-size 2 \
  --out-dir results_5000
```

评测 ABC-joint 最优模型：

```bash
python scripts/evalCalvin.py \
  --ckpt outputs/ABC_5000/checkpoint_best.pt \
  --split splitD \
  --name ABC_5000_best_D \
  --evalnum 3000 \
  --batch-size 2 \
  --out-dir results_5000
```

评测结果保存在：

```text
results_5000/A_5000_best_D_eval.csv
results_5000/ABC_5000_best_D_eval.csv
```

## 6. 重新生成曲线图

如果已有训练日志，可以使用 `plotActResults.py` 重新生成训练曲线和指标表：

```bash
python scripts/plotActResults.py \
  --A outputs/A_5000/train_log.csv \
  --ABC outputs/ABC_5000/train_log.csv \
  --out-dir results_5000
```

画图结果保存在：

```text
results_5000/train_loss_curve.png
results_5000/eval_l1_curve.png
results_5000/summary_metrics.csv
```
