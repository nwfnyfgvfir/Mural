# Kaggle Notebook 训练指南（正确版）

## 1. 环境特点

- **推荐 GPU**：T4（不要用 P100，默认 PyTorch 不支持 sm_60）
- **工作目录**：`/kaggle/working/`
- **数据集只读挂载**：`/kaggle/input/<dataset-slug>/`
- **不要使用 `--fp16`**：本仓库已去掉 apex 依赖

## 2. 数据集目录（核心）

`AlignedDataset` 要求：

```text
<dataroot>/train_A/   # 输入：不完整壁画
<dataroot>/train_B/   # 目标：完整壁画
```

本项目本地对应：

```text
datasets/Incomplet-mural-data/seq1/train_A
datasets/Incomplet-mural-data/seq1/train_B
```

因此：

```bash
--dataroot .../seq1
```

**错误示例**（会报 `train_A is not a valid directory`）：

- `dataroot=./datasets/Incomplet-mural-data`（少了 `seq1`，且下面没有直接的 `train_A` 若结构不同）
- 数据只上传到 `/kaggle/input/...`，却用 `./datasets/...` 相对路径且未复制
- zip 未解压 / 多嵌套一层目录

## 3. 上机前检查命令

```python
import os
# 改成你的实际路径
p = "/kaggle/input/xxx/Incomplet-mural-data/seq1"
print("train_A exists:", os.path.isdir(os.path.join(p, "train_A")))
print("train_B exists:", os.path.isdir(os.path.join(p, "train_B")))
```

或搜索：

```python
!find /kaggle/input /kaggle/working -type d -name train_A 2>/dev/null
```

## 4. 推荐训练命令

```python
!python train.py \
  --name mural_kaggle \
  --netG global_ca_aspp \
  --dataroot /kaggle/input/<你的slug>/Incomplet-mural-data/seq1 \
  --label_nc 0 --no_instance \
  --niter 100 --niter_decay 50 \
  --save_epoch_freq 20 \
  --display_freq 50 \
  --print_freq 20 \
  --batchSize 1 \
  --loadSize 512 --fineSize 512
```

显存不足改用 `--netG global`。

## 5. 常见错误

| 现象 | 处理 |
|------|------|
| `train_A is not a valid directory` | 修正 `dataroot` 到含 `train_A`/`train_B` 的父目录 |
| CUDA sm_60 / capability 警告 | Session 设置里换 **GPU T4** |
| `No module named apex` | 去掉 `--fp16` |
| `fractions.gcd` | 使用已修复 `train.py`（`math.gcd`） |
| OOM | `--batchSize 1`、`--loadSize 512 --fineSize 512`、`--netG global` |

## 6. 输出位置

`/kaggle/working/Mural/checkpoints/mural_kaggle/`
