# Kaggle Notebook 完整训练模板（壁画修复模型）

**复制下面代码块到 Kaggle Notebook 依次运行**

---

## 0. 环境与 GPU 说明（必读）

| 项目 | 说明 |
|------|------|
| 推荐 GPU | **T4**（Accelerator → GPU T4 x2 或 GPU T4） |
| 避免 | **P100**：当前 Kaggle 默认 PyTorch 不支持 sm_60，会出现 CUDA capability 警告且可能无法真正用 GPU |
| Python | 3.10 / 3.12 均可 |
| 不要加 `--fp16` | 本仓库已禁用 apex AMP；Kaggle 通常也没有 apex |

---

## 1. 克隆 / 上传代码

```python
# 方式 A：从 GitHub 克隆（若仓库可访问）
!git clone https://github.com/nwfnyfgvfir/Mural.git
%cd Mural

# 方式 B：若你已把代码作为 Dataset 上传，或手动上传到 /kaggle/working/
# 则把工作目录切到代码根目录，例如：
# %cd /kaggle/working/Mural
```

---

## 2. 安装依赖（不要强行装 apex）

```python
!pip install -q dominate scipy pillow
# pytorch_ssim 若仓库自带目录则无需 pip；否则：
# !pip install -q pytorch-ssim
```

> 不要执行 `pip install torch ... cu121` 覆盖 Kaggle 自带 PyTorch（容易和 GPU 不匹配）。  
> 若必须用 **P100**，再考虑安装支持 sm_60 的旧版 PyTorch（不推荐，优先换 T4）。

---

## 3. 准备数据集（最容易出错）

### 3.1 代码要求的目录结构

```text
<dataroot>/
  train_A/     # 不完整壁画（输入）
  train_B/     # 完整壁画（目标），文件名与 train_A 一一对应
```

本仓库本地结构为：

```text
datasets/Incomplet-mural-data/seq1/train_A/
datasets/Incomplet-mural-data/seq1/train_B/
```

因此 **`--dataroot` 必须指向含有 `train_A` 和 `train_B` 的那一层**，即 `.../seq1`。

### 3.2 在 Notebook 中检查路径（先跑这段）

```python
import os

# 常见挂载位置，按你实际上传方式改
candidates = [
    "/kaggle/input",
    "/kaggle/working",
    "/kaggle/working/Mural",
    "/kaggle/working/Mural/datasets",
]

print("=== 搜索 train_A ===")
for root, dirs, files in os.walk("/kaggle"):
    if "train_A" in dirs:
        full = os.path.join(root, "train_A")
        n = len([f for f in os.listdir(full) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        print(f"找到: {full}  (约 {n} 张图)")
        print(f"  → 建议 dataroot = {root}")
```

### 3.3 三种常见上传方式

**方式 1：把 `Incomplet-mural-data` 整个打成 Dataset 上传**

Kaggle 会挂到类似：

```text
/kaggle/input/<你的dataset名>/Incomplet-mural-data/seq1/train_A
```

或（解压后少一层）：

```text
/kaggle/input/<你的dataset名>/seq1/train_A
```

用第 3.2 步搜到的路径设置 `dataroot`。

**方式 2：复制到代码目录下（与本地一致）**

```python
!mkdir -p /kaggle/working/Mural/datasets
# 按你实际 input 路径改左边
!cp -r /kaggle/input/<你的dataset名>/Incomplet-mural-data /kaggle/working/Mural/datasets/
!ls /kaggle/working/Mural/datasets/Incomplet-mural-data/seq1/train_A | head
```

然后：

```bash
--dataroot ./datasets/Incomplet-mural-data/seq1
```

**方式 3：只上传了 zip，需要解压**

```python
!unzip -q /kaggle/input/<dataset>/Incomplet-mural-data.zip -d /kaggle/working/Mural/datasets/
!ls /kaggle/working/Mural/datasets/Incomplet-mural-data/seq1/
# 应看到 train_A  train_B
```

### 3.4 最终校验（必须通过再训练）

```python
import os
dataroot = "/kaggle/input/你的实际路径/seq1"  # 改成你搜到的
assert os.path.isdir(os.path.join(dataroot, "train_A")), "缺少 train_A"
assert os.path.isdir(os.path.join(dataroot, "train_B")), "缺少 train_B"
print("A:", len(os.listdir(os.path.join(dataroot, "train_A"))))
print("B:", len(os.listdir(os.path.join(dataroot, "train_B"))))
print("OK")
```

---

## 4. 启动训练（不要加 --fp16）

把下面 `DATAROOT` 换成第 3 步校验通过的路径：

```python
DATAROOT = "/kaggle/input/你的实际路径/seq1"  # 或 ./datasets/Incomplet-mural-data/seq1

!python train.py \
  --name mural_kaggle \
  --netG global_ca_aspp \
  --dataroot {DATAROOT} \
  --label_nc 0 --no_instance \
  --load_pretrain '' \
  --niter 100 --niter_decay 50 \
  --save_epoch_freq 20 \
  --display_freq 50 \
  --print_freq 20 \
  --batchSize 1 \
  --loadSize 512 --fineSize 512
```

> **必须**加 `--load_pretrain ''`，否则会去找不存在的 `checkpoints/global/latest_net_G.pth` 并报错。  
> 若要用已有权重微调，把 pth 放到某目录后写 `--load_pretrain /path/to/dir`（目录内需有 `latest_net_G.pth`）。

显存不够时：

```python
!python train.py \
  --name mural_kaggle \
  --netG global \
  --dataroot {DATAROOT} \
  --label_nc 0 --no_instance \
  --load_pretrain '' \
  --niter 100 --niter_decay 50 \
  --save_epoch_freq 20 \
  --batchSize 1 \
  --loadSize 512 --fineSize 512
```

---

## 5. 训练中查看

```python
!tail -n 30 /kaggle/working/Mural/checkpoints/mural_kaggle/loss_log.txt
!ls /kaggle/working/Mural/checkpoints/mural_kaggle/
```

权重在：`checkpoints/mural_kaggle/`（`latest_net_G.pth` 等）。

---

## 6. 推理

```python
!python test.py \
  --name mural_kaggle \
  --which_epoch latest \
  --netG global_ca_aspp \
  --dataroot /kaggle/input/你的实际路径/test \
  --checkpoints_dir ./checkpoints \
  --label_nc 0 --no_instance \
  --how_many 10
```

注意：测试集目录需含 `test_A/`（本仓库 `datasets/Incomplet-mural-data/test/test_A`）。

---

## 7. 当前报错对照

| 报错 | 原因 | 处理 |
|------|------|------|
| `fractions has no attribute gcd` | Python 3.9+ 移除 | 使用已修复的 `train.py` |
| `train_A is not a valid directory` | `dataroot` 指错层 / 未上传数据 | 用第 3 节搜索并校验 |
| CUDA capability sm_60 | P100 + 新 PyTorch | 换 **T4** GPU |
| `No module named apex` | 装了 `--fp16` | **不要**加 `--fp16` |
| `Generator must exist` / `latest_net_G.pth not exists` | 默认去加载 `checkpoints/global` | 加 `--load_pretrain ''` 从零训 |

---

**一句话**：先 `find`/`os.walk` 找到真正的 `train_A`，把 `--dataroot` 设成它的**父目录**（`seq1`），再训练。
