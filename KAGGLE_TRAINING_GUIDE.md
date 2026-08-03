# Kaggle Notebook 训练指南（正确版）

**注意**：Kaggle Notebook 与普通 Linux 服务器有很大不同，需要使用 `!` 前缀运行命令。

---

## 1. Kaggle Notebook 环境特点

- **GPU**：Tesla T4（16GB）或 P100/V100（32GB）
- **Python**：3.10
- **所有文件**保存在 `/kaggle/working/`
- **数据集路径**：`/kaggle/input/你的数据集名/`

---

## 2. 正确训练命令（推荐）

### 在 Kaggle Notebook 中直接运行

```python
!python train.py \
  --name mural_kaggle \
  --netG global_ca_aspp \
  --dataroot /kaggle/input/Incomplet-mural-data/seq1 \
  --label_nc 0 --no_instance \
  --niter 100 --niter_decay 50 \
  --save_epoch_freq 20 \
  --display_freq 50 \
  --print_freq 20 \
  --batchSize 1 \
  --fp16
```

### 如果显存不足（T4 16GB）

```python
!python train.py \
  --name mural_kaggle \
  --netG global \
  --dataroot /kaggle/input/Incomplet-mural-data/seq1 \
  --label_nc 0 --no_instance \
  --niter 100 --save_epoch_freq 20 \
  --fp16
```

---

## 3. 完整训练步骤（Kaggle Notebook 方式）

### 步骤1：上传代码
1. 上传所有 `.py` 文件到 `/kaggle/working/`
2. 上传数据集（`Incomplet-mural-data.zip`）到 `/kaggle/input/`

### 步骤2：安装依赖
```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install dominate scipy numpy pillow
!pip install apex
```

### 步骤3：启动训练
```python
!python train.py --name mural_kaggle --netG global_ca_aspp --dataroot /kaggle/input/Incomplet-mural-data/seq1 --label_nc 0 --no_instance --niter 100 --save_epoch_freq 20 --fp16
```

---

## 4. 注意事项

### 4.1 显存限制
- **T4 (16GB)**：推荐使用 `--netG global` 或 `--netG unet`
- **P100/V100 (32GB)**：可以使用 `global_ca_aspp`

### 4.2 常见错误
- `CUDA out of memory` → 减少 `--batchSize` 或使用 `--fp16`
- `out of storage` → 清理不必要的文件

### 4.3 训练后查看
训练完成后，权重会保存在：
`/kaggle/working/checkpoints/mural_kaggle/`

---

## 5. 推荐训练配置（Kaggle）

```python
!python train.py \
  --name mural_kaggle \
  --netG global_ca_aspp \
  --dataroot /kaggle/input/Incomplet-mural-data/seq1 \
  --label_nc 0 --no_instance \
  --niter 100 --niter_decay 50 \
  --save_epoch_freq 20 \
  --display_freq 50 \
  --print_freq 20 \
  --batchSize 1 \
  --fp16
```

---

**建议你直接在 Kaggle Notebook 中运行这个命令：**

```python
!python train.py --name mural_kaggle --netG global_ca_aspp --dataroot /kaggle/input/Incomplet-mural-data/seq1 --label_nc 0 --no_instance --niter 100 --save_epoch_freq 20 --fp16
```

需要我帮你：
1. **生成完整的 Kaggle Notebook 模板**（包含所有步骤）
2. **调整训练参数**（缩短时间 / 减小显存）
3. **写推理代码**（如何在 Kaggle 上使用训练好的权重）

告诉我你的需求。