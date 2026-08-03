# 壁画修复模型训练指南

**版本**：1.0  
**适用架构**：`global_ca_aspp`、`global`、`unet`、`line_dist`

---

## 1. 环境准备（强烈推荐使用 Conda）

```bash
# 创建 Conda 环境
conda create -n mural-restoration python=3.10 -y
conda activate mural-restoration

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install dominate scipy numpy pillow

# 安装 AMP（可选，但推荐）
pip install apex
```

---

## 2. 数据集准备

项目已有现成数据集：

```bash
# 确认数据集路径
ls datasets/Incomplet-mural-data/seq1/
```

如果没有数据集，请参考原始 pix2pixHD 的数据集准备方式（Cityscapes 标签图 → 街景图）。

---

## 3. 推荐训练命令

### 首推配置（效果最佳）
```bash
# 推荐：global_ca_aspp（CoordAtt + ASPP）
python train.py \
  --name mural_ca_aspp \
  --netG global_ca_aspp \
  --dataroot ./datasets/Incomplet-mural-data/seq1 \
  --label_nc 0 --no_instance \
  --niter 200 --niter_decay 100 \
  --save_epoch_freq 50 \
  --display_freq 100 \
  --print_freq 100 \
  --batchSize 1
```

### 其他常用配置

| 架构          | 命令 |
|---------------|------|
| 全局生成器（默认） | `--netG global` |
| UNet          | `--netG unet` |
| 线稿专用      | `--netG line_dist` |
| ASPP 版       | `--netG global_aspp` |
| 注意力+ASPP   | `--netG global_ca_aspp` |

---

## 4. 训练脚本（推荐使用）

直接使用项目提供的脚本：

```bash
# 512p 训练（推荐）
bash scripts/train_512p_fp16.sh

# 或自定义训练
bash scripts/train_512p.sh
```

---

## 5. 训练过程中查看进度

```bash
# 查看当前损失
tail -f checkpoints/mural_ca_aspp/loss_log.txt

# 查看生成图像
ls checkpoints/mural_ca_aspp/web/images/
```

---

## 6. 训练完成后查看权重

训练结束后会在 `checkpoints/mural_ca_aspp/` 下生成：

- `latest_net_G.pth`
- `latest_net_D.pth`
- `200_net_G.pth`（如果每 50 轮保存）

---

## 7. 注意事项

1. **GPU 显存需求**
   - 512p：至少 12GB 显存
   - 1024p：至少 24GB 显存（推荐使用 AMP）

2. **推荐使用 AMP 加速**
   ```bash
   # 添加 --fp16 参数
   python train.py --fp16 ...
   ```

3. **如果内存不足**
   - 使用 `--resize_or_crop scale_width`（默认）
   - 或使用 12G 显存版脚本

4. **多 GPU 训练**
   ```bash
   python -m torch.distributed.launch train.py --gpu_ids 0,1,2,3 --batchSize 4 ...
   ```

---

## 8. 完整训练命令（推荐）

```bash
# 最推荐的训练命令
python train.py \
  --name mural_final \
  --netG global_ca_aspp \
  --dataroot ./datasets/Incomplet-mural-data/seq1 \
  --label_nc 0 --no_instance \
  --niter 200 --niter_decay 100 \
  --save_epoch_freq 50 \
  --display_freq 200 \
  --print_freq 100 \
  --fp16
```

---

**训练开始前请确认**：
- 环境已激活 `mural-restoration`
- 数据集路径正确
- GPU 显存充足

需要我帮你**生成完整的训练配置模板**或**调试训练过程中的问题**吗？