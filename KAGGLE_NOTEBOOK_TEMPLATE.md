# Kaggle Notebook 完整训练模板（壁画修复模型）

**复制下面所有代码到 Kaggle Notebook 即可直接运行**

---

## 第一步：安装依赖

```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install dominate scipy numpy pillow
!pip install apex
```

---

## 第二步：上传代码和数据集

1. 上传所有 `.py` 文件到 `/kaggle/working/`
2. 上传数据集（`Incomplet-mural-data.zip`）到 `/kaggle/input/`

---

## 第三步：启动训练（推荐命令）

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

## 第四步：训练过程中查看进度

```python
# 查看损失日志
!tail -f /kaggle/working/checkpoints/mural_kaggle/loss_log.txt

# 查看生成图像
!ls /kaggle/working/checkpoints/mural_kaggle/web/images/
```

---

## 第五步：训练完成后查看权重

```python
!ls /kaggle/working/checkpoints/mural_kaggle/
```

---

## 第六步：推理代码（测试训练好的权重）

```python
!python test.py \
  --name mural_kaggle \
  --which_epoch 100 \
  --netG global_ca_aspp \
  --dataroot /kaggle/input/Incomplet-mural-data/test \
  --checkpoints_dir /kaggle/working/checkpoints \
  --label_nc 0 --no_instance \
  --how_many 10
```

---

## 第七步：交互式 UI 测试

```python
!python test.py \
  --name mural_kaggle \
  --which_epoch 100 \
  --netG global_ca_aspp \
  --model UIModel \
  --dataroot /kaggle/input/Incomplet-mural-data/test \
  --checkpoints_dir /kaggle/working/checkpoints \
  --label_nc 0 --no_instance
```

---

**完整一键训练命令（推荐直接复制使用）：**

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

**使用说明**：
1. 把上面所有代码块**依次复制到 Kaggle Notebook**
2. 运行后等待训练完成
3. 训练完成后，权重保存在 `/kaggle/working/checkpoints/mural_kaggle/`

需要我帮你：
- 添加 TensorBoard 记录
- 增加多阶段训练（先 512p 再 1024p）
- 写推理 + UI 的完整脚本

告诉我你的需求。