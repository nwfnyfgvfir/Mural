# 壁画修复模型（Mural Restoration Model）

**版本**：1.0  
**基础**：pix2pixHD（NVIDIA）  
**修改目标**：针对壁画不完整 → 完整修复任务进行深度定制

---

## 1. 项目概述

本项目是基于 **pix2pixHD**（高分辨率图像到图像翻译的条件 GAN）开发的**壁画修复模型**。核心任务是将**不完整/损坏的壁画图片**修复为**完整版壁画图片**，属于图像到图像的修复任务。

- **原始仓库**：NVIDIA/pix2pixHD
- **主要贡献**：
  - 针对壁画场景深度定制生成器架构
  - 加入壁画边缘保真度评估指标
  - 支持交互式壁画编辑
  - 提供 TensorRT/ONNX 推理优化

---

## 2. 核心功能

- **壁画修复**：不完整壁画 → 完整壁画（RGB → RGB 修复）
- **线稿/边缘修复**：专门的 `LineDist` 网络
- **交互式编辑**：点击改变物体、添加笔刷、换风格
- **多分辨率训练**：支持 512p / 1024p
- **推理加速**：TensorRT / ONNX 导出

---

## 3. 技术架构

### 3.1 生成器（Generator）—— 核心自定义部分

`models/networks.py` 中定义了以下生成器：

| 生成器名称       | 主要特点                              | 适用场景                  |
|------------------|---------------------------------------|---------------------------|
| `GlobalGenerator` | 基础全局生成器，可选 CoordAtt + ASPP | 通用修复                  |
| `GlobalGenerator`（+CA/ASPP） | 坐标注意力 + 空洞卷积                  | 壁画细致纹理              |
| `LocalEnhancer`   | 局部增强器                            | 细节修复                  |
| `UNet`            | 经典 U-Net                            | 简单修复                  |
| `LineDist`        | 残差编码器 + 下采样                   | 线稿/边缘图修复           |

### 3.2 判别器（Discriminator）
- Multiscale PatchGAN（默认 2 个尺度）

### 3.3 损失函数
- LSGAN（默认）
- GAN 特征匹配
- VGG 感知损失
- **自定义指标**（`train.py` 新增）：
  - PSNR / SSIM（图像质量）
  - 二值边缘 IoU / F1（壁画线条保真度）

---

## 4. 数据集

### 训练集
- 路径：`datasets/Incomplet-mural-data/seq1/`
- 包含 100 组不完整 → 完整壁画图片（`train_A` → `train_B`）

### 测试集
- 路径：`datasets/Incomplet-mural-data/test/`
- 34 张不完整壁画图片

### 使用方式
```bash
python train.py --label_nc 0 --no_instance --dataroot ./datasets/Incomplet-mural-data/ ...
```

---

## 5. 安装与运行

### 依赖
- PyTorch + CUDA
- Python 2/3
- dominate（可视化）

### 基本运行
```bash
# 训练
python train.py --name mural_global --netG global --dataroot ./datasets/Incomplet-mural-data/

# 测试
python test.py --name mural_global --which_epoch 200 --netG global --dataroot ./datasets/Incomplet-mural-data/test
```

---

## 6. 交互式使用（UI 模式）

```bash
python test.py --name mural_global --which_epoch 200 --netG global --model UIModel ...
```

进入 UI 后支持：
- 点击改变物体
- 添加笔刷
- 换风格生成

---

## 7. 训练与推理命令

### 推荐训练配置
```bash
python train.py --label_nc 0 --no_instance --dataroot ./datasets/Incomplet-mural-data/ \
  --netG global_ca_aspp --niter 200 --save_epoch_freq 50
```

### 推理命令
```bash
# 使用 200 epoch 权重
python test.py --name . --which_epoch 200 --netG global \
  --dataroot ./datasets/Incomplet-mural-data/test \
  --checkpoints_dir checkpoints --no_instance --label_nc 0
```

---

## 8. 现有权重说明

当前项目仅在 `checkpoints/` 根目录保存了权重：

- `200_net_G.pth` （推荐）
- `latest_net_G.pth`
- 对应 `*_net_D.pth`

`unet/`、`line_dist/` 等子目录**只有可视化结果**，无独立权重。

---

## 9. 技术债务与注意事项

- `base_model.py` 中 `update_learning_rate` 缺少 `self`
- 部分代码仍使用旧 PyTorch API（`Variable`、`volatile`）
- `opt.name` 为空 → 权重保存在 `checkpoints/` 根目录
- `run_engine.py` 有多个未定义变量

---

## 10. 未来优化方向

1. 为不同架构分别保存权重
2. 清理已弃用的代码
3. 加入真实壁画超大分辨率数据集
4. 增加纹理增强和风格迁移功能
5. 完善 UI 交互体验

---

**作者**：Claude Code  
**日期**：2026-08-03

---

**参考文献**：
- 原 pix2pixHD 论文：https://arxiv.org/pdf/1711.11585.pdf
- 壁画修复视觉效果示例：（需补充图片）

---

**使用建议**：
建议先用 `200_net_G.pth` 验证效果，再在 `global_ca_aspp` 或 `LineDist` 架构上进行进一步训练。
