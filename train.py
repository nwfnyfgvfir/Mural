from copy import deepcopy
import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import math
def lcm(a, b): return abs(a * b) // math.gcd(a, b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import pytorch_ssim


def binary_iou(preds, targets, threshold=0.5, smooth=1e-6):
    """
    输入形状: (B, 1, H, W)
    - preds:    预测边缘图 (B, 1, H, W) ∈ {0,1}
    - targets:  真实边缘图 (B, 1, H, W) ∈ {0,1}
    """
    
    # 展平张量以简化计算 (B, 1, H, W) -> (B, H*W)
    preds_flat = preds.view(preds.shape[0], -1)
    targets_flat = targets.view(targets.shape[0], -1)
    
    # 计算交集与并集
    intersection = (preds_flat * targets_flat).sum(dim=1)  # (B,)
    union = (preds_flat + targets_flat).sum(dim=1) - intersection  # (B,)
    
    # 计算每个样本的IoU并求批次平均
    iou_per_sample = (intersection + smooth) / (union + smooth)
    batch_iou = iou_per_sample.mean()  # 标量
    
    return batch_iou



def f1_score(preds, targets, smooth=1e-6):
    """
    计算二值边缘图的F1分数
    输入: 
        preds   : 预测边缘图 (B, 1, H, W) ∈ {0,1}
        targets : 真实边缘图 (B, 1, H, W) ∈ {0,1}
        smooth  : 平滑系数防除零
    输出: 
        f1     : 批次平均F1分数 (标量)
    """
    # 展平空间维度 (B, 1, H, W) -> (B, H*W)
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    
    # 计算TP/FP/FN
    tp = (preds_flat * targets_flat).sum(dim=1)        # 真正例：预测为边缘且正确
    fp = (preds_flat * (1 - targets_flat)).sum(dim=1)  # 假正例：预测为边缘但错误
    fn = ((1 - preds_flat) * targets_flat).sum(dim=1)  # 假反例：漏检边缘
    
    # 计算Precision与Recall
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    
    # 计算F1分数（调和平均）
    f1_per_sample = 2 * (precision * recall) / (precision + recall + smooth)
    return f1_per_sample.mean()  # 批次平均


opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
# fp16 disabled due to CUDA compatibility and missing apex
model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

metric_results_list = []
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    metric_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'iou': 0, 'iou_sum': 0, 'batch_sizes': 0}
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        losses, generated, (fake_image, real_image) = model(Variable(data['label']), Variable(data['inst']), 
            Variable(data['image']), Variable(data['feat']), infer=save_fake, metrics=True)

        # 计算指标
        fake_image = (fake_image + 1) / 2
        real_image = (real_image + 1) / 2
        fake_image_bin = fake_image.mean(dim=1, keepdim=True)
        fake_image_bin_t = fake_image_bin.clone()
        fake_image_bin[fake_image_bin_t < 0.5] = 1
        fake_image_bin[fake_image_bin_t >= 0.5] = 0
        real_image_bin = real_image.mean(dim=1, keepdim=True)
        real_image_bin_t = real_image_bin.clone()
        real_image_bin[real_image_bin_t < 0.5] = 1
        real_image_bin[real_image_bin_t >= 0.5] = 0
        batch_mse = ((fake_image - real_image) ** 2).data.mean()
        metric_results['batch_sizes'] += opt.batchSize
        metric_results['mse'] += batch_mse * opt.batchSize
        batch_ssim = pytorch_ssim.ssim(fake_image, real_image).item()
        metric_results['ssims'] += batch_ssim * opt.batchSize
        metric_results['psnr'] = 10 * math.log10((1**2) / (metric_results['mse'] / metric_results['batch_sizes']))
        metric_results['ssim'] = metric_results['ssims'] / metric_results['batch_sizes']
        metric_results['iou_sum'] += binary_iou(fake_image_bin, real_image_bin).item() * opt.batchSize
        metric_results['iou'] = metric_results['iou_sum'] / metric_results['batch_sizes']

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)

        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        if opt.fp16:                                
            with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: scaled_loss.backward()                
        else:
            loss_G.backward()          
        optimizer_G.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        if opt.fp16:                                
            with amp.scale_loss(loss_D, optimizer_D) as scaled_loss: scaled_loss.backward()                
        else:
            loss_D.backward()        
        optimizer_D.step()        

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
            t = (time.time() - iter_start_time) / opt.print_freq
            errors_metrics = deepcopy(errors)
            errors_metrics.update({
                "PSNR": metric_results['psnr'], 
                "SSIM": metric_results['ssim'],
                "IOU":metric_results['iou']
                })
            visualizer.print_current_errors(epoch, epoch_iter, errors_metrics, t)
            visualizer.plot_current_errors(errors, total_steps)
            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

        ### display output images
        if save_fake:
            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                   ('synthesized_image', util.tensor2im(generated.data[0])),
                                   ('real_image', util.tensor2im(data['image'][0]))])
            visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break
    
    metric_results_list.append([metric_results['psnr'], metric_results['ssim'], metric_results['iou']])
    metric_results_list_np = np.array(metric_results_list)
    metric_path = os.path.join(opt.checkpoints_dir, opt.name, 'metrics.txt')
    np.savetxt(metric_path, metric_results_list_np, fmt="%.4f")
       
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
