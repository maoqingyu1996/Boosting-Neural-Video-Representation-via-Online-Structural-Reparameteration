import math
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim

def quantize_per_tensor(t, bit=8, axis=-1):
    """
    对输入张量 t 进行逐张量量化，返回 (量化后的整数值, 反量化后的近似值)
    参数:
        t   : 输入张量
        bit : 量化位宽，默认 8 位
        axis: 指定沿哪个维度做“逐通道”量化
              -1 → 全局量化（整个张量共享 scale/zero-point）
               0 → 按第 0 维（如 batch 维）分别量化
               1 → 按第 1 维（如 channel 维）分别量化
    """
    if axis == -1:
        # 全局量化：先排除 0 值，再计算最小/最大
        t_valid = t != 0
        t_min, t_max = t[t_valid].min(), t[t_valid].max()
        scale = (t_max - t_min) / 2**bit          # 量化步长
    elif axis == 0:
        # 按 axis=0（如 batch）逐条量化：每条单独统计 min/max
        min_max_list = []
        for i in range(t.size(0)):
            t_valid = t[i] != 0
            if t_valid.sum():
                min_max_list.append([t[i][t_valid].min(), t[i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)
        scale = (min_max_tf[:, 1] - min_max_tf[:, 0]) / 2**bit
        # 将 scale 与 zero-point 扩展成与 t 相同的形状，便于广播
        if t.dim() == 4:
            scale = scale[:, None, None, None]
            t_min = min_max_tf[:, 0, None, None, None]
        elif t.dim() == 2:
            scale = scale[:, None]
            t_min = min_max_tf[:, 0, None]
    elif axis == 1:
        # 按 axis=1（如 channel）逐条量化
        min_max_list = []
        for i in range(t.size(1)):
            t_valid = t[:, i] != 0
            if t_valid.sum():
                min_max_list.append([t[:, i][t_valid].min(), t[:, i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)
        scale = (min_max_tf[:, 1] - min_max_tf[:, 0]) / 2**bit
        if t.dim() == 4:
            scale = scale[None, :, None, None]
            t_min = min_max_tf[None, :, 0, None, None]
        elif t.dim() == 2:
            scale = scale[None, :]
            t_min = min_max_tf[None, :, 0]

    # 线性量化：先减 zero-point，再除以 scale，四舍五入取整
    quant_t = ((t - t_min) / (scale + 1e-19)).round()
    # 反量化：整数值 * scale + zero-point
    new_t = t_min + scale * quant_t
    return quant_t, new_t
    
def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def all_reduce(tensors, average=True):
    """
    All reduce the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all reduce across all processes in
        all machines.
        average (bool): scales the reduced tensor by the number of overall
        processes across all machines.
    """

    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


class PositionalEncoding(nn.Module):
    def __init__(self, pe_embed):
        super(PositionalEncoding, self).__init__()
        self.pe_embed = pe_embed.lower()
        if self.pe_embed == 'none':
            self.embed_length = 1
        else:
            self.lbase, self.levels = [float(x) for x in pe_embed.split('_')]
            self.levels = int(self.levels)
            self.embed_length = 2 * self.levels

    def forward(self, pos):
        if self.pe_embed == 'none':
            return pos[:,None]
        else:
            pe_list = []
            for i in range(self.levels):
                temp_value = pos * self.lbase **(i) * math.pi
                pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
            return torch.stack(pe_list, 1)


def psnr2(img1, img2):
    mse = (img1 - img2) ** 2
    PIXEL_MAX = 1
    psnr = -10 * torch.log10(mse)
    psnr = torch.clamp(psnr, min=0, max=50)
    return psnr

def loss_fn(pred, target, args):
    target = target.detach()

    if args.loss_type == 'L2':
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.mean()       
    elif args.loss_type == 'L1':
        loss = torch.mean(torch.abs(pred - target))
    elif args.loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    elif args.loss_type == 'Fusion1':
        loss = 0.3 * F.mse_loss(pred, target) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion2':
        loss = 0.3 * torch.mean(torch.abs(pred - target)) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion3':
        loss = 0.5 * F.mse_loss(pred, target) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion4':
        loss = 0.5 * torch.mean(torch.abs(pred - target)) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion5':
        loss = 0.7 * F.mse_loss(pred, target) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion6':
        loss = 0.7 * torch.mean(torch.abs(pred - target)) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion7':
        loss = 0.7 * F.mse_loss(pred, target) + 0.3 * torch.mean(torch.abs(pred - target))
    elif args.loss_type == 'Fusion8':
        loss = 0.5 * F.mse_loss(pred, target) + 0.5 * torch.mean(torch.abs(pred - target))
    elif args.loss_type == 'Fusion9':
        loss = 0.9 * torch.mean(torch.abs(pred - target)) + 0.1 * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion10':
        loss = 0.7 * torch.mean(torch.abs(pred - target)) + 0.3 * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion11':
        loss = 0.9 * torch.mean(torch.abs(pred - target)) + 0.1 * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion12':
        loss = 0.8 * torch.mean(torch.abs(pred - target)) + 0.2 * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif args.loss_type == 'Fusion13':
        pred_freq = torch.fft.fft2(pred, dim=(-2, -1))
        pred_freq = torch.stack([pred_freq.real, pred_freq.imag], -1)
        target_freq = torch.fft.fft2(target, dim=(-2, -1))
        target_freq = torch.stack([target_freq.real, target_freq.imag], -1)
        freq_loss = F.l1_loss(pred_freq, target_freq, reduction='none').flatten(1).mean(1)
        loss = 60 * (0.7 * torch.mean(torch.abs(pred - target)) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=True))) + \
               freq_loss
    elif args.loss_type == 'Fusion15':
        pred_freq = torch.fft.fft2(pred, dim=(-2, -1))
        pred_freq = torch.stack([pred_freq.real, pred_freq.imag], -1)
        target_freq = torch.fft.fft2(target, dim=(-2, -1))
        target_freq = torch.stack([target_freq.real, target_freq.imag], -1)
        freq_loss = F.l1_loss(pred_freq, target_freq, reduction='none').flatten(1).mean(1)
        loss = 60*(0.7 * torch.mean(torch.abs(pred - target)) + 0.3 * (1 - ms_ssim(pred, target, data_range=1, size_average=True))) + \
               freq_loss
    return loss

def psnr_fn(output_list, target_list):
    psnr_list = []
    for output, target in zip(output_list, target_list):
        l2_loss = F.mse_loss(output.detach(), target.detach(), reduction='mean')
        psnr = -10 * torch.log10(l2_loss)
        psnr = psnr.view(1, 1).expand(output.size(0), -1)
        psnr_list.append(psnr)
    psnr = torch.cat(psnr_list, dim=1) #(batchsize, num_stage)
    return psnr

def msssim_fn(output_list, target_list):
    msssim_list = []
    for output, target in zip(output_list, target_list):
        if output.size(-2) >= 160:
            msssim = ms_ssim(output.float().detach(), target.detach(), data_range=1, size_average=True)
        else:
            msssim = torch.tensor(0).to(output.device)
        msssim_list.append(msssim.view(1))
    msssim = torch.cat(msssim_list, dim=0) #(num_stage)
    msssim = msssim.view(1, -1).expand(output_list[-1].size(0), -1) #(batchsize, num_stage)
    return msssim

def RoundTensor(x, num=2, group_str=False):
    """
    将张量 x 的元素按指定位数四舍五入后转为字符串。
    参数:
        x        : 输入张量
        num      : 保留的小数位数，默认 2
        group_str: 是否按行分组输出
                   True  → 每行元素用逗号拼接，行间用斜杠分隔
                   False → 所有元素展平后用逗号拼接
    返回:
        out_str  : 拼接后的字符串
    """
    if group_str:
        str_list = []
        for i in range(x.size(0)):
            # 对第 i 行元素逐个四舍五入并转为字符串
            x_row =  [str(round(ele, num)) for ele in x[i].tolist()]
            # 行内用逗号连接
            str_list.append(','.join(x_row))
        # 行间用斜杠连接
        out_str = '/'.join(str_list)
    else:
        # 展平张量后逐个四舍五入并转为字符串，再用逗号连接
        str_list = [str(round(ele, num)) for ele in x.flatten().tolist()]
        out_str = ','.join(str_list)
    return out_str

def adjust_lr(optimizer, cur_epoch, cur_iter, data_size, args):
    cur_epoch = cur_epoch + (float(cur_iter) / data_size)
    if args.lr_type == 'cosine':
        lr_mult = 0.5 * (math.cos(math.pi * (cur_epoch - args.warmup)/ (args.epochs - args.warmup)) + 1.0)
    elif args.lr_type == 'step':
        lr_mult = 0.1 ** (sum(cur_epoch >= np.array(args.lr_steps)))
    elif args.lr_type == 'const':
        lr_mult = 1
    elif args.lr_type == 'plateau':
        lr_mult = 1
    else:
        raise NotImplementedError

    if cur_epoch < args.warmup:
        lr_mult = 0.1 + 0.9 * cur_epoch / args.warmup

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = args.lr * lr_mult

    return args.lr * lr_mult

def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return

def split_channels(num_channels, num_splits=6):
    base_size = num_channels // num_splits
    remain = num_channels % num_splits
    split_sizes = [base_size] * num_splits

    for i in range(remain):
        split_sizes[i] += 1

    return split_sizes

class PositionalEncodingTrans(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, pos):
        index = torch.round(pos * self.max_len).long()
        p = self.pe[index]
        return p
