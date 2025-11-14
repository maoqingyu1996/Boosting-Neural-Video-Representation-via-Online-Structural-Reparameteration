from __future__ import print_function

import argparse
import os
import random
import shutil
from datetime import datetime
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from thop import profile

from model import CustomDataSet, Generator, NeRVBlock
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

    # dataset parameters
    parser.add_argument('--vid',  default=[None], type=int,  nargs='+', help='video id list for training')
    parser.add_argument('--scale', type=int, default=1, help='scale-up facotr for data transformation,  added to suffix!!!!')
    parser.add_argument('--frame_gap', type=int, default=1, help='frame selection gap')
    parser.add_argument('--augment', type=int, default=0, help='augment frames between frames,  added to suffix!!!!')
    parser.add_argument('--dataset', type=str, default='UVG', help='dataset',)
    parser.add_argument('--test_gap', default=1, type=int, help='evaluation gap')

    # NERV architecture parameters
    # embedding parameters
    parser.add_argument('--embed', type=str, default='1.25_80', help='base value/embed length for position encoding')

    # FC + Conv parameters
    parser.add_argument('--stem_dim_num', type=str, default='1024_1', help='hidden dimension and length')
    parser.add_argument('--fc_hw_dim', type=str, default='9_16_128', help='out size (h,w) for mlp')
    parser.add_argument('--expansion', type=float, default=8, help='channel expansion from fc to conv')
    parser.add_argument('--reduction', type=int, default=2)
    parser.add_argument('--strides', type=int, nargs='+', default=[5, 3, 2, 2, 2], help='strides list')
    parser.add_argument('--num_blocks', type=int, default=1)

    parser.add_argument('--norm', default='none', type=str, help='norm layer for generator', choices=['none', 'bn', 'in'])
    parser.add_argument('--act', type=str, default='gelu', help='activation to use', choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])
    parser.add_argument('--lower_width', type=int, default=32, help='lowest channel width for output feature maps')
    parser.add_argument("--single_res", action='store_true', help='single resolution,  added to suffix!!!!')
    parser.add_argument("--conv_type", default='conv', type=str,  help='upscale methods, can add bilinear and deconvolution methods', choices=['conv', 'deconv', 'bilinear'])
    parser.add_argument("--branch_type", default='NeRV_vanilla', type=str,  help='branch type', choices=['NeRV_vanilla', 'ERB', 'ACB', 'RepVGG', 'DBB', 'ECB'])

    # General training setups
    parser.add_argument('-j', '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--not_resume_epoch', action='store_true', help='resuming start_epoch from checkpoint')
    parser.add_argument('-e', '--epochs', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--cycles', type=int, default=1, help='epoch cycles for training')
    parser.add_argument('--warmup', type=float, default=0.2, help='warmup epoch ratio compared to the epochs, default=0.2,  added to suffix!!!!')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
    parser.add_argument('--lr_type', type=str, default='cosine', help='learning rate type, default=cosine')
    parser.add_argument('--lr_steps', default=[], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10,  added to suffix!!!!')
    parser.add_argument('--beta', type=float, default=0.5, help='beta for adam. default=0.5,  added to suffix!!!!')
    parser.add_argument('--loss_type', type=str, default='L2', help='loss type, default=L2')
    parser.add_argument('--lw', type=float, default=1.0, help='loss weight,  added to suffix!!!!')
    parser.add_argument('--sigmoid', action='store_true', help='using sigmoid for output prediction')

    # evaluation parameters
    parser.add_argument('--deploy', action='store_true', default=False, help='whether to reparam')
    parser.add_argument('--eval_only', action='store_true', default=False, help='do evaluation only')
    parser.add_argument('--eval_freq', type=int, default=50, help='evaluation frequency,  added to suffix!!!!')
    parser.add_argument('--quant_bit', type=int, default=-1, help='bit length for model quantization')
    parser.add_argument('--quant_axis', type=int, default=0, help='quantization axis (-1 means per tensor)')
    parser.add_argument('--dump_images', action='store_true', default=False, help='dump the prediction images')
    parser.add_argument('--eval_fps', action='store_true', default=False, help='fwd multiple times to test the fps ')

    # pruning paramaters
    parser.add_argument('--prune_steps', type=float, nargs='+', default=[0.,], help='prune steps')
    parser.add_argument('--prune_ratio', type=float, default=1.0, help='pruning ratio')

    # distribute learning parameters
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:9888', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('-d', '--distributed', action='store_true', default=False, help='distributed training,  added to suffix!!!!')

    # logging, output directory, 
    parser.add_argument('--debug', action='store_true', help='defbug status, earlier for train/eval')  
    parser.add_argument('-p', '--print_freq', default=50, type=int,)
    parser.add_argument('--weight', default='None', type=str, help='pretrained weights for ininitialization')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the output dir if already exists')
    parser.add_argument('--outf', default='unify', help='folder to output images and model checkpoints')
    parser.add_argument('--suffix', default='', help="suffix str for outf")
    parser.add_argument('--finetune', action='store_true', default=False, help="whether to finetune after pruning")
    parser.add_argument('--finetune_epochs', type=int, default=100, help='number of training epcohs after pruning and fine-tuning')

    args = parser.parse_args()
        
    args.warmup = int(args.warmup * args.epochs)
    
    # 评估模式时，评估频率设为1，输出目录设为RESULT/debug
    if args.debug:
        args.eval_freq = 1
        args.outf = 'result/debug'
    else:
        args.outf = os.path.join('result', args.outf)  ### output directory
        
    # 输出文件名称字符串
    if args.prune_ratio < 1 and not args.eval_only: 
        prune_str = '_Prune{}_{}'.format(args.prune_ratio, ','.join([str(x) for x in args.prune_steps]))
    else:
        prune_str = ''
    extra_str = '_Strd{}_{}Res{}{}'.format( ','.join([str(x) for x in args.strides]),  'Sin' if args.single_res else f'_lw{args.lw}_multi',  
            '_dist' if args.distributed else '', f'_eval' if args.eval_only else '')
    norm_str = '' if args.norm == 'none' else args.norm
    
    exp_id = f'{args.dataset}/embed{args.embed}_{args.stem_dim_num}_fc_{args.fc_hw_dim}__exp{args.expansion}_reduce{args.reduction}_low{args.lower_width}_blk{args.num_blocks}_cycle{args.cycles}' + \
            f'_gap{args.frame_gap}_e{args.epochs}_warm{args.warmup}_b{args.batchSize}_{args.conv_type}_lr{args.lr}_{args.lr_type}' + \
            f'_{args.loss_type}{norm_str}{extra_str}{prune_str}'
    
    exp_id += f'_act{args.act}_{args.suffix}'
    args.exp_id = exp_id
    exp_id_simplied = f'{args.suffix}'
    args.outf = os.path.join(args.outf, exp_id_simplied)
    
    # 创建输出文件目录
    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)
        
    port = hash(args.exp_id) % 20000 + 10000
    args.init_method =  f'tcp://127.0.0.1:{port}'
    print(f'init_method: {args.init_method}', flush=True)
    
    # 设置随机种子和分布式训练
    torch.set_printoptions(precision=2)
    local_rank = torch.cuda.current_device() if torch.cuda.is_available() else 0
    eval(local_rank, args)

def eval(local_rank, args):
    # 确认并设置当前 CUDA 设备，避免隐式使用非期望设备导致张量分散（中文注释）
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    # 固定随机种子确保可复现性
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)
    
    # 初始化训练指标和两个标志位
    train_best_psnr, train_best_msssim, val_best_psnr, val_best_msssim = [torch.tensor(0) for _ in range(4)]
    is_train_best, is_val_best = False, False
    
    # 初始化位置编码和模型
    # ERB 分支条件下且不需要微调模型，直接设置为部署模式
    # 其他情况保持训练模式即可测试
    if args.branch_type == 'ERB' and not args.finetune: 
        args.deploy = True
        PE = PositionalEncoding(args.embed)
        args.embed_length = PE.embed_length
        model = Generator(embed_length=args.embed_length, stem_dim_num=args.stem_dim_num, fc_hw_dim=args.fc_hw_dim, expansion=args.expansion, 
                          num_blocks=args.num_blocks, norm=args.norm, act=args.act, bias = True, reduction=args.reduction, conv_type=args.conv_type,
                      stride_list=args.strides, sin_res=args.single_res, lower_width=args.lower_width, sigmoid=args.sigmoid, 
                      deploy=args.deploy, branch_type=args.branch_type)
        
    # 其他情况保持训练模式
    else:
        args.deploy = False
        PE = PositionalEncoding(args.embed)
        args.embed_length = PE.embed_length
        model = Generator(embed_length=args.embed_length, stem_dim_num=args.stem_dim_num, fc_hw_dim=args.fc_hw_dim, expansion=args.expansion, 
                          num_blocks=args.num_blocks, norm=args.norm, act=args.act, bias = True, reduction=args.reduction, conv_type=args.conv_type,
                      stride_list=args.strides, sin_res=args.single_res, lower_width=args.lower_width, sigmoid=args.sigmoid, 
                      deploy=args.deploy, branch_type=args.branch_type)
    
    # 打印模型初始化信息：分支结构、参数量、部署/训练状态、是否需要微调
    info_str = (
        f"初始化模型分支结构: {args.branch_type}\n"
        f"模型总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M\n"
        f"是否需要微调训练: {'是' if args.finetune else '否'}\n"
    )
    print(info_str, end='')

    # 设置数据加载器
    img_transforms = transforms.ToTensor()
    DataSet = CustomDataSet
    train_data_dir = f'../data/{args.dataset.lower()}' ### dataset path
    val_data_dir = f'../data/{args.dataset.lower()}'
        
    # 训练数据集
    train_dataset = DataSet(train_data_dir, img_transforms, vid_list=args.vid, frame_gap=args.frame_gap)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=(train_sampler is None),
                                                num_workers=args.workers, pin_memory=True, 
                                                sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn)

    # 测试数据集
    val_dataset = DataSet(val_data_dir, img_transforms, vid_list=args.vid, frame_gap=args.test_gap)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if args.distributed else None
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False,
                                                num_workers=args.workers, pin_memory=True, 
                                                sampler=val_sampler, drop_last=False, worker_init_fn=worker_init_fn)
    data_size = len(train_dataset)
        
    # *! 如果需要剪枝微调，直接加载最新模型
    # *! 需要微调时的模型加载--》剪枝--》训练--》重参到部署态
    prune_net = args.prune_ratio < 1 
    if args.finetune and prune_net:
        # 确认模型路径
        checkpoint_path = os.path.join(args.outf, 'model_latest.pth')  
        
        # 确认模型文件存在
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"待加载的模型文件不存在: {checkpoint_path}")

        # 优先使用安全加载模式，兼容旧版 PyTorch
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except (TypeError, AttributeError):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 清理非权重项（若存在）
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            keys_to_remove = [key for key in state_dict.keys() if 'total_ops' in key or 'total_params' in key]
            for key in keys_to_remove:
                del state_dict[key]
            checkpoint['state_dict'] = state_dict
            
        # 加载模型权重
        model.load_state_dict(state_dict, strict=False)
        info_str += f"已加载微调前训练态模型权重: {checkpoint_path}, 分支类型为 {args.branch_type}\n"
        print(f"已加载微调前训练态模型权重: {checkpoint_path}, 分支类型为 {args.branch_type}")
        
        # *! NeRV与RepNeRV的剪枝方式不同，再次区分处理
        if args.branch_type == 'NeRV_vanilla': 
            param_list = []
            for k, v in model.named_parameters():
                if 'weight' in k:
                    if 'stem' in k:
                        # stem 是 MLP 的若干 Linear 层，这里按索引加入对应 Linear 模块用于剪枝
                        stem_ind = int(k.split('.')[1])
                        param_list.append(model.stem[stem_ind])
                        info_str += f"添加 stem 层 {stem_ind} 的 Linear(权重)到剪枝列表\n"
                        print(f"添加 stem 层 {stem_ind} 的 Linear(权重)到剪枝列表")
                    elif 'layers' in k[:6]:
                        # 针对 NeRVBlock：训练态使用 branch，部署态使用 rbr_reparam
                        layer_ind = int(k.split('.')[1])
                        layer = model.layers[layer_ind]
                        target = None
                        if hasattr(layer, 'branch'):
                            target = layer.branch
                            info_str += f"NeRV_vanilla 训练态：添加第 {layer_ind} 层 branch 卷积到剪枝列表\n"
                            print(f"NeRV_vanilla 训练态：添加第 {layer_ind} 层 branch 卷积到剪枝列表")
                        elif hasattr(layer, 'rbr_reparam'):
                            target = layer.rbr_reparam
                            print(f"NeRV_vanilla 部署态：添加第 {layer_ind} 层 rbr_reparam 卷积到剪枝列表")
                        if target is not None:
                            param_list.append(target)
            param_to_prune = [(ele, 'weight') for ele in param_list]
            prune_base_ratio = args.prune_ratio
            
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=prune_base_ratio,
            )
            
            # 检查剪枝是否生效：统计所有被剪枝模块的掩码零元素比例（越接近设定比例越好）
            mask_modules = [m for (m, name) in param_to_prune if hasattr(m, 'weight_mask')]
            total_mask_elems = sum(m.weight_mask.numel() for m in mask_modules)
            zero_mask_elems = sum((m.weight_mask == 0).sum().item() for m in mask_modules)
            actual_ratio = (zero_mask_elems / total_mask_elems) if total_mask_elems > 0 else 0.0
            if total_mask_elems == 0:
                # 若未发现 weight_mask，说明剪枝 reparam 可能未注入或模块选择不当
                print("警告：未检测到 weight_mask，剪枝可能未生效（请检查待剪枝模块选择）")
            else:
                tol = 0.05  # 允许的比例偏差
                status = "剪枝成功" if (actual_ratio > 0 and abs(actual_ratio - prune_base_ratio) <= tol) else "剪枝完成但比例偏差较大"
                info_str += f"{status}，完成全局剪枝，设定剪枝比例: {prune_base_ratio}，｜掩码零元素 {zero_mask_elems}/{total_mask_elems}，实际剪枝比例 {actual_ratio:.3f}\n"
                print(f"{status}，完成全局剪枝，设定剪枝比例: {prune_base_ratio}，｜掩码零元素 {zero_mask_elems}/{total_mask_elems}，实际剪枝比例 {actual_ratio:.3f}")
            
            # 将模型转移到 GPU
            model = model.cuda(local_rank)
            
        elif args.branch_type == 'ERB':
            # ERB 分支结构：训练态为多分支（rbr_3x3_branch 等），部署态为单一卷积 rbr_reparam
            # 为保证剪枝作用于真实推理卷积核，若处于训练态则先切换到部署态，再对 rbr_reparam 进行剪枝
            param_list = []
            # 先收集 stem 线性层，用于统一剪枝
            for k, v in model.named_parameters():
                if 'weight' in k and 'stem' in k:
                    stem_ind = int(k.split('.')[1])
                    param_list.append(model.stem[stem_ind])
                    info_str += f"添加 stem 层 {stem_ind} 的 Linear(权重)到剪枝列表\n"
                    print(f"添加 stem 层 {stem_ind} 的 Linear(权重)到剪枝列表")

            # 遍历卷积层，确保部署态并收集 rbr_reparam
            for layer_ind, layer in enumerate(model.layers):
                if isinstance(layer, NeRVBlock):
                    # 按训练态进行剪枝：直接选择各个 ERB 分支的卷积核
                    added = 0
                    if hasattr(layer, 'rbr_3x3_branch'):
                        param_list.append(layer.rbr_3x3_branch)
                        info_str += f"ERB 训练态：添加第 {layer_ind} 层 rbr_3x3_branch 到剪枝列表\n"
                        print(f"ERB 训练态：添加第 {layer_ind} 层 rbr_3x3_branch 到剪枝列表")
                        added += 1
                    if hasattr(layer, 'rbr_3x1_branch'):
                        param_list.append(layer.rbr_3x1_branch)
                        info_str += f"ERB 训练态：添加第 {layer_ind} 层 rbr_3x1_branch 到剪枝列表\n"
                        print(f"ERB 训练态：添加第 {layer_ind} 层 rbr_3x1_branch 到剪枝列表")
                        added += 1
                    if hasattr(layer, 'rbr_1x3_branch'):
                        param_list.append(layer.rbr_1x3_branch)
                        info_str += f"ERB 训练态：添加第 {layer_ind} 层 rbr_1x3_branch 到剪枝列表\n"
                        print(f"ERB 训练态：添加第 {layer_ind} 层 rbr_1x3_branch 到剪枝列表")
                        added += 1
                    # 序列卷积分支（1x1 -> 3x3 -> 1x1），如果存在则一并剪枝
                    if hasattr(layer, 'rbr_1x1_3x3_1x1_branch_1x1_1'):
                        param_list.append(layer.rbr_1x1_3x3_1x1_branch_1x1_1)
                        info_str += f"ERB 训练态：添加第 {layer_ind} 层 rbr_1x1_3x3_1x1_branch_1x1_1 到剪枝列表\n"
                        print(f"ERB 训练态：添加第 {layer_ind} 层 rbr_1x1_3x3_1x1_branch_1x1_1 到剪枝列表")
                        added += 1
                    if hasattr(layer, 'rbr_1x1_3x3_1x1_branch_3x3'):
                        param_list.append(layer.rbr_1x1_3x3_1x1_branch_3x3)
                        info_str += f"ERB 训练态：添加第 {layer_ind} 层 rbr_1x1_3x3_1x1_branch_3x3 到剪枝列表\n"
                        print(f"ERB 训练态：添加第 {layer_ind} 层 rbr_1x1_3x3_1x1_branch_3x3 到剪枝列表")
                        added += 1
                    if hasattr(layer, 'rbr_1x1_3x3_1x1_branch_1x1_2'):
                        param_list.append(layer.rbr_1x1_3x3_1x1_branch_1x1_2)
                        info_str += f"ERB 训练态：添加第 {layer_ind} 层 rbr_1x1_3x3_1x1_branch_1x1_2 到剪枝列表\n"
                        print(f"ERB 训练态：添加第 {layer_ind} 层 rbr_1x1_3x3_1x1_branch_1x1_2 到剪枝列表")
                        added += 1
                    if added == 0:
                        print(f"警告：第 {layer_ind} 层未发现 ERB 训练态分支，可能已是部署态或结构不匹配，跳过该层剪枝")

            param_to_prune = [(ele, 'weight') for ele in param_list]
            prune_base_ratio = args.prune_ratio

            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=prune_base_ratio,
            )

            # 剪枝成功性验证：统计掩码零元素比例
            mask_modules = [m for (m, name) in param_to_prune if hasattr(m, 'weight_mask')]
            total_mask_elems = sum(m.weight_mask.numel() for m in mask_modules)
            zero_mask_elems = sum((m.weight_mask == 0).sum().item() for m in mask_modules)
            actual_ratio = (zero_mask_elems / total_mask_elems) if total_mask_elems > 0 else 0.0
            if total_mask_elems == 0:
                print("警告：未检测到 weight_mask，ERB 剪枝可能未生效（请检查部署态与待剪枝模块选择）")
            else:
                tol = 0.05
                status = "剪枝成功" if (actual_ratio > 0 and abs(actual_ratio - prune_base_ratio) <= tol) else "剪枝完成但比例偏差较大"
                info_str += f"{status}，完成全局剪枝，设定剪枝比例: {prune_base_ratio}，｜掩码零元素 {zero_mask_elems}/{total_mask_elems}，实际剪枝比例 {actual_ratio:.3f}\n"
                print(f"{status}，完成全局剪枝，设定剪枝比例: {prune_base_ratio}，｜掩码零元素 {zero_mask_elems}/{total_mask_elems}，实际剪枝比例 {actual_ratio:.3f}")

            # 将模型转移到 GPU
            model = model.cuda(local_rank)
            
        # 统一设备字符串（在任何张量迁移之前先定义）
        loc = 'cuda:{}'.format(local_rank if local_rank is not None else 0)

        # 从检查点恢复训练状态
        args.start_epoch = 0
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] 
            train_best_psnr = checkpoint['train_best_psnr'].to(torch.device(loc))
            train_best_msssim = checkpoint['train_best_msssim'].to(torch.device(loc))
            val_best_psnr = checkpoint['val_best_psnr'].to(torch.device(loc))
            val_best_msssim = checkpoint['val_best_msssim'].to(torch.device(loc))
            # 关于优化器状态恢复与设备迁移的处理，调整到优化器创建之后
            pass

        # 强制对齐设备：确保所有参数、梯度与优化器状态均在同一 GPU 上（中文注释）
        param_device = torch.device(loc)
        model = model.to(param_device)  # 将模型迁移到指定设备
        # 额外对齐：确保所有缓冲（如均值/方差、注册的running_*）也在同一设备
        for buf in model.buffers():
            if buf.device != param_device:
                buf.data = buf.data.to(param_device)
        # 二次确认参数设备，并对任何残留的 CPU 参数进行强制迁移与提示
        off_params = []
        for name, p in model.named_parameters():
            if p.device != param_device:
                off_params.append((name, str(p.device)))
                p.data = p.data.to(param_device)
                if p.grad is not None and p.grad.device != param_device:
                    p.grad = p.grad.to(param_device)
        if len(off_params) > 0:
            print(f"提示：发现 {len(off_params)} 个参数未在目标设备，已强制迁移到 {param_device}。样例：{off_params[:3]}")
        for p in model.parameters():
            # 梯度可能仍在 CPU，确保其与参数一致
            if p.grad is not None and p.grad.device != param_device:
                p.grad = p.grad.to(param_device)
        
        # 在设备对齐后再初始化优化器，避免首次创建状态张量落在 CPU
        optimizer = optim.Adam(model.parameters(), betas=(args.beta, 0.999), foreach=False)
        
        # 若允许恢复旧优化器状态（不在剪枝微调场景），加载并统一迁移到 GPU
        if checkpoint is not None and not (args.finetune and (args.prune_ratio < 1)):
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and v.device != param_device:
                        state[k] = v.to(param_device)
                    elif isinstance(v, list):
                        state[k] = [t.to(param_device) if isinstance(t, torch.Tensor) and t.device != param_device else t for t in v]
        elif checkpoint is not None:
            print("提示：已进行剪枝或结构变更，跳过优化器状态恢复，使用全新优化器参数状态。")

        # 设备一致性断言与提示，便于定位残留的 CPU 张量
        devices_params = {str(p.device) for p in model.parameters()}
        devices_states = set()
        for st in optimizer.state.values():
            for vv in st.values():
                if isinstance(vv, torch.Tensor):
                    devices_states.add(str(vv.device))
                elif isinstance(vv, list):
                    for tt in vv:
                        if isinstance(tt, torch.Tensor):
                            devices_states.add(str(tt.device))
        if len(devices_params | devices_states) > 1:
            print(f"警告：发现参数/优化器状态存在多设备混用：params={devices_params}, states={devices_states}，将导致 foreach 优化器报错。")
        else:
            print(f"设备对齐完成：所有参数与优化器状态均在 {devices_params.pop() if devices_params else param_device} 上。")
            
            
        # 将 info_str 写入 fname 对应的 txt 文件
        fname = 'finetune_e{}_pr{:.2f}_q{}.txt'.format(
            args.finetune_epochs,
            args.prune_ratio,
            args.quant_bit if args.quant_bit != -1 else 'none'
        )
        with open('{}/{}'.format(args.outf, fname), 'a') as f:
            f.write(info_str)
            

        
        # 微调训练
        start = datetime.now() # 获取当前时间
        total_epochs = args.start_epoch + args.finetune_epochs # 计算总的训练轮数
        for epoch in range(args.start_epoch, total_epochs):
            model.train()
            
            epoch_start_time = datetime.now()
            psnr_list = []
            msssim_list = []
            # 训练一个epoch
            for i, (data,  norm_idx) in enumerate(train_dataloader):
                embed_input = PE(norm_idx)
                if local_rank is not None:
                    data = data.cuda(local_rank, non_blocking=True)
                    embed_input = embed_input.cuda(local_rank, non_blocking=True)
                else:
                    data, embed_input = data.cuda(non_blocking=True), embed_input.cuda(non_blocking=True)
                    
                # forward and backward
                output_list = model(embed_input)
                target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
                
                # 计算损失函数
                loss_list = [loss_fn(output, target, args) for output, target in zip(output_list, target_list)]
                loss_list = [loss_list[i] * (args.lw if i < len(loss_list) - 1 else 1) for i in range(len(loss_list))]
                loss_sum = sum(loss_list)
                
                lr = adjust_lr(optimizer, epoch % total_epochs, i, data_size, args)
                optimizer.zero_grad()
                # 避免出现“Trying to backward through the graph a second time”错误：
                # 对 ERB 在线重参数化分支，某些中间图在同一迭代内可能被重复引用，
                # 保守地在该分支使用 retain_graph=True；其他分支保持默认 False。
                # 注意：retain_graph 会增加显存占用，如无该报错可改回 False。
                loss_sum.backward(retain_graph=(args.branch_type == 'ERB'))
                # 运行期兜底：对齐每个参数的梯度与优化器状态到参数所在设备
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is not None and p.grad.device != p.device:
                            # 将梯度迁移到参数所在设备，避免 CPU/GPU 混用
                            p.grad = p.grad.to(p.device)
                        st = optimizer.state.get(p, None)
                        if st is None:
                            continue
                        for k, v in st.items():
                            if isinstance(v, torch.Tensor) and v.device != p.device:
                                st[k] = v.to(p.device)
                            elif isinstance(v, list):
                                st[k] = [t.to(p.device) if isinstance(t, torch.Tensor) and t.device != p.device else t for t in v]
                # 在首次 step 之前清空旧状态（尤其是剪枝/结构变更后），确保 Adam 在 GPU 上重新初始化状态
                if epoch == args.start_epoch and i == 0:
                    optimizer.state.clear()
                optimizer.step()
                
                # compute psnr and msssim（在 no_grad 中计算，避免构建/持有计算图，防止潜在二次反向问题）
                with torch.no_grad():
                    psnr_list.append(psnr_fn(output_list, target_list))
                    msssim_list.append(msssim_fn(output_list, target_list))
                if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
                    train_psnr = torch.cat(psnr_list, dim=0) #(batchsize, num_stage)
                    train_psnr = torch.mean(train_psnr, dim=0) #(num_stage)
                    train_msssim = torch.cat(msssim_list, dim=0) #(batchsize, num_stage)
                    train_msssim = torch.mean(train_msssim.float(), dim=0) #(num_stage)
                    time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                    print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}, MSSSIM: {}'.format(
                        time_now_string, local_rank, epoch+1, total_epochs, i+1, len(train_dataloader), lr, 
                        RoundTensor(train_psnr, 2, False), RoundTensor(train_msssim, 4, False))
                    print(print_str, flush=True)
                    with open('{}/{}'.format(args.outf, fname), 'a') as f:
                        f.write(print_str + '\n')
            
            if local_rank in [0, 1, 2, 3, None]:
                h, w = output_list[-1].shape[-2:]
                is_train_best = train_psnr[-1] > train_best_psnr
                train_best_psnr = train_psnr[-1] if train_psnr[-1] > train_best_psnr else train_best_psnr
                train_best_msssim = train_msssim[-1] if train_msssim[-1] > train_best_msssim else train_best_msssim
            
                print_str = '\t{}p: current: {:.2f}\t best: {:.2f}\t msssim_best: {:.4f}\t'.format(h, train_psnr[-1].item(), train_best_psnr.item(), train_best_msssim.item())
                
                epoch_end_time = datetime.now()
                time_str = "Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format( (epoch_end_time - epoch_start_time).total_seconds(), \
                        (epoch_end_time - start).total_seconds() / (epoch + 1 - args.start_epoch) )
                print_str += time_str
                print(print_str, flush=True)
                with open('{}/{}'.format(args.outf, fname), 'a') as f:
                    f.write(print_str + '\n')

        # 微调训练后，将 ERB 分支模型切换为部署态
        if args.branch_type == 'ERB':
            # Generator 不存在整体 switch_to_deploy 方法，需要逐层切换
            # 逐层调用 NeRVBlock.switch_to_deploy，将训练态分支融合为单一 3x3 卷积（中文注释）
            deploy_count = 0
            for layer in model.layers:
                if isinstance(layer, NeRVBlock) and hasattr(layer, 'switch_to_deploy'):
                    layer.switch_to_deploy()
                    deploy_count += 1
            print_str = f"微调训练结束，ERB 分支模型已调整为部署态，共切换 {deploy_count} 个 NeRVBlock"
            print(print_str)
            with open('{}/{}'.format(args.outf, fname), 'a') as f:
                    f.write(print_str + '\n')
                    
        
                    
    # 加载模型权重：不需要微调时，ERB 分支加载部署态模型，NeRV_vanilla 分支加载最新模型
    # *! 无需微调时的模型加载和剪枝逻辑
    if not args.finetune and prune_net:
        if args.branch_type == 'ERB':
            # 非微调评估场景：加载部署态权重并执行“全局剪枝”（中文注释）
            deploy_path = os.path.join(args.outf, 'model_latest_deploy.pth')
            if not os.path.isfile(deploy_path):
                raise FileNotFoundError(f"部署态模型文件不存在: {deploy_path}")
            # 安全加载（优先 weights_only），并清理无关键
            try:
                checkpoint = torch.load(deploy_path, map_location='cpu', weights_only=True)
            except (TypeError, AttributeError):
                checkpoint = torch.load(deploy_path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
            if isinstance(state_dict, dict):
                keys_to_remove = [k for k in state_dict.keys() if ('total_ops' in k or 'total_params' in k)]
                for k in keys_to_remove:
                    del state_dict[k]
            model.load_state_dict(state_dict, strict=False)
            info_str += f"已安全加载部署态模型权重: {deploy_path}，将对部署态单卷积进行剪枝\n"
            print(f"已安全加载部署态模型权重: {deploy_path}，将对部署态单卷积进行剪枝")

            # 部署态 ERB：同时剪 MLP 的 stem 线性层与卷积的 rbr_reparam（中文注释）
            param_list = []
            # 先收集 MLP（stem）层的 Linear 权重
            for k, v in model.named_parameters():
                if 'weight' in k and 'stem' in k:
                    stem_ind = int(k.split('.')[1])
                    info_str += f"ERB 部署态：添加 stem 层 {stem_ind} 的 Linear(权重)到剪枝列表\n"
                    print(f"ERB 部署态：添加 stem 层 {stem_ind} 的 Linear(权重)到剪枝列表")
                    param_list.append(model.stem[stem_ind])
            # 再收集所有 NeRVBlock 的部署态卷积 rbr_reparam
            for idx, layer in enumerate(model.layers):
                if isinstance(layer, NeRVBlock) and hasattr(layer, 'rbr_reparam'):
                    info_str += f"ERB 部署态：添加第 {idx} 层 rbr_reparam 到剪枝列表\n"
                    print(f"ERB 部署态：添加第 {idx} 层 rbr_reparam 到剪枝列表")
                    param_list.append(layer.rbr_reparam)
            param_to_prune = [(m, 'weight') for m in param_list]
            prune.global_unstructured(param_to_prune, pruning_method=prune.L1Unstructured, amount=args.prune_ratio)
            # 统计剪枝掩码实际比例，做一致性校验与提示
            mask_modules = [m for (m, name) in param_to_prune if hasattr(m, 'weight_mask')]
            total_mask_elems = sum(m.weight_mask.numel() for m in mask_modules)
            zero_mask_elems = sum((m.weight_mask == 0).sum().item() for m in mask_modules)
            actual_ratio = (zero_mask_elems / total_mask_elems) if total_mask_elems > 0 else 0.0
            status = "剪枝成功" if total_mask_elems > 0 else "警告：未检测到 weight_mask，剪枝可能未生效"
            info_str += f"{status}（ERB-Deploy）：设定比例 {args.prune_ratio}，实际 {actual_ratio:.3f}，掩码零元素 {zero_mask_elems}/{total_mask_elems}\n"
            print(f"{status}（ERB-Deploy）：设定比例 {args.prune_ratio}，实际 {actual_ratio:.3f}，掩码零元素 {zero_mask_elems}/{total_mask_elems}")

        elif args.branch_type == 'NeRV_vanilla':
            # 非微调评估场景：加载训练态最新权重并执行“全局剪枝”（中文注释）
            latest_path = os.path.join(args.outf, 'model_latest.pth')
            if not os.path.isfile(latest_path):
                raise FileNotFoundError(f"最新模型文件不存在: {latest_path}")
            try:
                checkpoint = torch.load(latest_path, map_location='cpu', weights_only=True)
            except (TypeError, AttributeError):
                checkpoint = torch.load(latest_path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
            if isinstance(state_dict, dict):
                keys_to_remove = [k for k in state_dict.keys() if ('total_ops' in k or 'total_params' in k)]
                for k in keys_to_remove:
                    del state_dict[k]
            model.load_state_dict(state_dict, strict=False)
            info_str += f"已安全加载最新训练态模型权重: {latest_path}，将对 MLP+Conv 进行剪枝\n"
            print(f"已安全加载最新训练态模型权重: {latest_path}，将对 MLP+Conv 进行剪枝")

            # NeRV_vanilla：stem 的 Linear 与 NeRVBlock 的 branch/rbr_reparam 参与剪枝
            param_list = []
            for k, v in model.named_parameters():
                if 'weight' in k:
                    if 'stem' in k:
                        # 打印 MLP（stem）层剪枝收集信息
                        stem_ind = int(k.split('.')[1])
                        info_str += f"NeRV_vanilla 训练态：添加 stem 层 {stem_ind} 的 Linear(权重)到剪枝列表\n"
                        print(f"添加 stem 层 {stem_ind} 的 Linear(权重)到剪枝列表")
                        param_list.append(model.stem[stem_ind])
                    elif 'layers' in k[:6]:
                        # 打印卷积块剪枝收集信息：训练态 branch 或部署态 rbr_reparam
                        layer_ind = int(k.split('.')[1])
                        layer = model.layers[layer_ind]
                        target = None
                        if hasattr(layer, 'branch'):
                            target = layer.branch
                            info_str += f"NeRV_vanilla 训练态：添加第 {layer_ind} 层 branch 卷积到剪枝列表\n"
                            print(f"NeRV_vanilla 训练态：添加第 {layer_ind} 层 branch 卷积到剪枝列表")
                        elif hasattr(layer, 'rbr_reparam'):
                            target = layer.rbr_reparam
                            info_str += f"NeRV_vanilla 部署态：添加第 {layer_ind} 层 rbr_reparam 卷积到剪枝列表\n"
                            print(f"NeRV_vanilla 部署态：添加第 {layer_ind} 层 rbr_reparam 卷积到剪枝列表")
                        if target is not None:
                            param_list.append(target)
            param_to_prune = [(ele, 'weight') for ele in param_list]
            prune.global_unstructured(param_to_prune, pruning_method=prune.L1Unstructured, amount=args.prune_ratio)
            mask_modules = [m for (m, name) in param_to_prune if hasattr(m, 'weight_mask')]
            total_mask_elems = sum(m.weight_mask.numel() for m in mask_modules)
            zero_mask_elems = sum((m.weight_mask == 0).sum().item() for m in mask_modules)
            actual_ratio = (zero_mask_elems / total_mask_elems) if total_mask_elems > 0 else 0.0
            status = "剪枝成功" if total_mask_elems > 0 else "警告：未检测到 weight_mask，剪枝可能未生效"
            info_str += f"{status}（NeRV_vanilla）：设定比例 {args.prune_ratio}，实际 {actual_ratio:.3f}，掩码零元素 {zero_mask_elems}/{total_mask_elems}\n"
            print(f"{status}（NeRV_vanilla）：设定比例 {args.prune_ratio}，实际 {actual_ratio:.3f}，掩码零元素 {zero_mask_elems}/{total_mask_elems}")
            
    
    # # 量化流程占位：避免未完成代码导致语法错误（中文注释）
    if args.quant_bit != -1:
        with torch.no_grad():
            print(f"进行量化处理，量化位宽: {args.quant_bit}")
            # 记录量化前模型所在设备，后续保持一致，避免 CPU/CUDA 设备不一致
            model_device_before = next(model.parameters()).device
            cur_ckt = model.state_dict()
            from dahuffman import HuffmanCodec
            quant_weitht_list = []
            for k,v in cur_ckt.items():
                large_tf = (v.dim() in {2,4} and 'bias' not in k) # 对 2D 卷积核和 4D 卷积核进行量化
                quant_v, new_v = quantize_per_tensor(v, args.quant_bit, args.quant_axis if large_tf else -1) # 对大张量（如卷积核）按轴量化，小张量（如偏置）按张量量化
                # 统一在 CPU 上进行统计，避免 GPU/CPU 掩码设备不一致
                quant_v_cpu = quant_v.detach().cpu()
                mask_cpu = (v.detach().cpu() != 0)
                valid_quant_v = quant_v_cpu[mask_cpu] # 只对非零权重进行量化统计
                quant_weitht_list.append(valid_quant_v.flatten()) # 对量化后的非零权重进行拼接
                # 保持新权重与模型原始设备与数据类型一致（使用量化前模型设备），避免 forward 时设备不匹配
                cur_ckt[k] = new_v.detach().to(model_device_before).type_as(v) # 更新模型状态字典中的权重
            # 量化成功日志：统计已量化的张量数量（按非零权重统计集合）
            info_str += f"量化成功：已处理 {len(quant_weitht_list)} 个参数张量，并写回量化权重到 state_dict\n"
            print(f"量化成功：已处理 {len(quant_weitht_list)} 个参数张量，并写回量化权重到 state_dict")
            cat_param = torch.cat(quant_weitht_list) # 对所有量化后的非零权重进行拼接
            input_code_list = cat_param.tolist() # 将拼接后的张量转换为列表
            unique, counts = np.unique(input_code_list, return_counts=True) # 统计每个量化值的出现次数
            num_freq = dict(zip(unique, counts)) # 将量化值和出现次数转换为字典
            
            # generating HuffmanCoding table
            codec = HuffmanCodec.from_data(input_code_list) # 从非零权重列表中生成 Huffman 编码器
            code_table = codec.get_code_table() # 获取编码表（符号 -> (码长, 编码串)）
            # 熵编码成功日志：符号数量与码长范围
            min_len = min(v[0] for v in code_table.values()) if len(code_table) > 0 else 0
            max_len = max(v[0] for v in code_table.values()) if len(code_table) > 0 else 0
            info_str += f"熵编码成功：哈夫曼码表生成，符号数 {len(code_table)}，码长范围 [{min_len}, {max_len}]\n"
            print(f"熵编码成功：哈夫曼码表生成，符号数 {len(code_table)}，码长范围 [{min_len}, {max_len}]")
            sym_bit_dict = {}
            for k, v in code_table.items():
                sym_bit_dict[k] = v[0]
            total_bits = 0
            for num, freq in num_freq.items():
                total_bits += freq * sym_bit_dict[num]
            avg_bits = total_bits / len(input_code_list)    
            # import pdb; pdb.set_trace; from IPython import embed; embed()       
            encoding_efficiency = avg_bits / args.quant_bit
            # 打印平均码长与效率
            print(f"平均码长：{avg_bits:.4f} bit/符号")
            print_str = f'Entropy encoding efficiency for bit {args.quant_bit}: {encoding_efficiency}'
            print(print_str)
            if local_rank in [0, 1, 2, 3, None]:
                with open('{}/rank{}.txt'.format(args.outf, local_rank), 'a') as f:
                    f.write(print_str + '\n')       
            # 将量化权重载入模型，完成量化回写
            model.load_state_dict(cur_ckt)
            # 加载后强制模型保持在量化前设备，避免出现 CPU/CUDA 不一致
            current_device = next(model.parameters()).device
            if current_device != model_device_before:
                print(f"量化后设备不一致：模型当前 {current_device}，恢复到 {model_device_before}")
                model.to(model_device_before)
            # 统一将模型移动到 GPU 设备（若可用），避免后续 embed_input 在 CUDA 而模型仍在 CPU
            if torch.cuda.is_available():
                model = model.cuda(local_rank)
                print(f"量化后模型已移动到 CUDA 设备：cuda:{local_rank}")

            # 计算 BPP（Bits Per Pixel）：使用熵编码后的总比特与验证集像素数量（中文注释）
            try:
                # 读取验证集首帧以获取分辨率（形状为 [C, H, W]）
                sample_img, _ = val_dataset[0]
                H, W = sample_img.shape[-2], sample_img.shape[-1]
                frame_count = len(val_dataset)
                total_pixels = frame_count * H * W
                # 注意：total_bits 为哈夫曼编码后的数据比特总量，不包含码表开销；如需更严谨，可加上码表存储成本（此处先不计）
                bpp = total_bits / total_pixels if total_pixels > 0 else 0.0
                bpp_str = f"BPP 统计：总比特 {int(total_bits)}，帧数 {frame_count}，分辨率 {H}x{W}，BPP={bpp:.6f} bit/pixel"
                print(bpp_str)
                if local_rank in [0, 1, 2, 3, None]:
                    with open('{}/bpp_rank{}.txt'.format(args.outf, local_rank), 'a') as f:
                        f.write(bpp_str + '\n')
            except Exception as e:
                print(f"BPP 统计失败：{e}")
                
        
    # 构造文件名：包含剪枝比率和量化位宽，且标记为“未微调”
    only_name = 'only_prune{:.2f}_quant{}.txt'.format(args.prune_ratio, args.quant_bit if args.quant_bit > 0 else 'full')
    with open('{}/{}'.format(args.outf, only_name), 'w', encoding='utf-8') as f:
        f.write(info_str)
    print(f"已将 info_str 写入文件: {only_name}")
            
    psnr_list = []
    msssim_list = []
    time_list = []
    # 设置每次统计 FPS 时的前向次数（默认为 10，可通过命令行覆盖）（中文注释）
    fwd_num = getattr(args, 'fwd_num', 10)
    model.eval()
    for i, (data,  norm_idx) in enumerate(val_dataloader):
        embed_input = PE(norm_idx)
        if local_rank is not None:
            data = data.cuda(local_rank, non_blocking=True)
            embed_input = embed_input.cuda(local_rank, non_blocking=True)
        else:
            data, embed_input = data.cuda(non_blocking=True), embed_input.cuda(non_blocking=True)
            
        # 统计前向耗时：重复 fwd_num 次前向，记录耗时并用于计算 FPS（中文注释）
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _start = time.time()
            out = None
            for _ in range(fwd_num):
                out = model(embed_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time_list.append(time.time() - _start)
        output_list = out
        
        
        # 计算验证集第一张图像的输出FPS、MACs和FLOPs
        if i == 0:  # 仅对第一张验证图像进行统计
            model.eval()
            # 使用位置编码作为模型输入（embed_input），避免将图像张量传入模型导致维度不匹配（中文注释）
            dummy_embed = embed_input[:1]
            # 预热
            for _ in range(5):
                _ = model(dummy_embed)
            torch.cuda.synchronize() if torch.cuda.is_available() else None

            # 测FPS
            start = time.time()
            repeat = 50
            for _ in range(repeat):
                _ = model(dummy_embed)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            fps = repeat / (time.time() - start)
            eval_str = f"[验证集首张图像] FPS: {fps:.2f}\n"
            print(f"[验证集首张图像] FPS: {fps:.2f}")

            # 测MACs/FLOPs
            macs, params = profile(model, inputs=(dummy_embed,), verbose=False)
            flops = 2 * macs
            eval_str += f"[验证集首张图像] MACs: {macs / 1e9:.3f} G, FLOPs: {flops / 1e9:.3f} G\n"
            print(f"[验证集首张图像] MACs: {macs / 1e9:.3f} G, FLOPs: {flops / 1e9:.3f} G")
            model.train()
        
        
        # dump predictions
        if args.dump_images:
            from torchvision.utils import save_image
            visual_dir = f'{args.outf}/visualize'
            print(f'Saving predictions to {visual_dir}')
            if not os.path.isdir(visual_dir):
                os.makedirs(visual_dir)
            for batch_ind in range(args.batchSize):
                full_ind = i * args.batchSize + batch_ind
                save_image(output_list[-1][batch_ind], f'{visual_dir}/pred_{full_ind}.png')
                # save_image(data[batch_ind], f'{visual_dir}/gt_{full_ind}.png')
        
        # compute psnr and ms-ssim
        target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
        psnr_list.append(psnr_fn(output_list, target_list))
        msssim_list.append(msssim_fn(output_list, target_list))
        val_psnr = torch.cat(psnr_list, dim=0)              #(batchsize, num_stage)
        val_psnr = torch.mean(val_psnr, dim=0)              #(num_stage)
        val_msssim = torch.cat(msssim_list, dim=0)          #(batchsize, num_stage)
        val_msssim = torch.mean(val_msssim.float(), dim=0)  #(num_stage)        
        if i % args.print_freq == 0 or i == len(val_dataloader) - 1:
            fps = fwd_num * (i+1) * args.batchSize / sum(time_list)
            print_str = 'Rank:{}, Step [{}/{}], PSNR: {}, MSSSIM: {} FPS: {}'.format(
                local_rank, i+1, len(val_dataloader),
                RoundTensor(val_psnr, 2, False), RoundTensor(val_msssim, 4, False), round(fps, 2))
            print(print_str)

            if args.finetune:
                fname = fname
            else:
                fname = only_name
            with open('{}/{}'.format(args.outf, fname), 'a') as f:
                f.write(print_str + '\n')
                f.write(eval_str + '\n')
    


if __name__ == '__main__':
    main()

        
            
            
        
        
    