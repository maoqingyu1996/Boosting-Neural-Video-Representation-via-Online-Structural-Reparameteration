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
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

    args = parser.parse_args()
        
    args.warmup = int(args.warmup * args.epochs)

    # print(args)
    torch.set_printoptions(precision=4) 

    # 评估模式时，评估频率设为1，输出目录设为RESULT/debug
    if args.debug:
        args.eval_freq = 1
        args.outf = 'RESULT/debug'
    else:
        args.outf = os.path.join('RESULT', args.outf)  ### output directory
    
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
    
    # 若输出文件目录已存在，且overwrite参数为True，则删除已存在目录
    if args.overwrite and os.path.isdir(args.outf):
        print('Will overwrite the existing output dir!')
        shutil.rmtree(args.outf)

    # 创建输出文件目录
    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    # 分布式训练时，设置init_method参数为localhost的随机端口
    port = hash(args.exp_id) % 20000 + 10000
    args.init_method =  f'tcp://127.0.0.1:{port}'
    print(f'init_method: {args.init_method}', flush=True)

    # 设置随机种子和分布式训练
    torch.set_printoptions(precision=2) 
    args.ngpus_per_node = torch.cuda.device_count()
    if args.distributed and args.ngpus_per_node > 1:
        mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))
    else:
        local_rank = 0
        train(local_rank, args)

def train(local_rank, args):
    # 固定随机种子确保可复现性
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    # 初始化训练指标和两个标志位
    train_best_psnr, train_best_msssim, val_best_psnr, val_best_msssim = [torch.tensor(0) for _ in range(4)]
    is_train_best, is_val_best = False, False

    # 初始化模型和位置编码
    PE = PositionalEncoding(args.embed)
    args.embed_length = PE.embed_length
    model = Generator(embed_length=args.embed_length, stem_dim_num=args.stem_dim_num, fc_hw_dim=args.fc_hw_dim, expansion=args.expansion, 
                      num_blocks=args.num_blocks, norm=args.norm, act=args.act, bias = True, reduction=args.reduction, conv_type=args.conv_type,
                      stride_list=args.strides, sin_res=args.single_res, lower_width=args.lower_width, sigmoid=args.sigmoid, 
                      deploy=args.deploy, branch_type=args.branch_type)

    # 计算模型参数量并记录到txt中
    total_params = sum([p.data.nelement() for p in model.parameters()]) / 1e6 
    if local_rank in [0, 1, 2, 3, None]:
        print(f'{args}\n {model}\n Model Params: {total_params}M')
        with open('{}/rank{}.txt'.format(args.outf, local_rank), 'a') as f:
            f.write(str(model) + '\n' + f'Params: {total_params}M\n')
        writer = SummaryWriter(os.path.join(args.outf, f'param_{total_params}M', 'tensorboard'))
    else:
        writer = None
    
    # 判断是否需要剪枝，评估模式则剪枝
    prune_net = args.prune_ratio < 1
    # prune model params and flops 训练前剪枝部分
    if prune_net and args.branch_type == 'NeRV_vanilla': 
        param_list = [] # 存储需要剪枝的参数
        for k, v in model.named_parameters(): # 遍历模型参数
            if 'weight' in k:
                if 'stem' in k:
                    stem_ind = int(k.split('.')[1])
                    param_list.append(model.stem[stem_ind])
                elif 'layers' in k[:6] and 'conv' in k:
                    layer_ind = int(k.split('.')[1])
                    param_list.append(model.layers[layer_ind].conv.conv)
        param_to_prune = [(ele, 'weight') for ele in param_list] # 
        prune_base_ratio = args.prune_ratio ** (1. / len(args.prune_steps))
        args.prune_steps = [int(x * args.epochs) for x in args.prune_steps]
        prune_num = 0

        if args.eval_only:
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )

    # distrite model to gpu or parallel 分布式训练部分
    print("Use GPU: {} for training".format(local_rank))
    if args.distributed and args.ngpus_per_node > 1:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            world_size=args.ngpus_per_node,
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)
        assert torch.distributed.is_initialized()        
        args.batchSize = int(args.batchSize / args.ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], \
                                                          output_device=local_rank, find_unused_parameters=False)
    else:
        device = torch.device('cuda:{}'.format(local_rank))
        model = model.to(device)

    # ** 初始化优化器
    optimizer = optim.Adam(model.parameters(), betas=(args.beta, 0.999))
    loc = 'cuda:{}'.format(local_rank if local_rank is not None else 0)

    # *!从指定权重文件恢复，不确定是训练还是测试
    checkpoint = None
    if args.weight != 'None':
        print("=> loading checkpoint '{}'".format(args.weight))
        checkpoint_path = args.weight
        try:
            # 优先使用安全加载，仅反序列化权重张量
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 取出 state_dict，并做键清理（去掉统计项、兼容 blocks.0 前缀）
        orig_ckt = checkpoint['state_dict']
        new_ckt = {k.replace('blocks.0.', ''): v for k, v in orig_ckt.items()}
        new_ckt = {k: v for k, v in new_ckt.items() if not ("total_ops" in k or "total_params" in k)}

        # ==== 关键修复：根据检查点键名判断其形态，然后让模型匹配该形态再加载 ====
        is_erb_branch = (args.branch_type == 'ERB')
        ckpt_is_deploy = any(('rbr_reparam' in k) for k in new_ckt.keys()) if is_erb_branch else False
        # 简单判断当前模型是否已在部署态（存在 rbr_reparam 模块）
        model_is_deploy = False
        if is_erb_branch:
            try:
                # 若第一个块已经有 rbr_reparam 属性，视为部署态
                model_is_deploy = hasattr(model.layers[0], 'rbr_reparam') and isinstance(model.layers[0].rbr_reparam, torch.nn.Module)
            except Exception:
                model_is_deploy = False

        if is_erb_branch:
            if ckpt_is_deploy and not model_is_deploy:
                # 检查点是部署态；模型为训练态 ⇒ 先切换模型到部署态再加载
                for layer in model.layers:
                    if isinstance(layer, NeRVBlock):
                        layer.switch_to_deploy()  # 将多分支融合为 rbr_reparam
                model_is_deploy = True
            elif (not ckpt_is_deploy) and model_is_deploy:
                # 检查点是训练态；模型已是部署态 ⇒ 重新构建一个训练态模型再加载
                prev_device = next(model.parameters()).device # 记录当前设备
                model = Generator(
                    embed_length=args.embed_length, stem_dim_num=args.stem_dim_num, fc_hw_dim=args.fc_hw_dim,
                    expansion=args.expansion, num_blocks=args.num_blocks, norm=args.norm, act=args.act, bias=True,
                    reduction=args.reduction, conv_type=args.conv_type, stride_list=args.strides,
                    sin_res=args.single_res, lower_width=args.lower_width, sigmoid=args.sigmoid,
                    deploy=False, branch_type=args.branch_type  # 强制训练态
                ).to(prev_device)
                model_is_deploy = False
            # 其他情况：两者形态一致，无需变更

        # 统一处理 DataParallel/DistributedDataParallel 的 'module.' 前缀后再加载
        if 'module' in list(orig_ckt.keys())[0] and not hasattr(model, 'module'):
            new_ckt = {k.replace('module.', ''): v for k, v in new_ckt.items()}
            model.load_state_dict(new_ckt)
        elif 'module' not in list(orig_ckt.keys())[0] and hasattr(model, 'module'):
            model.module.load_state_dict(new_ckt)
        else:
            model.load_state_dict(new_ckt)

        print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint.get('epoch', 'N/A')))

        # 2) 如需 ERB 部署态与剪枝：加载完成后再切换与剪枝
        prune_net = args.prune_ratio < 1
        if args.branch_type == 'ERB':
            # 切换为部署态（融合多分支到 rbr_reparam）
            for layer in model.layers:
                if isinstance(layer, NeRVBlock):
                    layer.switch_to_deploy()
            # 重新统计参数量（部署态）
            re_params = sum([p.data.nelement() for p in model.parameters()]) / 1e6

            # 收集剪枝对象：stem 的线性层 + 各块的 rbr_reparam
            param_list = []
            for k, v in model.named_parameters():
                if 'weight' in k and 'stem' in k:
                    stem_ind = int(k.split('.')[1])
                    param_list.append(model.stem[stem_ind])
                elif 'layers' in k[:6] and 'rbr_reparam' in k:
                    layer_ind = int(k.split('.')[1])
                    param_list.append(model.layers[layer_ind].rbr_reparam)

            param_to_prune = [(ele, 'weight') for ele in param_list]
            args.prune_steps = [int(x * args.epochs) for x in args.prune_steps]
            prune_base_ratio = args.prune_ratio ** (1. / len(args.prune_steps)) if len(args.prune_steps) > 0 else 1.0
            prune_num = 0

            if prune_net and args.eval_only:
                prune.global_unstructured(
                    param_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=1 - prune_base_ratio ** prune_num,
                )
                sparsity_num = sum([(param.weight == 0).sum() for param in param_list])
                print(f'Model sparsity: {sparsity_num / 1e6 / re_params}')
        
                if 'module' in list(orig_ckt.keys())[0] and not hasattr(model, 'module'):
                    new_ckt={k.replace('module.',''):v for k,v in new_ckt.items()}
                    model.load_state_dict(new_ckt)
                elif 'module' not in list(orig_ckt.keys())[0] and hasattr(model, 'module'):
                    model.module.load_state_dict(new_ckt)
                else:
                    model.load_state_dict(new_ckt)

                print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch']))

    # resume from model_latest 从最新模型权重文件恢复训练
    checkpoint_path = os.path.join(args.outf, 'model_latest.pth')
    if os.path.isfile(checkpoint_path):
        try:
            # 同样使用安全模式加载最新模型
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # 清理非权重项（若存在）
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            keys_to_remove = [key for key in state_dict.keys() if 'total_ops' in key or 'total_params' in key]
            for key in keys_to_remove:
                del state_dict[key]
            checkpoint['state_dict'] = state_dict
        
        if prune_net:
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )
            
            sparsity_num = 0.
            for param in param_list:
                sparsity_num += (param.weight == 0).sum()

            if args.branch_type == 'NeRV_vanilla':
                print(f'Model sparsity: {sparsity_num / 1e6 / total_params}')
            elif args.branch_type == 'ERB':
                print(f'Model sparsity: {sparsity_num / 1e6 / re_params}')
            
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Auto resume loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
        print("=> No resume checkpoint found at '{}'".format(checkpoint_path))
    
    # 从检查点恢复训练状态
    args.start_epoch = 0
    if checkpoint is not None:
        args.start_epoch = checkpoint['epoch'] 
        train_best_psnr = checkpoint['train_best_psnr'].to(torch.device(loc))
        train_best_msssim = checkpoint['train_best_msssim'].to(torch.device(loc))
        val_best_psnr = checkpoint['val_best_psnr'].to(torch.device(loc))
        val_best_msssim = checkpoint['val_best_msssim'].to(torch.device(loc))
        if not prune_net or args.branch_type == 'NeRV-vanilla':
            optimizer.load_state_dict(checkpoint['optimizer'])

    # 如果不恢复训练状态，从0开始训练
    if args.not_resume_epoch:
        args.start_epoch = 0

    # 设置数据加载器
    img_transforms = transforms.ToTensor()
    DataSet = CustomDataSet
    train_data_dir = f'../data/{args.dataset.lower()}' ### dataset path
    val_data_dir = f'../data/{args.dataset.lower()}'

    train_dataset = DataSet(train_data_dir, img_transforms, vid_list=args.vid, frame_gap=args.frame_gap)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, 
                                                   sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn)

    val_dataset = DataSet(val_data_dir, img_transforms, vid_list=args.vid, frame_gap=args.test_gap)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if args.distributed else None
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True, 
                                                 sampler=val_sampler, drop_last=False, worker_init_fn=worker_init_fn)
    data_size = len(train_dataset)

    ###########################################Evaluation 评估模式
    if args.eval_only:
        print('\nEvaluation ...')
        time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        print_str = f'{time_str}\t Results for checkpoint: {args.weight}\n'

        if args.branch_type == 'ERB' and not prune_net:
            for layer in model.layers:
                if isinstance(layer, NeRVBlock):
                    layer.switch_to_deploy()
            print("Switched to reparameterized model.")
            re_params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
            print(f"Rep-model Params: {re_params}M\n")
            with open('{}/rank{}.txt'.format(args.outf, local_rank), 'a') as f:
                f.write('\nRep-model:\n' + str(model) + '\n' + f'Params: {re_params}M\n')

        if prune_net:
            for param in param_to_prune:
                prune.remove(param[0], param[1])
            sparsity_num = 0.
            for param in param_list:
                sparsity_num += (param.weight == 0).sum()

            if args.branch_type == 'NeRV_vanilla':
                sparsity = sparsity_num / 1e6 / total_params
            elif args.branch_type == 'ERB':
                sparsity = sparsity_num / 1e6 / re_params
            
            print(f'Model sparsity at Epoch{args.start_epoch}: {sparsity}')
            with open('{}/rank{}.txt'.format(args.outf, local_rank), 'a') as f:
                f.write(f'Model sparsity at Epoch{args.start_epoch}: {sparsity}\n')

        start_time = time.time()
        val_psnr, val_msssim = evaluate(model, val_dataloader, PE, local_rank, args)
        end_time = time.time()
        val_time = end_time - start_time

        print_str += f'PSNR/MS_SSIM on validate set for bit {args.quant_bit} with axis {args.quant_axis}: {round(val_psnr.item(),2)}/{round(val_msssim.item(),4)}\n'
        print_str += f'Total validation time: {val_time:.2f} seconds\n'
        print(print_str)
        with open('{}/rank{}.txt'.format(args.outf, local_rank), 'a') as f:
            f.write(print_str + '\n\n')
        return
    
    ###### 剪枝后的处理
    if prune_net and args.branch_type == 'ERB':
        for layer in model.layers:
            if isinstance(layer, NeRVBlock):
                layer.switch_to_deploy()
        print("Switched to reparameterized model.")
        re_params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
        print(f"Rep-model Params: {re_params}M\n")
        with open('{}/rank{}.txt'.format(args.outf, local_rank), 'a') as f:
            f.write('\nRep-model:\n' + str(model) + '\n' + f'Params: {re_params}M\n')

        param_list = []
        for k, v in model.named_parameters():
            if 'weight' in k:
                if 'stem' in k:
                    stem_ind = int(k.split('.')[1])
                    param_list.append(model.stem[stem_ind])
            elif 'layers' in k[:6]:
                layer_ind = int(k.split('.')[1])
                if 'rbr_reparam' in k:
                    param_list.append(model.layers[layer_ind].rbr_reparam)
        param_to_prune = [(ele, 'weight') for ele in param_list]
        prune_base_ratio = args.prune_ratio ** (1. / len(args.prune_steps))
        args.prune_steps = [int(x * args.epochs) for x in args.prune_steps]
        prune_num = 0

    # Training
    start = datetime.now() # 获取当前时间
    total_epochs = args.epochs * args.cycles # 计算总的训练轮数
    for epoch in range(args.start_epoch, total_epochs):
        model.train() 
        # prune the network if needed
        if prune_net and epoch in args.prune_steps: # 如果剪枝标志位为真且当前epoch在剪枝步骤中
            prune_num += 1 
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )
        
            sparsity_num = 0.
            for param in param_list:
                sparsity_num += (param.weight == 0).sum()
            if args.branch_type == 'NeRV_vanilla':
                sparsity = sparsity_num / 1e6 / total_params
            elif args.branch_type == 'ERB':
                sparsity = sparsity_num / 1e6 / re_params

            if epoch == args.start_epoch:
                print(f'Model sparsity at Epoch{args.start_epoch}: {sparsity}')
                with open('{}/rank{}.txt'.format(args.outf, local_rank), 'a') as f:
                    f.write(f'Model sparsity at Epoch{args.start_epoch}: {sparsity}\n')
        
        epoch_start_time = datetime.now()
        psnr_list = []
        msssim_list = []
        # iterate over dataloader
        for i, (data,  norm_idx) in enumerate(train_dataloader):
            if i > 10 and args.debug:
                break
            embed_input = PE(norm_idx)
            if local_rank is not None:
                data = data.cuda(local_rank, non_blocking=True)
                embed_input = embed_input.cuda(local_rank, non_blocking=True)
            else:
                data, embed_input = data.cuda(non_blocking=True), embed_input.cuda(non_blocking=True)

            # forward and backward
            output_list = model(embed_input)
            target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]

            loss_list = [loss_fn(output, target, args) for output, target in zip(output_list, target_list)]
            loss_list = [loss_list[i] * (args.lw if i < len(loss_list) - 1 else 1) for i in range(len(loss_list))]
            loss_sum = sum(loss_list)

            lr = adjust_lr(optimizer, epoch % args.epochs, i, data_size, args)
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            # compute psnr and msssim
            psnr_list.append(psnr_fn(output_list, target_list))
            msssim_list.append(msssim_fn(output_list, target_list))
            if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
                train_psnr = torch.cat(psnr_list, dim=0) #(batchsize, num_stage)
                train_psnr = torch.mean(train_psnr, dim=0) #(num_stage)
                train_msssim = torch.cat(msssim_list, dim=0) #(batchsize, num_stage)
                train_msssim = torch.mean(train_msssim.float(), dim=0) #(num_stage)
                time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}, MSSSIM: {}'.format(
                    time_now_string, local_rank, epoch+1, args.epochs, i+1, len(train_dataloader), lr, 
                    RoundTensor(train_psnr, 2, False), RoundTensor(train_msssim, 4, False))
                print(print_str, flush=True)
                if local_rank in [0, 1, 2, 3, None]:
                    with open('{}/rank{}.txt'.format(args.outf, local_rank), 'a') as f:
                        f.write(print_str + '\n')

        # collect numbers from other gpus
        if args.distributed and args.ngpus_per_node > 1:
            train_psnr = all_reduce([train_psnr.to(local_rank)])
            train_msssim = all_reduce([train_msssim.to(local_rank)])

        # ADD train_PSNR TO TENSORBOARD
        if local_rank in [0, 1, 2, 3, None]:
            h, w = output_list[-1].shape[-2:]
            is_train_best = train_psnr[-1] > train_best_psnr
            train_best_psnr = train_psnr[-1] if train_psnr[-1] > train_best_psnr else train_best_psnr
            train_best_msssim = train_msssim[-1] if train_msssim[-1] > train_best_msssim else train_best_msssim
            writer.add_scalar(f'Train/PSNR_{h}X{w}_gap{args.frame_gap}', train_psnr[-1].item(), epoch+1)
            writer.add_scalar(f'Train/MSSSIM_{h}X{w}_gap{args.frame_gap}', train_msssim[-1].item(), epoch+1)
            writer.add_scalar(f'Train/best_PSNR_{h}X{w}_gap{args.frame_gap}', train_best_psnr.item(), epoch+1)
            writer.add_scalar(f'Train/best_MSSSIM_{h}X{w}_gap{args.frame_gap}', train_best_msssim, epoch+1)
            print_str = '\t{}p: current: {:.2f}\t best: {:.2f}\t msssim_best: {:.4f}\t'.format(h, train_psnr[-1].item(), train_best_psnr.item(), train_best_msssim.item())
            
            writer.add_scalar('Train/lr', lr, epoch+1)
            epoch_end_time = datetime.now()
            time_str = "Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format( (epoch_end_time - epoch_start_time).total_seconds(), \
                    (epoch_end_time - start).total_seconds() / (epoch + 1 - args.start_epoch) )
            print_str += time_str
            print(print_str, flush=True)
            with open('{}/rank{}.txt'.format(args.outf, local_rank), 'a') as f:
                f.write(print_str + '\n')

        state_dict = model.state_dict()
        save_checkpoint = {
            'epoch': epoch+1,
            'state_dict': state_dict,
            'train_best_psnr': train_best_psnr,
            'train_best_msssim': train_best_msssim,
            'val_best_psnr': val_best_psnr,
            'val_best_msssim': val_best_msssim,
            'optimizer': optimizer.state_dict(),   
        }    
        
        # evaluation
        if (epoch + 1) % args.eval_freq == 0 or epoch > total_epochs - 10:
            val_start_time = datetime.now()
            val_psnr, val_msssim = evaluate(model, val_dataloader, PE, local_rank, args)
            val_end_time = datetime.now()
            if args.distributed and args.ngpus_per_node > 1:
                val_psnr = all_reduce([val_psnr.to(local_rank)])
                val_msssim = all_reduce([val_msssim.to(local_rank)])            
            if local_rank in [0, 1, 2, 3, None]:
                # ADD val_PSNR TO TENSORBOARD
                h, w = output_list[-1].shape[-2:]
                print_str = f'Eval best_PSNR at epoch{epoch+1}:'
                is_val_best = val_psnr[-1] > val_best_psnr
                val_best_psnr = val_psnr[-1] if is_val_best else val_best_psnr
                val_best_msssim = val_msssim[-1] if val_msssim[-1] > val_best_msssim else val_best_msssim
                writer.add_scalar(f'Val/PSNR_{h}X{w}_gap{args.test_gap}', val_psnr[-1], epoch+1)
                writer.add_scalar(f'Val/MSSSIM_{h}X{w}_gap{args.test_gap}', val_msssim[-1], epoch+1)
                writer.add_scalar(f'Val/best_PSNR_{h}X{w}_gap{args.test_gap}', val_best_psnr, epoch+1)
                writer.add_scalar(f'Val/best_MSSSIM_{h}X{w}_gap{args.test_gap}', val_best_msssim, epoch+1)
                print_str += '\t{}p: current: {:.2f}\tbest: {:.2f} \tbest_msssim: {:.4f}\t Time/epoch: {:.2f}'.format(h, val_psnr[-1].item(),
                     val_best_psnr.item(), val_best_msssim.item(), (val_end_time - val_start_time).total_seconds())
                print(print_str)
                with open('{}/rank{}.txt'.format(args.outf, local_rank), 'a') as f:
                    f.write(print_str + '\n')
                if is_val_best:
                    torch.save(save_checkpoint, '{}/model_val_best.pth'.format(args.outf))

        if local_rank in [0, 1, 2, 3, None]:
            # state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.outf))
            if is_train_best:
                torch.save(save_checkpoint, '{}/model_train_best.pth'.format(args.outf))

    print_total_time = f'Training complete in: {str(datetime.now() - start)}'
    with open('{}/rank{}.txt'.format(args.outf, local_rank), 'a') as f:
        f.write(print_total_time + '\n')
    print(print_total_time)


@torch.no_grad()
def evaluate(model, val_dataloader, pe, local_rank, args):
    # Model Quantization
    if args.quant_bit != -1:
        cur_ckt = model.state_dict()
        from dahuffman import HuffmanCodec
        quant_weitht_list = []
        for k,v in cur_ckt.items():
            large_tf = (v.dim() in {2,4} and 'bias' not in k)
            quant_v, new_v = quantize_per_tensor(v, args.quant_bit, args.quant_axis if large_tf else -1)
            valid_quant_v = quant_v[v!=0] # only include non-zero weights
            quant_weitht_list.append(valid_quant_v.flatten())
            cur_ckt[k] = new_v
        cat_param = torch.cat(quant_weitht_list)
        input_code_list = cat_param.tolist()
        unique, counts = np.unique(input_code_list, return_counts=True)
        num_freq = dict(zip(unique, counts))

        # generating HuffmanCoding table
        codec = HuffmanCodec.from_data(input_code_list)
        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]
        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num]
        avg_bits = total_bits / len(input_code_list)    
        # import pdb; pdb.set_trace; from IPython import embed; embed()       
        encoding_efficiency = avg_bits / args.quant_bit
        print_str = f'Entropy encoding efficiency for bit {args.quant_bit}: {encoding_efficiency}'
        print(print_str)
        if local_rank in [0, 1, 2, 3, None]:
            with open('{}/rank{}.txt'.format(args.outf, local_rank), 'a') as f:
                f.write(print_str + '\n')       
        model.load_state_dict(cur_ckt)

    psnr_list = []
    msssim_list = []
    time_list = []
    model.eval()
    for i, (data,  norm_idx) in enumerate(val_dataloader):
        if i > 10 and args.debug:
            break
        embed_input = pe(norm_idx)
        if local_rank is not None:
            data = data.cuda(local_rank, non_blocking=True)
            embed_input = embed_input.cuda(local_rank, non_blocking=True)
        else:
            data, embed_input = data.cuda(non_blocking=True), embed_input.cuda(non_blocking=True)

        # compute psnr and msssim
        fwd_num = 10 if args.eval_fps else 1
        for _ in range(fwd_num):
            # embed_input = embed_input.half()
            # model = model.half()
            start_time = datetime.now()
            output_list = model(embed_input)
            torch.cuda.synchronize()
            # torch.cuda.current_stream().synchronize()
            time_list.append((datetime.now() - start_time).total_seconds())

        # compute MACs and Params
        if i == 0:
            input_tensor = embed_input.to(next(model.parameters()).device)
            macs, flops = profile(model, inputs=(embed_input,), verbose=False)
            # macs, flops = clever_format([macs, flops], "%.3f")
            print(f"MACs: {macs / 10 ** 9 :.2f}G")

            if hasattr(model, 'total_ops'):
                del model.total_ops
            if hasattr(model, 'total_params'):
                del model.total_params
        
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
            if local_rank in [0, 1, 2, 3, None]:
                with open('{}/rank{}.txt'.format(args.outf, local_rank), 'a') as f:
                    f.write(print_str + '\n')
    model.train()

    return val_psnr, val_msssim


if __name__ == '__main__':
    main()
