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
from copy import deepcopy

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# *! 该文件仅仅用于训练模型，不包含推理/评估代码
# ** 功能说明：
# ** 1. 读取视频帧数据，按指定 gap 采样，支持数据增强与多尺度训练；
# ** 2. 构建 NeRV/RepNeRV 生成器，可选 ERB/RepVGG/DBB 等重参数分支；
# ** 3. 采用多阶段损失加权，实时计算 PSNR、MS-SSIM 并写入 TensorBoard；
# ** 4. 按 epoch 自动保存最新/最佳训练态与部署态（重参数后）权重；
# ** 5. 支持剪枝、量化、学习率 warmup、cosine/step 衰减等训练策略；
# ** 6. 训练日志、模型结构、参数量、耗时等信息均落盘到 outf/rank*.txt，方便后续排查与对比。
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
    
    torch.set_printoptions(precision=4) 
    
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

    exp_id = f'{args.dataset}/embed{args.embed}_{args.stem_dim_num}_fc_{args.fc_hw_dim}__exp{args.expansion}_reduce{args.reduction}_low{args.lower_width}_blk{args.num_blocks}' + \
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
    local_rank = torch.cuda.current_device() if torch.cuda.is_available() else 0
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
        # 输出模型结构和参数量
        print(f'{args}\n {model}\n Model Params: {total_params}M')
        with open('{}/rank{}.txt'.format(args.outf, local_rank), 'a') as f:
            f.write(str(model) + '\n' + f'Params: {total_params}M\n')
        writer = SummaryWriter(os.path.join(args.outf, f'param_{total_params}M', 'tensorboard'))
    else:
        writer = None
    
    # 输出当前使用的GPU设备
    print("Use GPU: {} for training".format(local_rank))
    device = torch.device('cuda:{}'.format(local_rank))
    model = model.to(device)
    
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), betas=(args.beta, 0.999))
    loc = 'cuda:{}'.format(local_rank if local_rank is not None else 0)
    
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
    
    # Training
    start = datetime.now() # 获取当前时间
    total_epochs = args.epochs # 计算总的训练轮数
    args.start_epoch = 0
    for epoch in range(args.start_epoch, total_epochs):
        # 训练模式
        model.train()
        epoch_start_time = datetime.now()
        psnr_list = []
        msssim_list = []
        
        for i, (data,  norm_idx) in enumerate(train_dataloader):
            if i > 10 and args.debug:
                break
            
            # get embed_input and move to GPU
            embed_input = PE(norm_idx)
            data, embed_input = data.cuda(non_blocking=True), embed_input.cuda(non_blocking=True)
            
            # forward
            output_list = model(embed_input)
            target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
            
            # calculate loss
            loss_list = [loss_fn(output, target, args) for output, target in zip(output_list, target_list)]
            loss_list = [loss_list[i] * (args.lw if i < len(loss_list) - 1 else 1) for i in range(len(loss_list))]
            loss_sum = sum(loss_list)
            
            # backward and update
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
            
            # ** 打印分辨率、当前PSNR、最佳PSNR、最佳MSSSIM
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
                
        if (epoch + 1) % args.eval_freq == 0 or epoch > total_epochs - 10:
            val_start_time = datetime.now()
            val_psnr, val_msssim = evaluate(model, val_dataloader, PE, local_rank, args)
            val_end_time = datetime.now()
            
            if local_rank in [0, 1, 2, 3, None]:
                h, w = output_list[-1].shape[-2:]
                print_str = f'Eval best_PSNR at epoch{epoch+1}:'
                is_val_best = val_psnr[-1] > val_best_psnr
                val_best_psnr = val_psnr[-1] if is_val_best else val_best_psnr
                val_best_msssim = val_msssim[-1] if val_msssim[-1] > val_best_msssim else val_best_msssim
                writer.add_scalar(f'Val/PSNR_{h}X{w}_gap{args.test_gap}', val_psnr[-1], epoch+1)
                writer.add_scalar(f'Val/MSSSIM_{h}X{w}_gap{args.test_gap}', val_msssim[-1], epoch+1)
                writer.add_scalar(f'Val/best_PSNR_{h}X{w}_gap{args.test_gap}', val_best_psnr, epoch+1)
                writer.add_scalar(f'Val/best_MSSSIM_{h}X{w}_gap{args.test_gap}', val_best_msssim, epoch+1)
                print_str += '\t{}p: current: {:.2f}\tbest: {:.2f} \tbest_msssim: {:.4f}\t Time/epoch: {:.2f}'.format(h, val_psnr[-1].item(), val_best_psnr.item(), val_best_msssim.item(), (val_end_time - val_start_time).total_seconds())
                print(print_str)
                with open('{}/rank{}.txt'.format(args.outf, local_rank), 'a') as f:f.write(print_str + '\n')
                if is_val_best:
                    torch.save(save_checkpoint, '{}/model_val_best.pth'.format(args.outf))
                            
        # *! 保存ERB重参数化模型，分别保存训练态模型和部署态模型。
        if local_rank in [0, 1, 2, 3, None] and args.branch_type == 'ERB':
            # 保存训练态模型
            torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.outf))
            if is_train_best:
                torch.save(save_checkpoint, '{}/model_train_best.pth'.format(args.outf))
            
            # 将模型调整为部署态
            copy_model = deepcopy(model)
            for layer in copy_model.layers:
                if isinstance(layer, NeRVBlock):
                    layer.switch_to_deploy()  # 将多分支融合为 rbr_reparam
            model_is_deploy = True
            
            deploy_dict = copy_model.state_dict()
            deploy_checkpoint = {
                'epoch': epoch+1,
                'state_dict': deploy_dict,
                'train_best_psnr': train_best_psnr,
                'train_best_msssim': train_best_msssim,
                'val_best_psnr': val_best_psnr,
                'val_best_msssim': val_best_msssim,
                'optimizer': optimizer.state_dict(),   
                }
            # 保存部署态模型
            torch.save(deploy_checkpoint, '{}/model_latest_deploy.pth'.format(args.outf))
            if is_train_best:
                torch.save(deploy_checkpoint, '{}/model_train_best_deploy.pth'.format(args.outf))
                
        # ** NeRV或其他模型均保存训练态模型
        else:
            model_is_deploy = False
            torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.outf))
            if is_train_best:
                torch.save(save_checkpoint, '{}/model_train_best.pth'.format(args.outf))
                
    # 如果是重参数模型
    if model_is_deploy:
        # 计算部署态 copy_model 的参数量并打印（单位：百万），方便对比训练态/部署态参数规模
        deploy_total_params = sum(p.data.nelement() for p in copy_model.parameters()) / 1e6
        rep_params_str = f'Deploy Rep-Model Params: {deploy_total_params:.3f}M'
        with open('{}/rank{}.txt'.format(args.outf, local_rank), 'a') as f:
            f.write(rep_params_str + '\n')
        print(rep_params_str)
    
    # 输出总训练时间并写入
    print_total_time = f'Training complete in: {str(datetime.now() - start)}'
    with open('{}/rank{}.txt'.format(args.outf, local_rank), 'a') as f:
        f.write(print_total_time + '\n')
    print(print_total_time)

                    

@torch.no_grad()
def evaluate(model, val_dataloader, pe, local_rank, args):
    # 初始化指标表格，并将模型转到eval
    psnr_list = []
    msssim_list = []
    time_list = []
    model.eval()
    
    for i, (data,  norm_idx) in enumerate(val_dataloader):
        # 将输入和数据转到GPU
        if i > 10 and args.debug:
            break
        embed_input = pe(norm_idx)
        if local_rank is not None:
            data = data.cuda(local_rank, non_blocking=True)
            embed_input = embed_input.cuda(local_rank, non_blocking=True)
        else:
            data, embed_input = data.cuda(non_blocking=True), embed_input.cuda(non_blocking=True)
        
        # 得到输出，重复fwd_num次以计算推理时间
        fwd_num = 10 if args.eval_fps else 1
        for _ in range(fwd_num):
            # embed_input = embed_input.half()
            # model = model.half()
            start_time = datetime.now()
            output_list = model(embed_input)
            torch.cuda.synchronize()
            # torch.cuda.current_stream().synchronize()
            time_list.append((datetime.now() - start_time).total_seconds())
        
        # 计算MACs
        if i == 0:
            input_tensor = embed_input.to(next(model.parameters()).device)
            macs, flops = profile(model, inputs=(embed_input,), verbose=False)
            # macs, flops = clever_format([macs, flops], "%.3f")
            print(f"MACs: {macs / 10 ** 9 :.2f}G")

            if hasattr(model, 'total_ops'):
                del model.total_ops
            if hasattr(model, 'total_params'):
                del model.total_params
                
        # 计算PSNR和MS-SSIM
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
                
        
                    
                        
                        
                    
                
            
            
            

        
                
    
    
    
    

