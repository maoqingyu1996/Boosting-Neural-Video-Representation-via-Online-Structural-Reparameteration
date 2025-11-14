import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import math

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform, vid_list=[None], frame_gap=1,  visualize=False):
        """ 
        功能：初始化自定义数据集，读取主目录下的所有图像帧并建立索引，支持按间隔采样与指定帧子集。
        参数：
            main_dir: 图像帧所在的主目录路径
            transform: 用于图像的预处理/增强的变换函数(例如 ToTensor、Normalize)
            vid_list: 可选的帧索引列表；如为默认 [None] 则使用全部帧
            frame_gap: 采样间隔，按该间隔从序列中取样
        返回：无（初始化内部状态）
        """
        self.main_dir = main_dir
        self.transform = transform
        frame_idx, self.frame_path = [], []
        accum_img_num = []
        all_imgs = os.listdir(main_dir)
        all_imgs.sort()

        num_frame = 0 
        for img_id in all_imgs:
            self.frame_path.append(img_id)      # 保存图像文件名
            frame_idx.append(num_frame)         # 保存当前帧序号
            num_frame += 1                    # 帧计数器自增

        # import pdb; pdb.set_trace; from IPython import embed; embed()
        accum_img_num.append(num_frame)
        self.frame_idx = [float(x) / len(frame_idx) for x in frame_idx]
        self.accum_img_num = np.asfarray(accum_img_num)
        # 如果 vid_list 不是默认的 [None]，则只保留指定索引的帧
        if None not in vid_list:
            self.frame_idx = [self.frame_idx[i] for i in vid_list]
        self.frame_gap = frame_gap

    def __len__(self):
        """
        功能：返回数据集中可被采样的样本数量。
        参数：无
        返回：整数，等于总帧数除以采样间隔 frame_gap
        """
        return len(self.frame_idx) // self.frame_gap

    def __getitem__(self, idx):
        """
        功能：根据给定索引返回一条样本，包括图像张量和归一化后的帧索引。
        参数：
            idx: 数据集中样本的索引（已按 frame_gap 计算）
        返回：
            tensor_image: 处理后的图像张量，形状为 [C, H, W] 或经必要转置
            frame_idx: 该样本对应的归一化帧索引标量张量
        """
        valid_idx = idx * self.frame_gap
        img_id = self.frame_path[valid_idx] # 获取对应帧文件名
        img_name = os.path.join(self.main_dir, img_id) # 拼接完整路径
        image = Image.open(img_name).convert("RGB") # 打开图像并转换为 RGB 模式
        tensor_image = self.transform(image) # 对图像应用预处理变换
        if tensor_image.size(1) > tensor_image.size(2): # 若高维大于宽维，转置
            tensor_image = tensor_image.permute(0,2,1)
        frame_idx = torch.tensor(self.frame_idx[valid_idx]) # 归一化后的帧索引

        return tensor_image, frame_idx # 返回处理后的图像张量与归一化后的帧索引

class Sin(nn.Module):
    def __init__(self, inplace: bool = False):
        """
        功能：初始化正弦激活模块(占位，未使用 inplace)。
        参数：
            inplace: 是否原地操作（此处未使用）
        返回：无
        """
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(input)


def ActivationLayer(act_type):  
    """
    功能：根据字符串标识返回对应的激活函数层/函数。
    参数：
        act_type: 激活类型字符串，如 'relu'、'leaky'、'gelu'、'sin' 等
    返回：
        对应的 PyTorch 激活层或函数
    异常：
        KeyError: 当传入未知激活类型时抛出
    """
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        act_layer = torch.sin
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer


def NormLayer(norm_type, ch_width): 
    """
    功能：根据字符串标识返回对应的归一化层。
    参数：
        norm_type: 归一化类型字符串，'none'、'bn'（批归一化）或 'in'（实例归一化）
        ch_width: 通道数，用于构造归一化层
    返回：
        对应的归一化层模块
    异常：
        NotImplementedError: 当传入不支持的归一化类型时抛出
    """
    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer


class CustomConv(nn.Module):  
    def __init__(self, **kargs):
        """
        功能：构建自定义卷积模块，支持普通卷积+像素重排、反卷积以及双线性上采样。
        参数（通过 kargs 提供）：
            ngf: 输入通道数
            new_ngf: 输出通道基数（与 stride 组合决定最终输出通道）
            stride: 步幅或上采样倍率
            conv_type: 'conv'、'deconv' 或 'bilinear' 指定分支
            bias: 是否使用偏置（用于部分分支）
        返回：无
        """
        super(CustomConv, self).__init__()

        ngf, new_ngf, stride = kargs['ngf'], kargs['new_ngf'], kargs['stride']
        self.conv_type = kargs['conv_type']
        if self.conv_type == 'conv':
            self.conv = nn.Conv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'])
            self.up_scale = nn.PixelShuffle(stride)
        elif self.conv_type == 'deconv':
            self.conv = nn.ConvTranspose2d(ngf, new_ngf, stride, stride)
            self.up_scale = nn.Identity()
        elif self.conv_type == 'bilinear':
            self.conv = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
            self.up_scale = nn.Conv2d(ngf, new_ngf, 2*stride+1, 1, stride, bias=kargs['bias'])

    def forward(self, x):
        out = self.conv(x)
        return self.up_scale(out)


def MLP(dim_list, act='relu', bias=True):  # 定义多层感知机（MLP），支持不同激活函数
    """
    功能：根据维度列表构建顺序的线性层 + 激活函数的 MLP。
    参数：
        dim_list: 各层维度列表，例如 [in_dim, h1, h2, ..., out_dim]
        act: 激活函数类型字符串，传给 ActivationLayer
        bias: 线性层是否使用偏置
    返回：
        nn.Sequential 封装的 MLP 模型
    """
    act_fn = ActivationLayer(act)
    fc_list = []
    for i in range(len(dim_list) - 1):
        fc_list += [nn.Linear(dim_list[i], dim_list[i+1], bias=bias), act_fn]
    return nn.Sequential(*fc_list)


class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type # 卷积类型，如 'conv1x1-sobelx' 或 'conv1x1-sobely'
        self.inp_planes = inp_planes # 输入通道数
        self.out_planes = out_planes # 输出通道数
        if self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0) # 1x1 卷积，用于提取特征
            self.k0 = conv0.weight # 1x1 卷积核权重
            self.b0 = conv0.bias # 1x1 卷积核偏置

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3 # 1x1 卷积核缩放因子
            self.scale = nn.Parameter(scale) # 变成网络参数

            bias = torch.randn(self.out_planes) * 1e-3 # 1x1 卷积核偏置
            bias = torch.reshape(bias, (self.out_planes,)) # 1x1 卷积核偏置向量
            self.bias = nn.Parameter(bias) # 变成网络参数

            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32) # 3x3 卷积核掩码
            for i in range(self.out_planes): # 对每个输出通道初始化 3x3 卷积核掩码
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False) # 变成网络参数

        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))

            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        # conv-1x1
        y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1) # 1x1 卷积，提取特征
        # explicitly padding with bias
        y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0) # 对特征图进行填充，填充值为 0
        b0_pad = self.b0.view(1, -1, 1, 1) # 填充以便后续3*3卷积
        y0[:, :, 0:1, :] = b0_pad
        y0[:, :, -1:, :] = b0_pad
        y0[:, :, :, 0:1] = b0_pad
        y0[:, :, :, -1:] = b0_pad
        # conv-3x3
        y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)  
        return y1

    def rep_params(self):
        device = self.k0.get_device() # 获取卷积核权重的设备
        if device < 0:
            device = None # 如果设备为 -1，则设为 None
        tmp = self.scale * self.mask # 计算 3x3 卷积核权重
        k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device) # 初始化 3x3 卷积核权重
        for i in range(self.out_planes): # 
            k1[i, i, :, :] = tmp[i, 0, :, :] # 
        b1 = self.bias
        # re-param conv kernel
        RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3)) 
        # re-param conv bias
        RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
        RB = F.conv2d(input=RB, weight=k1).view(-1, ) + b1
        return RK, RB

################################################ Modified
class NeRVBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        self.ngf, self.new_ngf, self.stride = kargs['ngf'], kargs['new_ngf'], kargs['stride']
        self.deploy = kargs['deploy']
        self.branch_type = kargs['branch_type']
        self.up_scale = nn.PixelShuffle(self.stride)
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])

        self.out_channels = self.new_ngf * self.stride * self.stride

        if self.deploy: # 部署模式下，使用单分支卷积核
            self.rbr_reparam = nn.Conv2d(in_channels=self.ngf, out_channels=self.out_channels, kernel_size=(3, 3), 
                                         stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        else: # 训练模式下，使用多分支卷积核
            if self.branch_type == "NeRV_vanilla":
                self.branch = nn.Conv2d(in_channels=self.ngf, out_channels=self.out_channels, kernel_size=(3, 3),
                                        stride=1, padding=1, dilation=1, groups=1, bias=kargs['bias'])

            elif self.branch_type == "ERB":
                #  三个独立的分支，分别是3x3 1x3 和 3x1 卷积
                self.rbr_3x3_branch = nn.Conv2d(in_channels=self.ngf, out_channels=self.out_channels, kernel_size=(3, 3),
                                                stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros')

                self.rbr_3x1_branch = nn.Conv2d(in_channels=self.ngf, out_channels=self.out_channels, kernel_size=(3, 1),
                                                stride=1, padding=(1, 0), dilation=1, groups=1, padding_mode='zeros')

                self.rbr_1x3_branch = nn.Conv2d(in_channels=self.ngf, out_channels=self.out_channels, kernel_size=(1, 3),
                                                stride=1, padding=(0, 1), dilation=1, groups=1, padding_mode='zeros')
                # 序列卷积，顺序为1x1 3x3 1x1
                self.rbr_1x1_3x3_1x1_branch_1x1_1 = nn.Conv2d(in_channels=self.ngf, out_channels=2 * self.ngf, kernel_size=(1, 1),
                                                              stride=1, padding=(0, 0), dilation=1, groups=1,
                                                              padding_mode='zeros', bias=False)
                self.rbr_1x1_3x3_1x1_branch_3x3 = nn.Conv2d(in_channels=2 * self.ngf, out_channels=self.out_channels, kernel_size=(3, 3),
                                                              stride=1, padding=1, dilation=1, groups=1, 
                                                              padding_mode='zeros', bias=False)
                self.rbr_1x1_3x3_1x1_branch_1x1_2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(1, 1),
                                                              stride=1, padding=(0, 0), dilation=1, groups=1,
                                                              padding_mode='zeros', bias=False)
            
            elif self.branch_type == "ACB":
                # 三个卷积分支，分别是3x3 1x3 和 3x1 卷积
                self.rbr_3x3_branch = nn.Conv2d(in_channels=self.ngf, out_channels=self.out_channels, kernel_size=(3, 3),
                                                stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros')

                self.rbr_3x1_branch = nn.Conv2d(in_channels=self.ngf, out_channels=self.out_channels, kernel_size=(3, 1),
                                                stride=1, padding=(1, 0), dilation=1, groups=1, padding_mode='zeros')

                self.rbr_1x3_branch = nn.Conv2d(in_channels=self.ngf, out_channels=self.out_channels, kernel_size=(1, 3),
                                                stride=1, padding=(0, 1), dilation=1, groups=1, padding_mode='zeros')
            
            elif self.branch_type == "RepVGG":
                ### Without identity
                ### 两个卷积分支，分别是3x3 1x1
                self.rbr_3x3_branch = nn.Conv2d(in_channels=self.ngf, out_channels=self.out_channels, kernel_size=(3, 3),
                                                stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros')

                self.rbr_1x1_branch = nn.Conv2d(in_channels=self.ngf, out_channels=self.out_channels, kernel_size=(1, 1),
                                                stride=1, padding=(0, 0), dilation=1, groups=1, padding_mode='zeros')

            elif self.branch_type == "DBB":
                # 两个卷积分支，分别是3x3 1x1
                self.rbr_3x3_branch = nn.Conv2d(in_channels=self.ngf, out_channels=self.out_channels, kernel_size=(3, 3),
                                                stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros')

                self.rbr_1x1_branch = nn.Conv2d(in_channels=self.ngf, out_channels=self.out_channels, kernel_size=(1, 1),
                                                stride=1, padding=(0, 0), dilation=1, groups=1, padding_mode='zeros')

                self.rbr_1x1_3x3_branch_1x1 = nn.Conv2d(in_channels=self.ngf, out_channels=2 * self.ngf, kernel_size=(1, 1),
                                                        stride=1, padding=(0, 0), dilation=1, groups=1, padding_mode='zeros', bias=False)
                self.rbr_1x1_3x3_branch_3x3 = nn.Conv2d(in_channels=2 * self.ngf, out_channels=self.out_channels, kernel_size=(3, 3),
                                                        stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', bias=False)

                self.rbr_1x1_avg_branch_1x1 = nn.Conv2d(in_channels=self.ngf, out_channels=self.out_channels, kernel_size=(1, 1),
                                                        stride=1, padding=(0, 0), dilation=1, groups=1, padding_mode='zeros', bias=False)
                self.rbr_1x1_avg_branch_avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

            elif self.branch_type == "ECB":
                self.rbr_3x3_branch = nn.Conv2d(in_channels=self.ngf, out_channels=self.out_channels, kernel_size=(3, 3),
                                                stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros')
                
                self.rbr_1x1_3x3_branch_1x1 = nn.Conv2d(in_channels=self.ngf, out_channels=2 * self.ngf, kernel_size=(1, 1),
                                                        stride=1, padding=(0, 0), dilation=1, groups=1, padding_mode='zeros', bias=False)
                self.rbr_1x1_3x3_branch_3x3 = nn.Conv2d(in_channels=2 * self.ngf, out_channels=self.out_channels, kernel_size=(3, 3),
                                                        stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', bias=False)

                self.rbr_conv1x1_sbx_branch = SeqConv3x3('conv1x1-sobelx', self.ngf, self.out_channels)
                self.rbr_conv1x1_sby_branch = SeqConv3x3('conv1x1-sobely', self.ngf, self.out_channels)
                self.rbr_conv1x1_lpl_branch = SeqConv3x3('conv1x1-laplacian', self.ngf, self.out_channels)

    def switch_to_deploy(self):
        """
        功能：将训练阶段的多分支结构融合为单一 3×3 卷积，切换至部署/推理模式。
        步骤：
            1. 调用 get_equivalent_kernel_bias() 计算融合后的等效卷积核与偏置；
            2. 若尚未创建部署态卷积层 rbr_reparam，则实例化一个 3×3 卷积；
            3. 将融合得到的 kernel 与 bias 拷贝至部署层；
            4. 删除所有训练分支子模块，释放显存并简化图结构；
            5. 将 self.deploy 置 True，标记当前处于部署态。
        返回：无
        """
        # 幂等保护：若已是部署态或训练分支已删除，直接返回
        # 已部署态应具备 rbr_reparam，且训练分支如 rbr_3x3_branch 可能已不存在
        if getattr(self, 'deploy', False) or not hasattr(self, 'rbr_3x3_branch'):
            if hasattr(self, 'rbr_reparam'):
                self.deploy = True
            return
        # 将多分支结构融合为单一 3x3 卷积，用于部署/推理阶段
        # 先计算各分支融合后的等效卷积核和偏置
        kernel, bias = self.get_equivalent_kernel_bias()

        # 如果尚未创建部署态的单层卷积，则在此实例化
        if not hasattr(self, 'rbr_reparam'):
            self.rbr_reparam = nn.Conv2d(
                in_channels=self.ngf,
                out_channels=self.out_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros'
            )

        # 将融合后的参数拷贝到部署态的卷积层
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        # 删除训练态的分支模块，释放显存与简化结构（按存在与否安全删除）
        for name in [
            'rbr_3x3_branch', 'rbr_3x1_branch', 'rbr_1x3_branch',
            'rbr_1x1_3x3_1x1_branch_1x1_1', 'rbr_1x1_3x3_1x1_branch_3x3', 'rbr_1x1_3x3_1x1_branch_1x1_2',
            'branch',  # NeRV_vanilla
            'rbr_1x1_branch',  # RepVGG
            'rbr_1x1_3x3_branch_1x1', 'rbr_1x1_3x3_branch_3x3',  # DBB/ECB
            'rbr_1x1_avg_branch_1x1', 'rbr_1x1_avg_branch_avg',  # DBB
            'rbr_conv1x1_sbx_branch', 'rbr_conv1x1_sby_branch', 'rbr_conv1x1_lpl_branch'  # ECB
        ]:
            if hasattr(self, name):
                self.__delattr__(name)

        # 标记为部署态
        self.deploy = True

    def get_equivalent_kernel_bias(self):
        """
        功能：将 ERB 分支中所有子分支（3x3、1x3+3x1、1x1→3x3→1x1）的权重与偏置等效融合为单个 3x3 卷积的参数。
        步骤：
            1. 提取 3x3 分支的 kernel/bias；
            2. 调用 _fuse_1x3_3x1_branch 融合 1x3 与 3x1 分支；
            3. 调用 _fuse_1x1_3x3_1x1_branch 融合 1x1→3x3→1x1 序列；
            4. 将三组结果逐元素相加，得到最终等效 kernel 与 bias。
        返回：
            fused_kernel: 融合后的 3x3 卷积权重，形状 [out_channels, in_channels, 3, 3]
            fused_bias  : 融合后的 3x3 卷积偏置，形状 [out_channels]
        """
        ### 3x3 branch
        kernel_3x3, bias_3x3 = self.rbr_3x3_branch.weight, self.rbr_3x3_branch.bias
        device = kernel_3x3.device

        ### 1x3 3x1 branch
        kernel_1x3_3x1_fuse, bias_1x3_3x1_fuse = self._fuse_1x3_3x1_branch(self.rbr_1x3_branch,
                                                                           self.rbr_3x1_branch)

        ### 1x1+3x3+1x1 branch
        kernel_1x1_3x3_1x1_fuse = self._fuse_1x1_3x3_1x1_branch(self.rbr_1x1_3x3_1x1_branch_1x1_1,
                                                                self.rbr_1x1_3x3_1x1_branch_3x3,
                                                                self.rbr_1x1_3x3_1x1_branch_1x1_2)

        fused_kernel = kernel_3x3 + kernel_1x3_3x1_fuse.to(device) + kernel_1x1_3x3_1x1_fuse.to(device)
        fused_bias = bias_3x3 + bias_1x3_3x1_fuse

        return fused_kernel, fused_bias

    def _fuse_1x3_3x1_branch(self, conv1, conv2):
        """
        功能：将 1x3 与 3x1 两个卷积的权重和偏置融合为等效 3x3 卷积的参数。
        参数：
            conv1: 1x3 卷积层，其 weight 形状为 [out_c, in_c, 1, 3]
            conv2: 3x1 卷积层，其 weight 形状为 [out_c, in_c, 3, 1]
        返回：
            weight: 融合后的 3x3 卷积权重，形状为 [out_c, in_c, 3, 3]
            bias: 融合后的 3x3 卷积偏置，形状为 [out_c]
        说明：
            1. 对 conv1 的权重在高度方向两侧各填充 1 个零，使其变为 [out_c, in_c, 3, 3]；
            2. 对 conv2 的权重在宽度方向两侧各填充 1 个零，使其变为 [out_c, in_c, 3, 3]；
            3. 将两者相加即得等效 3x3 卷积核；
            4. 偏置直接相加即可。
        """
        weight = F.pad(conv1.weight, (0, 0, 1, 1)) + F.pad(conv2.weight, (1, 1, 0, 0))
        bias = conv1.bias + conv2.bias
        return weight, bias

    def _fuse_1x1_3x3_1x1_branch(self, conv1, conv2, conv3):
        """
        功能：将连续的 1x1 → 3x3 → 1x1 三个卷积层等效融合为单个 3x3 卷积核。
        参数：
            conv1: 第一个 1x1 卷积层
            conv2: 中间的 3x3 卷积层
            conv3: 最后的 1x1 卷积层
        返回：
            weight: 融合后的 3x3 卷积核权重，形状为 [out_c, in_c, 3, 3]
        """
        # 将 conv1 的权重作为“卷积核”对 conv2 的权重做卷积，实现 1x1 与 3x3 的融合
        tmp = F.conv2d(conv2.weight, conv1.weight.permute(1, 0, 2, 3))
        K0 = tmp.permute(2, 3, 0, 1)  # 调整维度顺序，便于后续矩阵乘法
        # 将 conv3 的 1x1 权重扩展为 3x3 的等效矩阵（重复 3x3 次）
        K1 = conv3.weight.permute(2, 3, 0, 1).repeat(3, 3, 1, 1)
        # 矩阵乘法完成最终融合，再恢复标准卷积核维度
        weight = torch.matmul(K1, K0).permute(2, 3, 0, 1)
        return weight

    def forward(self, x):
        if self.deploy:
            output = self.rbr_reparam(x)
        else:
            if self.branch_type == "NeRV_vanilla":
                output = self.branch(x)

            elif self.branch_type == "ERB":
                ### offline re-parameterization
                # op0 = self.rbr_3x3_branch(x)
                # op1 = self.rbr_3x1_branch(x)
                # op2 = self.rbr_1x3_branch(x)
                # op3 = self.rbr_1x1_3x3_1x1_branch_1x1_2(self.rbr_1x1_3x3_1x1_branch_3x3(self.rbr_1x1_3x3_1x1_branch_1x1_1(x)))
                # output = op0 + op1 + op2 + op3

                ### online re-parameterization
                weight, bias = self.get_equivalent_kernel_bias()
                # 设备一致性保护：确保等效卷积核与偏置与输入张量处于同一设备
                dev = x.device
                weight = weight.to(dev)
                bias = bias.to(dev) if bias is not None else None
                output = F.conv2d(x, weight, bias, stride=1, padding=1, dilation=1, groups=1)

            elif self.branch_type == "ACB": 
                op0 = self.rbr_3x3_branch(x)
                op1 = self.rbr_3x1_branch(x)
                op2 = self.rbr_1x3_branch(x)
                output = op0 + op1 + op2

            elif self.branch_type == "RepVGG": 
                op0 = self.rbr_3x3_branch(x)
                op1 = self.rbr_1x1_branch(x)
                output = op0 + op1

            elif self.branch_type == "DBB": 
                op0 = self.rbr_3x3_branch(x)
                op1 = self.rbr_1x1_branch(x)
                op2 = self.rbr_1x1_3x3_branch_3x3(self.rbr_1x1_3x3_branch_1x1(x))
                op3 = self.rbr_1x1_avg_branch_avg(self.rbr_1x1_avg_branch_1x1(x))
                output = op0 + op1 + op2 + op3
                
            elif self.branch_type == "ECB":
                op0 = self.rbr_3x3_branch(x)
                op1 = self.rbr_1x1_3x3_branch_3x3(self.rbr_1x1_3x3_branch_1x1(x))
                op2 = self.rbr_conv1x1_sbx_branch(x)
                op3 = self.rbr_conv1x1_sby_branch(x)
                op4 = self.rbr_conv1x1_lpl_branch(x)
                output = op0 + op1 + op2 + op3 + op4

        return self.act(self.norm(self.up_scale(output))) # 上采样后激活归一化
#################################################


class Generator(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        stem_dim, stem_num = [int(x) for x in kargs['stem_dim_num'].split('_')] # 解析 stem 层的维度和数量
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in kargs['fc_hw_dim'].split('_')] # 解析MLP层的高度、宽度和维度
        mlp_dim_list = [kargs['embed_length']] + [stem_dim] * stem_num + [self.fc_h *self.fc_w *self.fc_dim] # 构建MLP层的维度列表
        self.stem = MLP(dim_list=mlp_dim_list, act=kargs['act']) # 构建 stem 层
        
        # BUILD CONV LAYERS
        self.layers, self.head_layers = [nn.ModuleList() for _ in range(2)]
        ngf = self.fc_dim
        for i, stride in enumerate(kargs['stride_list']):
            if i == 0:
                # expand channel width at first stage
                new_ngf = int(ngf * kargs['expansion'])
            else:
                # change the channel width for each stage
                new_ngf = max(ngf // (1 if stride == 1 else kargs['reduction']), kargs['lower_width'])

            for j in range(kargs['num_blocks']):
                self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=1 if j else stride,
                                             bias=kargs['bias'], norm=kargs['norm'], act=kargs['act'], 
                                             deploy=kargs['deploy'], conv_type=kargs['conv_type'], branch_type=kargs['branch_type']))
                ngf = new_ngf

            # build head classifier, upscale feature layer, upscale img layer 
            head_layer = [None]
            if kargs['sin_res']:
                if i == len(kargs['stride_list']) - 1:
                    head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=kargs['bias']) 
                    # head_layer = nn.Conv2d(ngf, 3, 3, 1, 1, bias=kargs['bias']) 
                else:
                    head_layer = None
            else:
                head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=kargs['bias'])
                # head_layer = nn.Conv2d(ngf, 3, 3, 1, 1, bias=kargs['bias'])
            self.head_layers.append(head_layer)
        self.sigmoid =kargs['sigmoid']

    def forward(self, input):
        output = self.stem(input)
        output = output.view(output.size(0), self.fc_dim, self.fc_h, self.fc_w)

        out_list = []

        for layer, head_layer in zip(self.layers, self.head_layers):
            output = layer(output)

            if head_layer is not None:
                img_out = head_layer(output)
                img_out = torch.sigmoid(img_out) if self.sigmoid else (torch.tanh(img_out) + 1) * 0.5
                out_list.append(img_out)

        return  out_list

