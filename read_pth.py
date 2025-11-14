#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
读取并解析 .pth/.pt 检查点文件的小脚本。

功能概述：
- 安全加载权重文件（优先使用 weights_only=True，兼容旧版 PyTorch）。
- 打印检查点顶层键；若存在 'state_dict'，打印其中参数的名称、形状和数据类型。
- 统计并打印 state_dict 中的张量数量与总元素个数（参数规模）。

使用示例：
    python read_pth.py checkpoints/nerv_erb_S.pth
    python read_pth.py RESULT/xxx/model_latest.pth --limit 100
"""

import argparse
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, Tuple

import torch


def safe_load_checkpoint(path: str) -> Dict[str, Any]:
    """
    安全加载检查点文件。

    说明：
    - 优先使用 `torch.load(..., weights_only=True)`，降低反序列化任意对象的风险。
    - 兼容旧版 PyTorch：若不支持该参数，会回退到普通的 `torch.load(...)`。
    - 始终将张量加载到 CPU（map_location='cpu'），避免 GPU 不可用或设备不一致问题。

    参数：
    - path: 检查点文件路径（.pth/.pt）。

    返回：
    - checkpoint 字典对象（通常包含 'state_dict' 等）。
    """
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location='cpu')
    return ckpt


def summarize_state_dict(state_dict: Dict[str, torch.Tensor], limit: int = 50) -> Tuple[int, int]:
    """
    遍历并打印 state_dict 的前若干项，输出名称、形状、dtype。

    参数：
    - state_dict: 由参数名到张量的映射（OrderedDict 或普通 dict）。
    - limit: 打印的最大条目数，避免过长输出。

    返回：
    - (num_tensors, total_elements): 张量数量与总元素个数，用于统计规模。
    """
    num_tensors = 0
    total_elements = 0

    # 为兼容 OrderedDict 或 dict，统一遍历 items
    items = list(state_dict.items())
    if len(items) == 0:
        print("state_dict 为空。")
        return 0, 0

    print(f"\nstate_dict 前 {min(limit, len(items))} 项预览：")
    for i, (name, tensor) in enumerate(items):
        if i >= limit:
            break
        if isinstance(tensor, torch.Tensor):
            shape = tuple(tensor.shape)
            dtype = str(tensor.dtype).replace('torch.', '')
            print(f"  - {name}: shape={shape}, dtype={dtype}")
            num_tensors += 1
            try:
                total_elements += tensor.numel()
            except Exception:
                pass
        else:
            # 非张量情况（极少见），仍做提示
            print(f"  - {name}: 非张量类型 ({type(tensor)})")

    # 若有剩余未打印的项，提示数量
    if len(items) > limit:
        print(f"  ... 还有 {len(items) - limit} 项未显示")

    return num_tensors, total_elements


def print_top_keys(ckpt: Dict[str, Any]) -> None:
    """
    打印检查点顶层键，并给出常见结构提示。

    说明：
    - 顶层通常包含 'state_dict'、'epoch'、'optimizer'、'train_best_psnr' 等。
    - 若存在 'state_dict'，提示其类型与长度；否则直接打印全部顶层键。
    """
    print("顶层键：")
    keys = list(ckpt.keys())
    if not keys:
        print("  （空字典）")
        return

    for k in keys:
        v = ckpt[k]
        tname = type(v).__name__
        if k == 'state_dict' and isinstance(v, (dict, OrderedDict)):
            print(f"  - {k}: {tname}, 项数={len(v)}")
        else:
            # 对标量或简单类型做简要展示
            brief = v
            try:
                # 避免打印过长对象，仅展示摘要
                if isinstance(v, (list, tuple, dict, OrderedDict)):
                    brief = f"{tname}(len={len(v)})"
            except Exception:
                brief = tname
            print(f"  - {k}: {brief}")


def main():
    """
    主函数：解析命令行参数，加载 .pth/.pt 文件并打印摘要信息。

    参数：
    - 位置参数 path：待阅读的检查点路径。
    - 可选参数 --limit：state_dict 预览条数（默认 50）。
    """
    parser = argparse.ArgumentParser(description="读取并解析 .pth/.pt 检查点文件")
    parser.add_argument("path", type=str, help=".pth/.pt 文件路径")
    parser.add_argument("--limit", type=int, default=50, help="state_dict 预览的最大条目数")
    args = parser.parse_args()

    path = args.path
    if not os.path.isfile(path):
        print(f"文件不存在：{path}")
        sys.exit(1)

    print(f"读取文件：{path}")
    ckpt = safe_load_checkpoint(path)

    # 顶层结构展示
    print_top_keys(ckpt)

    # 若包含 state_dict，则进一步展示参数信息
    state = ckpt.get('state_dict', None)
    if isinstance(state, (dict, OrderedDict)):
        num_tensors, total_elements = summarize_state_dict(state, limit=args.limit)
        print(f"\n统计：张量数={num_tensors}，总元素数={total_elements}")
        # 常见命名提示（例如 'module.' 或 ERB 的 'rbr_reparam'）
        names = list(state.keys())
        has_module_prefix = any(n.startswith('module.') for n in names)
        has_rbr_reparam = any(('rbr_reparam' in n) for n in names)
        has_erb_branches = any(('rbr_3x3_branch' in n or 'rbr_1x3_branch' in n or 'rbr_3x1_branch' in n) for n in names)
        if has_module_prefix:
            print("提示：检测到 'module.' 前缀，来自 DataParallel/DDP 保存。")
        if has_rbr_reparam:
            print("提示：检测到 'rbr_reparam'，检查点可能为 ERB 部署态权重。")
        if has_erb_branches:
            print("提示：检测到 ERB 训练态多分支权重（例如 rbr_3x3_branch 等）。")

    else:
        print("未发现 'state_dict'，已打印顶层键供参考。")


if __name__ == "__main__":
    main()