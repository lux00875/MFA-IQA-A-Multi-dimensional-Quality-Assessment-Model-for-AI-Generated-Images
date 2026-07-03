# train_fusion.py - 融合模型训练
import os
import argparse
import random
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append('/root/autodl-tmp/CLIP-main')

from model_fusion import FusionModule
from utils import set_random_seed
from torch.utils.data import DataLoader
from dataset import DatasetImage
import torch


def pil_collate_fn(batch):
    """Custom collate to keep PIL images as a list"""
    import torch
    images = [b['image'] for b in batch]
    prompts = [b.get('prompt', '') for b in batch]
    names = [b.get('name', '') for b in batch]

    out = {
        'image': images,
        'prompt': prompts,
        'name': names
    }

    # collect optional numeric columns into tensors
    if 'Quality' in batch[0]:
        out['Quality'] = torch.tensor([float(b.get('Quality', 0.0)) for b in batch], dtype=torch.float32)
    if 'Correspondence' in batch[0]:
        out['Correspondence'] = torch.tensor([float(b.get('Correspondence', 0.0)) for b in batch], dtype=torch.float32)
    if 'Authenticity' in batch[0]:
        out['Authenticity'] = torch.tensor([float(b.get('Authenticity', 0.0)) for b in batch], dtype=torch.float32)

    return out


def train_single_split(args, split_id):
    """训练单个数据划分"""
    print(f"\n{'='*50}")
    print(f"开始训练融合模型第 {split_id+1} 个划分")
    print(f"{'='*50}")

    current_seed = args.seed + split_id
    set_random_seed(current_seed)

    # 输出路径设置
    split_dir = os.path.join(args.output_dir, f"split_{split_id+1}")
    checkpoint_dir = os.path.join(split_dir, "checkpoints")
    log_dir = os.path.join(split_dir, "logs")
    csv_log_dir = os.path.join(split_dir, "csv_logs")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(csv_log_dir, exist_ok=True)

    # 检查点回调
    checkpoint_callback = ModelCheckpoint(
        monitor="val_corr_avg",
        dirpath=checkpoint_dir,
        filename=f"fusion_best_split_{split_id+1}",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    
    # Early stopping removed to ensure full epoch training

    # 日志记录器
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name="fusion_module"
    )
    
    csv_logger = CSVLogger(
        save_dir=csv_log_dir,
        name=f"split_{split_id+1}"
    )

    # 构建数据加载器
    train_dataset = DatasetImage('train', split_id)
    val_dataset = DatasetImage('val', split_id)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=pil_collate_fn,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=pil_collate_fn,
        persistent_workers=True if args.num_workers > 0 else False
    )

    print(f"训练样本: {len(train_dataset)}")
    print(f"验证样本: {len(val_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.num_workers}")

    # 初始化融合模型
    model = FusionModule(
        split_id=split_id,
        output_dir=args.output_dir,
        clip_weights=args.clip_weights,
        swin_weights=args.swin_weights,
        clip_frozen_ratio=args.clip_frozen_ratio,
        swin_freeze_strategy=args.swin_freeze_strategy,
        swin_freeze_ratio=args.swin_freeze_ratio,
        fusion_dim=args.fusion_dim,
        lr=args.lr,
        alpha=args.alpha,
        gamma=args.gamma,
        weight_decay=args.weight_decay,
        dropout=args.dropout
    )

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"冻结比例: {(total_params - trainable_params) / total_params * 100:.1f}%")

    # Lightning Trainer
    trainer = pl.Trainer(
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        logger=[tb_logger, csv_logger],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        gradient_clip_val=args.gradient_clip_val,
        check_val_every_n_epoch=1,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        enable_model_summary=True,
        accumulate_grad_batches=args.accumulate_grad_batches,
        deterministic=args.deterministic
    )

    # 主训练阶段
    print(f"\n开始主训练阶段 (Split {split_id+1})...")
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # 加载最佳模型进行最终验证
    best_ckpt = checkpoint_callback.best_model_path
    if best_ckpt and os.path.exists(best_ckpt):
        print(f"\n加载最佳模型进行最终验证: {best_ckpt}")
        best_model = FusionModule.load_from_checkpoint(best_ckpt)
        best_model.eval()
        
        # 在验证集上评估
        val_results = trainer.validate(best_model, val_dataloader, verbose=False)
        if val_results:
            print(f"最佳模型验证结果:")
            for key, value in val_results[0].items():
                print(f"  {key}: {value:.4f}")
    
    print(f"第 {split_id+1} 个划分训练完成 ✅")
    return best_ckpt


def main():
    parser = argparse.ArgumentParser(description="融合模型训练脚本")
    
    # 数据集参数
    parser.add_argument('--num_splits', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # 模型参数
    # 使用 CLIP ViT-L/14；可传 HuggingFace 仓库名或本地目录（需完整文件）
    parser.add_argument('--clip_weights', type=str,
                       default="openai/clip-vit-large-patch14")
    parser.add_argument('--swin_weights', type=str, default=None)
    parser.add_argument('--clip_frozen_ratio', type=float, default=0.5)
    parser.add_argument('--swin_freeze_strategy', type=str, default="partial",
                       choices=["none", "partial", "backbone"])
    parser.add_argument('--swin_freeze_ratio', type=float, default=0.5)
    parser.add_argument('--fusion_dim', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # 优化器参数
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=1.0, help='PLCC loss weight')
    parser.add_argument('--gamma', type=float, default=0.3, help='Rank loss weight')
    
    # 训练参数
    parser.add_argument('--max_epochs', type=int, default=25)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32])
    parser.add_argument('--deterministic', action='store_true', default=False)
    
    # 日志参数
    parser.add_argument('--log_every_n_steps', type=int, default=20)
    
    # 设备参数
    parser.add_argument('--devices', type=int, default=1)
    
    # 训练范围
    parser.add_argument('--output_dir', type=str,
                       default="/root/autodl-tmp/CLIP-main/fusion_module")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--only_split', type=int, default=None,
                       help='如果设置，只训练指定的划分')
    parser.add_argument('--run_all', action='store_true', default=True,
                       help='运行所有划分')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='调试模式，减少epoch数')
    # 自定义要运行的划分（逗号分隔，如 1,4,5）
    parser.add_argument('--run_splits', type=str, default=None,
                        help='仅运行指定的划分，逗号分隔的编号，如 1,4,5')

    args = parser.parse_args()
    
    # 调试模式
    if args.debug:
        args.batch_size = 4
        args.num_workers = 2
        print("调试模式启用，减少训练epoch")

    # 打印配置
    print("融合模型训练配置:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # 确定训练范围
    if args.run_splits:
        try:
            selected = [int(x.strip()) for x in args.run_splits.split(',') if x.strip()]
        except ValueError:
            raise ValueError(f"run_splits 参数格式错误: {args.run_splits}，示例: 1,4,5")
        # 转换为 0-based 并校验
        split_range = []
        for s in selected:
            if s < 1 or s > args.num_splits:
                raise ValueError(f"run_splits 包含非法划分编号: {s}，有效范围为 1..{args.num_splits}")
            split_range.append(s - 1)
        print(f"仅训练指定划分: {selected}")
    elif args.only_split is not None:
        s = args.only_split - 1
        if s < 0 or s >= args.num_splits:
            raise ValueError(f"only_split 超出范围: {args.only_split}")
        split_range = [s]
        print(f"仅训练单个划分: {args.only_split}")
    else:
        print(f'训练所有划分 (split_1..split_{args.num_splits})')
        split_range = list(range(args.num_splits))

    # 训练每个划分
    success_count = 0
    best_checkpoints = []
    
    for split_id in split_range:
        try:
            best_ckpt = train_single_split(args, split_id)
            if best_ckpt:
                best_checkpoints.append(best_ckpt)
            success_count += 1
        except Exception as e:
            print(f"第 {split_id+1} 个划分训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 打印总结
    print(f"\n{'='*60}")
    print("训练完成总结:")
    print(f"成功训练 {success_count}/{len(split_range)} 个划分")
    
    if best_checkpoints:
        print(f"\n最佳模型检查点:")
        for i, ckpt in enumerate(best_checkpoints):
            print(f"  Split {i+1}: {ckpt}")
    
    print(f"\n所有日志和检查点保存在: {args.output_dir}")
    print('='*60)


if __name__ == "__main__":
    main()