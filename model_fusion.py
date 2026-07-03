# model_fusion.py - 整合CLIP和Swin特征并进行融合
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('/root/autodl-tmp/CLIP-main')
sys.path.append('/root/autodl-tmp/CLIP-main/alignment_module/model')
sys.path.append('/root/autodl-tmp/CLIP-main/traditional_module/model')

from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import timm

from utils import save_metrics_to_file, setup_metrics_logging, Regress
from utils import calculate_plcc, calculate_srocc, combined_loss, performance_fit


# ===========================================================
# ✅ CLIP 包装器 (保持原样)
# ===========================================================
class CLIPWrapper(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14", image_size=224, frozen_ratio: float = 0.5):
        super().__init__()
        self.frozen_ratio = float(frozen_ratio)
        
        # 加载模型：优先本地目录，其次远程 repo id
        try:
            local_dir = os.path.isdir(model_name)
        except Exception:
            local_dir = False

        try:
            if local_dir:
                needed = ["config.json"]
                has_core = all(os.path.exists(os.path.join(model_name, f)) for f in needed)
                if not has_core:
                    raise RuntimeError(f"本地目录缺少必要文件: {needed}，目录: {model_name}")
                # 优先尝试 safetensors；若损坏则回退到二进制权重
                try:
                    self.model = CLIPModel.from_pretrained(model_name, local_files_only=True)
                except Exception:
                    self.model = CLIPModel.from_pretrained(model_name, local_files_only=True, use_safetensors=False)
                self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
            else:
                try:
                    self.model = CLIPModel.from_pretrained(model_name)
                except Exception:
                    # 远程下载 safetensors 文件损坏或不可用时，回退到 .bin
                    self.model = CLIPModel.from_pretrained(model_name, use_safetensors=False)
                self.processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP from '{model_name}': {e}")

        # enable gradient checkpointing if available
        try:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print('启用 CLIP gradient checkpointing')
        except Exception:
            pass

        self.hidden_size = self.model.config.projection_dim
        
        # 冻结策略
        self._setup_freeze_layers()

        # report frozen vs trainable counts
        try:
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"CLIP loaded: total_params={total}, trainable_params={trainable}")
        except Exception:
            pass

        # 融合权重
        self.fuse_w = nn.Parameter(torch.tensor([0.55, 0.35, 0.10], dtype=torch.float32))
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=8, batch_first=True)

    def _setup_freeze_layers(self):
        """Simpler, ratio-based freezing strategy"""
        if float(self.frozen_ratio) >= 1.0:
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            return

        try:
            layers = getattr(self.model.vision_model.encoder, 'layers', None)
            total_layers = len(layers) if layers is not None else 0
        except Exception:
            total_layers = 0

        if total_layers <= 0:
            freeze_until = 6
        else:
            freeze_until = int(total_layers * float(self.frozen_ratio))

        # apply freezing for vision encoder layers
        for name, param in self.model.named_parameters():
            if "vision_model.encoder.layers" in name:
                try:
                    layer_num = int(name.split("layers.")[1].split(".")[0])
                    param.requires_grad = (layer_num >= freeze_until)
                except Exception:
                    param.requires_grad = True
            else:
                param.requires_grad = True

    def forward(self, image, prompt):
        device = next(self.model.parameters()).device

        # 处理输入数据
        if isinstance(image, torch.Tensor):
            pixel_values = image.to(device)
            if isinstance(prompt, str):
                texts = [prompt] * pixel_values.shape[0]
            else:
                texts = prompt
            text_inputs = self.processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            for k, v in text_inputs.items():
                if isinstance(v, torch.Tensor):
                    text_inputs[k] = v.to(device)
            outputs = self.model(pixel_values=pixel_values, **text_inputs)
        else:
            if isinstance(prompt, str):
                try:
                    batch_size = len(image)
                    texts = [prompt] * batch_size
                except Exception:
                    texts = [prompt]
            else:
                texts = prompt
            inputs = self.processor(text=texts, images=image, return_tensors="pt", padding=True)
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            outputs = self.model(**inputs)

        img_feat = outputs.image_embeds  # [B, hidden_size]
        txt_feat = outputs.text_embeds   # [B, hidden_size]

        # 特征融合
        diff = torch.abs(img_feat - txt_feat)
        w = F.softmax(self.fuse_w, dim=0)
        base = w[0] * img_feat + w[1] * txt_feat + w[2] * diff
        
        # cross-attention
        try:
            attn_out, _ = self.cross_attn(base.unsqueeze(1), img_feat.unsqueeze(1), txt_feat.unsqueeze(1))
            attn_out = attn_out.squeeze(1)
            fused = base + attn_out
        except Exception:
            fused = base

        fused = F.layer_norm(fused, [fused.size(-1)])
        return fused


# ===========================================================
# ✅ Swin Transformer 包装器 (简化版)
# ===========================================================
class SwinWrapper(nn.Module):
    def __init__(self, swin_weights=None, freeze_strategy="partial", freeze_ratio=0.5):
        super().__init__()
        
        try:
            self.swin = timm.create_model(
                'swin_base_patch4_window12_384_in22k',
                pretrained=True,
                features_only=True,
                out_indices=(0, 1, 2, 3)
            )
            print("使用timm features_only Swin base（4尺度）")
        except Exception as e:
            print(f"无法直接创建 features_only Swin：{e}")
            self.swin = timm.create_model('swin_base_patch4_window12_384_in22k', pretrained=True)

        # 基于 timm 的预处理（用于 PIL/list 输入）
        try:
            from timm.data import create_transform
            self.transform = create_transform(
                input_size=(3, 384, 384),
                is_training=False,
                interpolation='bicubic',
                mean=getattr(self.swin, 'default_cfg', {}).get('mean', (0.5, 0.5, 0.5)),
                std=getattr(self.swin, 'default_cfg', {}).get('std', (0.5, 0.5, 0.5))
            )
            print("已创建Swin图像预处理 (384, bicubic)")
        except Exception as e:
            self.transform = None
            print(f"创建Swin预处理失败，forward中将尝试备用转换：{e}")

        if swin_weights:
            try:
                self.load_weights(swin_weights)
            except Exception as e:
                print(f"加载自定义swin权重失败：{e}，将使用timm预训练权重。")

        try:
            channels = self.swin.feature_info.channels()
            self.feature_channels = channels
            self.multi_in_features = sum(channels)
        except Exception:
            self.feature_channels = None
            self.multi_in_features = 1024
            print("无法获取 feature_info.channels()，回退 multi_in_features=1024")

        print(f"Swin多尺度特征通道和: {self.multi_in_features}")
        
        # 冻结策略
        self._setup_freeze(freeze_strategy, freeze_ratio)

    def load_weights(self, weight_path):
        try:
            checkpoint = torch.load(weight_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            missing, unexpected = self.swin.load_state_dict(state_dict, strict=False)
            print(f"加载自定义权重完成，missing:{len(missing)}, unexpected:{len(unexpected)}")
        except Exception as e:
            print(f"load_weights 出错: {e}")

    def _setup_freeze(self, freeze_strategy="partial", freeze_ratio=0.5):
        if freeze_strategy == "none":
            for param in self.swin.parameters():
                param.requires_grad = True
            print("所有Swin参数可训练")
        elif freeze_strategy == "partial":
            try:
                num_stages = len(self.swin.feature_info)
            except Exception:
                num_stages = 4
            freeze_stages = int(num_stages * freeze_ratio)
            for name, param in self.swin.named_parameters():
                should_freeze = False
                for i in range(freeze_stages):
                    if f'blocks' in name or f'layers.{i}' in name or f'stages.{i}' in name:
                        should_freeze = True
                        break
                if should_freeze:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            print(f"冻结策略: 部分冻结，冻结前 {freeze_stages} 个 stage")
        elif freeze_strategy == 'backbone':
            for param in self.swin.parameters():
                param.requires_grad = False
            print("整个SwinTransformer被冻结")
        else:
            for param in self.swin.parameters():
                param.requires_grad = True

    def forward(self, image):
        # 将输入统一为张量 [B,3,384,384] 并确保与 Swin 模型设备一致
        try:
            import torch
            import torchvision.transforms as T
            swin_device = next(self.swin.parameters()).device
            if isinstance(image, (list, tuple)):
                if self.transform is None:
                    basic = T.Compose([
                        T.Resize((384, 384), interpolation=T.InterpolationMode.BICUBIC),
                        T.ToTensor(),
                        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                    ])
                    imgs = [basic(img) for img in image]
                else:
                    imgs = [self.transform(img) for img in image]
                x = torch.stack(imgs, dim=0).to(swin_device, non_blocking=True)
            elif isinstance(image, torch.Tensor):
                x = image
                # 若尺寸/归一化不匹配，尽量调整到Swin期望大小
                if x.dim() == 4 and (x.shape[-1] != 384 or x.shape[-2] != 384):
                    x = torch.nn.functional.interpolate(x, size=(384, 384), mode='bicubic', align_corners=False)
                x = x.to(swin_device, non_blocking=True)
            else:
                # 单张 PIL 图像
                if self.transform is None:
                    basic = T.Compose([
                        T.Resize((384, 384), interpolation=T.InterpolationMode.BICUBIC),
                        T.ToTensor(),
                        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                    ])
                    x = basic(image).unsqueeze(0).to(swin_device, non_blocking=True)
                else:
                    x = self.transform(image).unsqueeze(0).to(swin_device, non_blocking=True)

            features = self.swin(x)
        except Exception as e:
            # 备用方案：直接前向并做均值池化
            x_fallback = x if 'x' in locals() else image
            try:
                if isinstance(x_fallback, torch.Tensor):
                    x_fallback = x_fallback.to(swin_device, non_blocking=True)
            except Exception:
                pass
            feature = self.swin(x_fallback)
            if feature.dim() == 4:
                feature = feature.mean(dim=(2,3))
            elif feature.dim() == 3:
                feature = feature.mean(dim=1)
            return feature

        if isinstance(features, (list, tuple)):
            pooled = []
            for f in features:
                if f.dim() == 4:
                    pooled_feat = f.mean(dim=(2, 3))
                elif f.dim() == 3:
                    pooled_feat = f.mean(dim=1)
                else:
                    pooled_feat = f.reshape(f.size(0), -1)
                pooled.append(pooled_feat)
            feature = torch.cat(pooled, dim=1)
        else:
            feature = features
            if feature.dim() == 4:
                feature = feature.mean(dim=(2,3))
            elif feature.dim() == 3:
                feature = feature.mean(dim=1)
            elif feature.dim() != 2:
                feature = feature.reshape(feature.size(0), -1)

        return feature


# ===========================================================
# ✅ 特征融合模块
# ===========================================================
class FeatureFusionModule(nn.Module):
    def __init__(self, clip_dim, swin_dim, fusion_dim=1024, dropout=0.3):
        super().__init__()
        
        # 分别对两个特征进行投影
        self.clip_proj = nn.Sequential(
            nn.Linear(clip_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.swin_proj = nn.Sequential(
            nn.Linear(swin_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim // 2, 
            num_heads=8, 
            batch_first=True
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # 自适应权重
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, clip_feat, swin_feat):
        # 投影到相同维度
        clip_proj = self.clip_proj(clip_feat)  # [B, fusion_dim//2]
        swin_proj = self.swin_proj(swin_feat)  # [B, fusion_dim//2]
        
        # 跨模态注意力
        clip_attn, _ = self.cross_attention(
            clip_proj.unsqueeze(1), 
            swin_proj.unsqueeze(1), 
            swin_proj.unsqueeze(1)
        )
        clip_attn = clip_attn.squeeze(1)
        
        swin_attn, _ = self.cross_attention(
            swin_proj.unsqueeze(1), 
            clip_proj.unsqueeze(1), 
            clip_proj.unsqueeze(1)
        )
        swin_attn = swin_attn.squeeze(1)
        
        # 自适应加权融合
        clip_fused = self.alpha * clip_proj + (1 - self.alpha) * clip_attn
        swin_fused = self.beta * swin_proj + (1 - self.beta) * swin_attn
        
        # 拼接并融合
        combined = torch.cat([clip_fused, swin_fused], dim=1)  # [B, fusion_dim]
        fused = self.fusion_layer(combined)
        
        return fused


# ===========================================================
# ✅ 融合模型主体
# ===========================================================
class FusionModule(pl.LightningModule):
    def __init__(self,
                 split_id=0,
                 output_dir="/root/autodl-tmp/CLIP-main/fusion_module",
                 clip_weights="/root/autodl-tmp/CLIP-main/clip-vit-large-patch14",
                 swin_weights=None,
                 clip_frozen_ratio=0.5,
                 swin_freeze_strategy="partial",
                 swin_freeze_ratio=0.5,
                 fusion_dim=1024,
                 lr=1e-5,
                 alpha=1.0,
                 gamma=0.3,
                 weight_decay=0.02,
                 dropout=0.3):
        super().__init__()

        self.lr = lr
        self.plcc_weight = alpha
        self.rank_weight = gamma
        self.weight_decay = weight_decay
        # 保存用于 forward 中的冻结策略判断
        self.swin_freeze_strategy = swin_freeze_strategy
        
        # 保存超参数
        self.save_hyperparameters()

        # 初始化两个主干网络
        self.clip_model = CLIPWrapper(
            model_name=clip_weights, 
            image_size=224, 
            frozen_ratio=clip_frozen_ratio
        )
        
        self.swin_model = SwinWrapper(
            swin_weights=swin_weights,
            freeze_strategy=swin_freeze_strategy,
            freeze_ratio=swin_freeze_ratio
        )
        
        # 通过一次前向探测实际的 Swin 输出维度，避免不一致
        try:
            probe_device = self.device
        except Exception:
            probe_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            probe = torch.randn(1, 3, 384, 384, device=probe_device)
            swin_out = self.swin_model(probe)
            if hasattr(swin_out, 'shape') and len(swin_out.shape) == 2:
                actual_swin_dim = int(swin_out.shape[1])
            else:
                actual_swin_dim = self.swin_model.multi_in_features
        
        # 特征融合模块
        self.fusion = FeatureFusionModule(
            clip_dim=self.clip_model.hidden_size,
            swin_dim=actual_swin_dim,
            fusion_dim=fusion_dim,
            dropout=dropout
        )
        
        # 回归头
        self.regress = Regress(in_features=fusion_dim, dropout=dropout)
        
        # 日志设置
        logs_path = os.path.join(output_dir, f"split_{split_id+1}")
        checkpoints_path = os.path.join(output_dir, f"split_{split_id+1}")
        self.log_file_path, self.metrics_history = setup_metrics_logging(logs_path, checkpoints_path)

        # 验证缓存
        self.validation_predictions = []
        self.validation_targets = []

    def get_input(self, batch):
        image = batch["image"]
        prompt = batch["prompt"]
        # Use Authenticity as training target instead of Authenticity
        score = batch["Authenticity"]
        device = self.device
        if isinstance(image, torch.Tensor):
            image = image.to(device, non_blocking=True)
        if isinstance(score, torch.Tensor):
            score = score.to(device, non_blocking=True)
        return image, prompt, score

    def forward(self, image, prompt):
        # 分别提取特征
        with torch.no_grad() if self.clip_model.frozen_ratio >= 1.0 else torch.enable_grad():
            clip_feat = self.clip_model(image, prompt)
        
        with torch.no_grad() if self.swin_freeze_strategy == 'backbone' else torch.enable_grad():
            swin_feat = self.swin_model(image)
        
        # 特征融合
        fused_feat = self.fusion(clip_feat, swin_feat)
        
        # 回归预测
        predicted_scores = self.regress(fused_feat)
        
        return predicted_scores

    def training_step(self, batch, batch_idx):
        image, prompt, target = self.get_input(batch)
        pred = self.forward(image, prompt)

        # 计算指标
        plcc = calculate_plcc(pred, target)
        srocc = calculate_srocc(pred, target)
        loss = combined_loss(pred, target, self.plcc_weight, self.rank_weight)

        # 记录指标
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_plcc', plcc, prog_bar=True, sync_dist=True)
        self.log('train_srocc', srocc, prog_bar=True, sync_dist=True)
        self.log('train_corr_avg', (plcc + srocc) / 2.0, prog_bar=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        image, prompt, target = self.get_input(batch)
        pred = self.forward(image, prompt)

        plcc, srocc = performance_fit(target, pred)
        loss = combined_loss(pred, target, self.plcc_weight, self.rank_weight)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_plcc', plcc, prog_bar=True, sync_dist=True)
        self.log('val_srocc', srocc, prog_bar=True, sync_dist=True)
        self.log('val_corr_avg', (plcc + srocc) / 2.0, prog_bar=True, sync_dist=True)

        self.validation_predictions.extend(pred.detach().cpu().numpy())
        self.validation_targets.extend(target.detach().cpu().numpy())
        return loss

    def on_validation_epoch_start(self):
        self.validation_predictions = []
        self.validation_targets = []

    def on_validation_epoch_end(self):
        if not self.validation_predictions:
            return
            
        preds = np.array(self.validation_predictions)
        targets = np.array(self.validation_targets)
        
        plcc, srocc = performance_fit(torch.tensor(targets), torch.tensor(preds))
        avg_corr = (plcc + srocc) / 2.0
        
        results = {
            'epoch': self.current_epoch + 1, 
            'plcc': float(plcc), 
            'srocc': float(srocc), 
            'avg': float(avg_corr)
        }
        
        self.metrics_history['val'].append(results)
        save_metrics_to_file(self.metrics_history, self.log_file_path)
        
        print(f"\n✅ Val Epoch {self.current_epoch+1} | "
              f"PLCC: {plcc:.4f} | SROCC: {srocc:.4f} | AVG: {avg_corr:.4f}")

    def configure_optimizers(self):
        # 只优化需要梯度的参数
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )

        # 学习率调度
        try:
            if hasattr(self.trainer, 'estimated_stepping_batches'):
                total_steps = self.trainer.estimated_stepping_batches
            else:
                total_steps = self.trainer.max_epochs * 1000
        except:
            total_steps = 30000

        total_steps = max(1, total_steps)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def check_gradients(self):
        """检查梯度情况"""
        total_norm = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                if param_norm > 1e-5:
                    print(f"  {name}: {param_norm:.6f}")
        
        total_norm = total_norm ** 0.5
        print(f"总梯度范数: {total_norm:.6f}")
        return total_norm