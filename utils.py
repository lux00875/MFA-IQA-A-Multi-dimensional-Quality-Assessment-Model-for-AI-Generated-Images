# utils.py - 修复版（保持原接口与功能，少量稳定性增强）
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor

import os
import json
import random
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from datetime import datetime
import math


# ====================== 基础工具 ======================
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    pl.seed_everything(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    print(f"已设置随机种子: {seed}")


class MemoryManager:
    """显存管理工具类"""
    @staticmethod
    def clear_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    @staticmethod
    def print_memory_stats():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU显存: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")


# ====================== 日志工具 ======================
def setup_metrics_logging(logs_path: str, checkpoints_path: str):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(logs_path, f"training_metrics_{timestamp}.json")
    metrics_history = {'val': []}

    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    return log_file_path, metrics_history


def save_metrics_to_file(metrics_history: dict, log_file_path: str):
    try:
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_history, f, indent=2, ensure_ascii=False)
        print(f"指标已保存到: {log_file_path}")
    except Exception as e:
        print(f"保存指标文件时出错: {e}")


# ====================== IQA指标函数 ======================
def calculate_plcc(pred: Tensor, target: Tensor):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    if len(pred) < 2:
        return 0.0
    plcc = stats.pearsonr(pred, target)[0]
    return plcc if not np.isnan(plcc) else 0.0


def calculate_srocc(pred: Tensor, target: Tensor):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    if len(pred) < 2:
        return 0.0
    srocc = stats.spearmanr(pred, target)[0]
    return srocc if not np.isnan(srocc) else 0.0


# ====================== 损失函数 ======================
def plcc_loss(y_pred: Tensor, y: Tensor) -> Tensor:
    if y_pred.dim() > 1:
        y_pred = y_pred.squeeze(-1)
    if y.dim() > 1:
        y = y.squeeze(-1)

    if y_pred.shape[0] < 2:
        return F.mse_loss(y_pred, y.float())

    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    if sigma_hat < 1e-8:
        y_pred_norm = y_pred
    else:
        y_pred_norm = (y_pred - m_hat) / (sigma_hat + 1e-8)

    sigma, m = torch.std_mean(y.float(), unbiased=False)
    if sigma < 1e-8:
        y_norm = y.float()
    else:
        y_norm = (y.float() - m) / (sigma + 1e-8)

    loss0 = F.mse_loss(y_pred_norm, y_norm) / 4
    rho = torch.mean(y_pred_norm * y_norm)
    rho = torch.clamp(rho, -0.99, 0.99)
    loss1 = F.mse_loss(rho * y_pred_norm, y_norm) / 4
    return (loss0 + loss1) / 2


def rank_loss(y_pred: Tensor, y: Tensor) -> Tensor:
    if y_pred.dim() > 1:
        y_pred = y_pred.squeeze(-1)
    if y.dim() > 1:
        y = y.squeeze()

    if y_pred.shape[0] < 2:
        return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

    pred_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
    target_diff = y.float().unsqueeze(0) - y.float().unsqueeze(1)
    target_sign = torch.sign(target_diff)
    ranking_loss = F.relu(pred_diff * target_sign)

    scale = 1 + torch.max(ranking_loss)
    if scale < 1e-8:
        return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

    total_loss = torch.sum(ranking_loss) / (y_pred.shape[0] * (y_pred.shape[0] - 1)) / scale
    return total_loss.float()


def combined_loss(y_pred: Tensor, y: Tensor, alpha: float = 1.0, beta: float = 0.3) -> Tensor:
    """PLCC + Rank loss (original behavior)."""
    p_loss = plcc_loss(y_pred, y)
    r_loss = rank_loss(y_pred, y)
    return alpha * p_loss + beta * r_loss


# ====================== Logistic拟合与性能计算 ======================
def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


def fit_function(y_label, y_output):
    # protect against too few data points for curve fitting
    y_label = np.asarray(y_label)
    y_output = np.asarray(y_output)
    if y_output.size <= 4:
        # Not enough points to fit 4 parameters; return original outputs (no fit)
        return y_output

    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    try:
        popt, _ = curve_fit(logistic_func, y_output, y_label, p0=beta, maxfev=100000000)
        y_output_logistic = logistic_func(y_output, *popt)
        return y_output_logistic
    except Exception:
        # curve_fit can fail for many reasons; in that case return original outputs
        return y_output


def performance_fit(y_label: Tensor, y_output: Tensor):
    try:
        y_label = y_label.detach().cpu().numpy()
        y_output = y_output.detach().cpu().numpy()
        # if there are too few samples, skip fitting and return 0.0 metrics
        if y_label.size < 2:
            return 0.0, 0.0

        y_output_logistic = fit_function(y_label, y_output)

        # compute PLCC and SROCC safely
        try:
            plcc = stats.pearsonr(y_output_logistic, y_label)[0]
        except Exception:
            plcc = 0.0
        try:
            srocc = stats.spearmanr(y_output, y_label)[0]
        except Exception:
            srocc = 0.0
        return plcc if not np.isnan(plcc) else 0.0, srocc if not np.isnan(srocc) else 0.0
    except Exception:
        return 0.0, 0.0


# ====================== 回归头 (保持简单有效) ======================
class Regress(nn.Module):
    """
    多层非线性回归头，用于多尺度特征输入。
    兼容 in_features 动态传入。
    """
    def __init__(self, in_features: int = 1024, dropout: float = 0.3):
        super().__init__()

        hidden1 = max(in_features // 2, 512)
        hidden2 = max(hidden1 // 2, 256)
        hidden3 = max(hidden2 // 2, 128)

        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.LayerNorm(hidden1),
            nn.Dropout(dropout),
            nn.GELU(),

            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.Dropout(dropout),
            nn.GELU(),

            nn.Linear(hidden2, hidden3),
            nn.LayerNorm(hidden3),
            nn.Dropout(dropout),
            nn.GELU(),

            nn.Linear(hidden3, 1)
        )

        print(f"Regress 初始化完成：输入维度={in_features}, 隐藏层={hidden1}-{hidden2}-{hidden3}")

    def forward(self, x: Tensor):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.layers(x).squeeze(-1)


# ====================== 调试工具 ======================
def check_model_outputs(model, dataloader, num_batches=2):
    """检查模型输出，用于调试"""
    model.eval()
    print("🔍 检查模型输出...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            image, prompt, target = batch["image"], batch["prompt"], batch["Authenticity"]
            pred = model(image, prompt)
            
            print(f"Batch {i}:")
            print(f"  Target range: [{target.min():.3f}, {target.max():.3f}]")
            print(f"  Pred range: [{pred.min():.3f}, {pred.max():.3f}]")
            print(f"  Target mean: {target.mean():.3f}")
            print(f"  Pred mean: {pred.mean():.3f}")
            
            plcc = calculate_plcc(pred, target)
            srocc = calculate_srocc(pred, target)
            print(f"  PLCC: {plcc:.4f}, SROCC: {srocc:.4f}")
            
    model.train()


def analyze_gradients(model):
    """分析梯度情况"""
    print("📊 梯度分析:")
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            if param_norm > 1e-5:  # 只显示有显著梯度的参数
                print(f"  {name}: {param_norm:.6f}")
    
    total_norm = total_norm ** 0.5
    print(f"总梯度范数: {total_norm:.6f}")
