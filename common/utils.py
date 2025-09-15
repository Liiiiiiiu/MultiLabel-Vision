import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union, List
import json
from dataclasses import dataclass, asdict
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet34_Weights, ResNet18_Weights, DenseNet121_Weights

def setup_device(device_str: str) -> torch.device:
    """设置训练设备"""
    if 'cuda' in device_str and torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"使用GPU训练，检测到 {torch.cuda.device_count()} 个GPU")
    else:
        device = torch.device('cpu')
        print("使用CPU训练")
    return device

def create_directories(model_dir: str):
    """创建模型保存目录"""
    os.makedirs(model_dir, exist_ok=True)
    print(f"模型将保存至: {model_dir}")
    return model_dir

def save_model(model: nn.Module, optimizer: optim.Optimizer, scheduler: Any,
              metrics_history: Dict, config: Any, epoch: Union[int, str],
              model_dir: str, is_best: bool = False):
    """保存模型检查点"""
    if isinstance(epoch, int):
        filename = f'model_epoch_{epoch:03d}.pth'
    else:
        filename = f'model_{epoch}.pth'
    
    save_path = os.path.join(model_dir, filename)
    
    # 获取模型状态（处理DataParallel）
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    state = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics_history': metrics_history,
        'config': asdict(config) if hasattr(config, '__dataclass_fields__') else config.__dict__,
        'is_best': is_best
    }
    
    torch.save(state, save_path)
    print(f'模型已保存至: {save_path}')
    
    # 保存最佳模型
    if is_best:
        best_path = os.path.join(model_dir, 'model_best.pth')
        torch.save(state, best_path)
        print(f'最佳模型已保存至: {best_path}')
    
    return save_path

def load_model(checkpoint_path: Optional[str], model: nn.Module, optimizer: Optional[optim.Optimizer] = None,
               scheduler: Optional[Any] = None, device: torch.device = None) -> Dict:
    """
    加载模型检查点，支持本地文件和自动下载预训练权重。

    Args:
        checkpoint_path: 本地检查点文件路径，若为 None 或文件不存在则尝试下载
        model: 要加载权重的模型
        optimizer: 可选的优化器，用于恢复训练
        scheduler: 可选的学习率调度器，用于恢复训练
        device: 目标设备

    Returns:
        检查点字典（包含加载的状态）
    """
    # 映射 backbone 到 torchvision 预训练权重
    weights_map = {
        'resnet50': ResNet50_Weights.DEFAULT,
        'resnet34': ResNet34_Weights.DEFAULT,
        'resnet18': ResNet18_Weights.DEFAULT,
        'densenet121': DenseNet121_Weights.DEFAULT
    }

    # 提取 backbone 名称（从模型的 name 属性或检查点路径推测）
    backbone = getattr(model, 'name', 'resnet50')  # 默认 resnet50

    try:
        if checkpoint_path and os.path.exists(checkpoint_path):
            # 加载本地检查点
            print(f"加载本地检查点: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # 处理检查点格式
            state_dict = checkpoint
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            
            # 处理多 GPU 的 module 前缀
            if isinstance(model, nn.DataParallel):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.module.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
            
            # 加载优化器和调度器状态
            if optimizer and isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and isinstance(checkpoint, dict) and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            print(f"已加载检查点: {checkpoint_path}, 训练轮次: {checkpoint.get('epoch', '未知')}")
            return checkpoint if isinstance(checkpoint, dict) else {'model_state_dict': checkpoint}
        
        else:
            # 本地文件不存在或未提供，尝试自动下载
            print(f"未找到本地检查点或未指定路径，尝试下载 {backbone} 的预训练权重...")
            weights = weights_map.get(backbone)
            if weights:
                state_dict = weights.get_state_dict(progress=True)
                # 处理多 GPU 的 module 前缀
                if isinstance(model, nn.DataParallel):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    model.module.load_state_dict(state_dict, strict=False)
                else:
                    model.load_state_dict(state_dict, strict=False)
                print(f"成功下载并加载 {backbone} 的预训练权重")
                return {'model_state_dict': state_dict}
            else:
                raise ValueError(f"不支持的 backbone: {backbone}. 可用的 backbone: {list(weights_map.keys())}")
    
    except Exception as e:
        print(f"加载模型失败: {e}")
        raise

def plot_training_curves(metrics_history: Dict, model_dir: str, num_labels: int):
    """绘制训练曲线"""
    if not metrics_history:
        return
    
    num_plots = 2 + min(5, num_labels)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots))
    
    if num_plots == 1:
        axes = [axes]
    
    # 绘制总损失
    axes[0].plot(metrics_history.get('train_loss', []), 'b-', label='训练损失')
    if 'val_loss' in metrics_history and metrics_history['val_loss']:
        axes[0].plot(metrics_history['val_loss'], 'r-', label='验证损失')
    axes[0].set_title('总损失')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制整体准确率
    axes[1].plot(metrics_history.get('train_sample_accuracy', []), 'b-', label='训练准确率')
    if 'val_sample_accuracy' in metrics_history and metrics_history['val_sample_accuracy']:
        axes[1].plot(metrics_history['val_sample_accuracy'], 'r-', label='验证准确率')
    axes[1].set_title('样本平均准确率')
    axes[1].legend()
    axes[1].grid(True)
    
    # 绘制前5个标签的准确率
    for i in range(min(5, num_labels)):
        train_acc = metrics_history.get(f'train_label{i}_acc', [])
        val_acc = metrics_history.get(f'val_label{i}_acc', [])
        
        if i + 2 < num_plots:
            axes[i+2].plot(train_acc, 'b-', label=f'标签{i}训练')
            if val_acc:
                axes[i+2].plot(val_acc, 'r-', label=f'标签{i}验证')
            axes[i+2].set_title(f'标签 {i} 准确率')
            axes[i+2].legend()
            axes[i+2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_curves.png'))
    plt.close()
    
    # 保存指标到文件
    metrics_file = os.path.join(model_dir, 'metrics_history.npy')
    np.save(metrics_file, metrics_history)
    print(f"指标历史已保存至: {metrics_file}")

def setup_model_for_multilabel(base_model: nn.Module, num_labels: int) -> nn.Module:
    """设置多标签分类模型"""
    # 替换最后的分类层以适应我们的标签数量
    if hasattr(base_model, 'classifier'):
        if isinstance(base_model.classifier, nn.Sequential):
            # 找到最后的线性层并替换
            for i, layer in enumerate(base_model.classifier):
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    base_model.classifier[i] = nn.Linear(in_features, num_labels)
                    break
        else:
            if isinstance(base_model.classifier, nn.Linear):
                in_features = base_model.classifier.in_features
                base_model.classifier = nn.Linear(in_features, num_labels)
    
    elif hasattr(base_model, 'fc'):
        if isinstance(base_model.fc, nn.Sequential):
            for i, layer in enumerate(base_model.fc):
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    base_model.fc[i] = nn.Linear(in_features, num_labels)
                    break
        else:
            if isinstance(base_model.fc, nn.Linear):
                in_features = base_model.fc.in_features
                base_model.fc = nn.Linear(in_features, num_labels)
    
    # 添加sigmoid激活函数
    model = nn.Sequential(
        base_model,
        nn.Sigmoid()
    )
    
    return model

def print_model_info(model: nn.Module, device: torch.device, num_labels: int):
    """打印模型信息"""
    print(f"模型已加载到设备: {next(model.parameters()).device}")
    print(f"模型输出维度: {num_labels} (多标签分类)")
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}")

def save_config(config: Any, model_dir: str):
    """保存训练配置"""
    config_path = os.path.join(model_dir, 'config.json')
    config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else config.__dict__
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"配置已保存至: {config_path}")

def load_config(config_path: str, config_class: Any):
    """加载训练配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    return config_class(**config_dict)
