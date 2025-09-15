import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional  # 添加Optional导入

class MultiLabelLoss(nn.Module):
    """多标签分类损失函数"""
    
    def __init__(self, loss_type: str = 'bce', weight: Optional[torch.Tensor] = None,
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.loss_type = loss_type.lower()
        
        if self.loss_type == 'bce':
            self.criterion = nn.BCELoss(weight=weight)
        elif self.loss_type == 'bce_with_logits':
            self.criterion = nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)
        elif self.loss_type == 'focal':
            self.criterion = FocalLoss(weight=weight, gamma=2.0)
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")
    
    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion(outputs, labels)

class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification"""
    
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def calculate_metrics(outputs: torch.Tensor, labels: torch.Tensor, num_labels: int) -> Dict[str, float]:
    """计算多标签分类评估指标"""
    preds = (outputs > 0.5).float()
    
    metrics = {}
    
    # 确保输出和标签形状一致
    if outputs.shape != labels.shape:
        min_dim = min(outputs.shape[1], labels.shape[1])
        outputs = outputs[:, :min_dim]
        labels = labels[:, :min_dim]
        preds = preds[:, :min_dim]
    
    # 整体准确率（所有标签都预测正确）
    exact_match = torch.all(preds == labels, dim=1).float().mean()
    metrics['exact_match_acc'] = exact_match.item()
    
    # 每个样本的平均准确率
    sample_accuracy = (preds == labels).float().mean(dim=1).mean()
    metrics['sample_accuracy'] = sample_accuracy.item()
    
    # 每个标签的准确率
    label_accuracy = (preds == labels).float().mean(dim=0)
    for i in range(min(num_labels, label_accuracy.shape[0])):
        metrics[f'label{i}_acc'] = label_accuracy[i].item()
    
    # F1分数
    tp = (preds * labels).sum(dim=0)
    fp = (preds * (1 - labels)).sum(dim=0)
    fn = ((1 - preds) * labels).sum(dim=0)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    metrics['macro_f1'] = f1.mean().item()
    metrics['micro_f1'] = (2 * tp.sum()) / (2 * tp.sum() + fp.sum() + fn.sum() + 1e-8).item()
    
    return metrics
