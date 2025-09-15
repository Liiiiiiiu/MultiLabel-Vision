import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass  # 已添加

# 导入自定义模块
import sys
sys.path.append('./common')
from utils import (
    setup_device, create_directories, save_model, load_model,
    plot_training_curves, setup_model_for_multilabel, print_model_info,
    save_config
)
from loss import MultiLabelLoss, calculate_metrics

sys.path.append('./LD_dataloader')
from dataload import create_dataloaders

# Local imports
from net import get_model

# Set matplotlib backend
import matplotlib
matplotlib.use('agg')

@dataclass
class MultiLabelTrainingConfig:
    """多标签分类训练配置"""
    data_root: str
    backbone: str
    batch_size: int
    num_epoch: int
    num_workers: int
    device: str
    val_size: float
    random_seed: int
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    step_size: int = 40
    gamma: float = 0.1
    stratify: bool = True
    pretrain: Optional[str] = None
    auto_pretrain: bool = False
    resume: Optional[str] = None
    loss_type: str = 'bce'  # 保持 bce，兼容 Sigmoid

class MultiLabelTrainer:
    """多标签分类训练器"""
    
    def __init__(self, config: MultiLabelTrainingConfig):
        self.config = config
        self.device = setup_device(config.device)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.dataloaders = {}
        self.dataset_sizes = {}
        self.metrics_history = {}
        self.num_labels = 3
        self.start_epoch = 1
        
        self.model_dir = create_directories(
            os.path.join('./checkpoints', 'multilabel', f'{config.backbone}_multilabel')
        )
        
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()  # 修改：添加差异化 LR
        self._setup_criterion()
        
        # 加载预训练或恢复训练
        self._load_checkpoint()
        
        # 保存配置
        save_config(self.config, self.model_dir)
    
    def _setup_data(self):
        """设置数据加载器"""
        train_loader, val_loader, num_labels = create_dataloaders(
            data_root=self.config.data_root,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            val_size=self.config.val_size,
            random_seed=self.config.random_seed,
            stratify=self.config.stratify
        )
        
        self.num_labels = num_labels
        self.dataloaders = {'train': train_loader}
        
        if val_loader:
            self.dataloaders['val'] = val_loader
            self.dataset_sizes = {
                'train': len(train_loader.dataset),
                'val': len(val_loader.dataset)
            }
        else:
            self.dataset_sizes = {'train': len(train_loader.dataset)}
        
        print(f"训练集大小: {self.dataset_sizes.get('train', 0)}")
        if 'val' in self.dataset_sizes:
            print(f"验证集大小: {self.dataset_sizes['val']}")
        print(f"标签数量: {self.num_labels}")
    
    def _setup_model(self):
        """设置多标签分类模型（修改：传递正确 num_label，跳过 setup_model_for_multilabel）"""
        # 获取基础模型（传递正确 num_label=3，确保创建 3 个 class_*）
        base_model = get_model(self.config.backbone, num_label=self.num_labels, use_id=False)
        
        # 跳过 setup_model_for_multilabel：Backbone_nFC 已内置多标签头（class_* + Sigmoid）
        # 输出直接是 [batch, num_labels] 概率
        self.model = base_model
        
        # 多GPU支持
        if torch.cuda.device_count() > 1 and 'cuda' in str(self.device):
            print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
            self.model = nn.DataParallel(self.model)
        
        self.model = self.model.to(self.device)
        
        # 调试打印（临时，确认结构）
        print(f"模型类型: {type(self.model)}")
        if isinstance(self.model, nn.DataParallel):
            inner = self.model.module
        else:
            inner = self.model
        print(f"内层类型: {type(inner)}, class_num: {getattr(inner, 'class_num', 'N/A')}")
        print(f"有 class_0: {hasattr(inner, 'class_0')}")
        
        print_model_info(self.model, self.device, self.num_labels)
    
    def _setup_optimizer(self):
        """设置优化器（修正：适配 Backbone_nFC 的 Sequential 结构）"""
        # 处理 DataParallel
        if isinstance(self.model, nn.DataParallel):
            model_for_params = self.model.module
        else:
            model_for_params = self.model
        
        # 提取 features 参数（backbone）
        if hasattr(model_for_params, 'features'):
            features_params = list(model_for_params.features.parameters())  # 转换为列表
        else:
            # Fallback：所有参数作为 backbone
            features_params = [p for p in model_for_params.parameters() if p.requires_grad]
            print("警告: 无 features 属性，使用统一 LR")
            self.optimizer = optim.SGD(
                features_params, lr=self.config.learning_rate,
                momentum=self.config.momentum, weight_decay=self.config.weight_decay, nesterov=True
            )
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.step_size, gamma=self.config.gamma
            )
            print(f"优化器: SGD (统一), LR: {self.config.learning_rate}")
            return
        
        # 提取 classifier 参数（所有 class_* 的整个 Sequential 层）
        classifier_params = []
        has_class = False
        
        for c in range(self.num_labels):
            class_attr = f'class_{c}'
            if hasattr(model_for_params, class_attr):
                class_block = getattr(model_for_params, class_attr)
                # 直接获取整个 class_x Sequential 层的参数
                classifier_params.extend(list(class_block.parameters()))  # 转换为列表
                has_class = True
                print(f"找到分类头 {class_attr}，参数数量: {sum(p.numel() for p in class_block.parameters())}")
            else:
                print(f"警告: 无 {class_attr} 属性")
        
        if not has_class:
            print("警告: 无 class_* 属性，使用统一 LR")
            self.optimizer = optim.SGD(
                features_params, lr=self.config.learning_rate,
                momentum=self.config.momentum, weight_decay=self.config.weight_decay, nesterov=True
            )
        else:
            self.optimizer = optim.SGD([
                {'params': features_params, 'lr': self.config.learning_rate},  # 0.01
                {'params': classifier_params, 'lr': self.config.learning_rate * 10}  # 0.1
            ], momentum=self.config.momentum, weight_decay=self.config.weight_decay, nesterov=True)
            
            # 计算参数数量而不是张量数量
            backbone_param_count = sum(p.numel() for p in features_params)
            classifier_param_count = sum(p.numel() for p in classifier_params)
            
            print(f"优化器: SGD, Backbone LR: {self.config.learning_rate}, Classifier LR: {self.config.learning_rate * 10}")
            print(f"Backbone 参数: {backbone_param_count:,} 个")
            print(f"Classifier 参数: {classifier_param_count:,} 个")
            print(f"总参数: {backbone_param_count + classifier_param_count:,} 个")
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.config.step_size, gamma=self.config.gamma
        )
        print(f"学习率调度器: StepLR, 步长: {self.config.step_size}, 衰减: {self.config.gamma}")
    
    def _setup_criterion(self):
        """设置损失函数（保持 bce）"""
        self.criterion = MultiLabelLoss(loss_type=self.config.loss_type)
        print(f"损失函数: {self.config.loss_type.upper()} (多标签分类)")
    
    def _load_checkpoint(self):
        """加载检查点（保持不变）"""
        if self.config.pretrain:
            print(f"加载预训练模型: {self.config.pretrain}")
            load_model(self.config.pretrain, self.model, device=self.device)
        
        elif self.config.auto_pretrain:
            print("自动下载预训练模型...")
            load_model(None, self.model, device=self.device)
        
        elif self.config.resume:
            print(f"恢复训练: {self.config.resume}")
            checkpoint = load_model(
                self.config.resume, self.model, self.optimizer, self.scheduler, self.device
            )
            self.metrics_history = checkpoint.get('metrics_history', {})
            self.start_epoch = checkpoint.get('epoch', 1) + 1
            print(f"从第 {self.start_epoch} 轮开始恢复训练")
    
    def train_epoch(self, phase: str) -> Dict[str, float]:
        """训练一个epoch（修改：添加 labels.float() 和打印调整）"""
        if phase not in self.dataloaders:
            return {}
            
        is_train = phase == 'train'
        self.model.train(is_train)
        
        running_loss = 0.0
        running_metrics = {
            'exact_match_acc': 0.0,
            'sample_accuracy': 0.0,
            'macro_f1': 0.0,
            'micro_f1': 0.0
        }
        for i in range(self.num_labels):
            running_metrics[f'label{i}_acc'] = 0.0
        
        start_time = time.time()
        dataloader = self.dataloaders[phase]
        dataset_size = self.dataset_sizes[phase]
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            labels = labels.float()  # 新增：旧版确保 float32
            
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            
            if is_train:
                self.optimizer.zero_grad()
            
            with torch.set_grad_enabled(is_train):
                outputs = self.model(images)
                
                # 确保输出和标签形状匹配
                if outputs.shape[1] != labels.shape[1]:
                    min_dim = min(outputs.shape[1], labels.shape[1])
                    outputs = outputs[:, :min_dim]
                    labels = labels[:, :min_dim]
                
                loss = self.criterion(outputs, labels)
                
                if is_train:
                    loss.backward()
                    self.optimizer.step()
            
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            
            batch_metrics = calculate_metrics(outputs, labels, self.num_labels)  # 假设已调整阈值 0.5
            for metric_name, metric_value in batch_metrics.items():
                if metric_name in running_metrics:
                    running_metrics[metric_name] += metric_value * batch_size
            
            if batch_idx % 100 == 0:
                elapsed_time = time.time() - start_time
                samples_processed = (batch_idx + 1) * batch_size
                print(f'{phase}: [{samples_processed:5d}/{dataset_size:5d}] '
                      f'({100. * (batch_idx + 1) / len(dataloader):3.0f}%) | '
                      f'标签损失: {loss.item():.4f} | 准确率: {batch_metrics.get("sample_accuracy", 0):.4f} | '  # 调整打印：标签损失
                      f'耗时: {elapsed_time:.2f}s')
                start_time = time.time()
        
        epoch_loss = running_loss / dataset_size
        epoch_metrics = {metric: value / dataset_size for metric, value in running_metrics.items()}
        
        return {'loss': epoch_loss, **epoch_metrics}
    
    def train(self):
        """主训练循环（修改：添加 cache 清空，scheduler 位置）"""
        print("=" * 60)
        print("开始多标签分类训练...")
        print("=" * 60)
        
        start_time = time.time()
        best_val_f1 = 0.0
        best_epoch = 0
        
        # 初始化指标记录
        if not self.metrics_history:
            self._init_metrics_history()
        
        for epoch in range(self.start_epoch, self.config.num_epoch + 1):
            torch.cuda.empty_cache()  # 新增：旧版清 cache
            
            print(f'\nEpoch {epoch}/{self.config.num_epoch}')
            print('-' * 50)
            epoch_start_time = time.time()
            
            # 训练阶段
            train_metrics = self.train_epoch('train')
            for metric, value in train_metrics.items():
                self.metrics_history[f'train_{metric}'].append(value)
            
            # 更新学习率（修改：移到 train 后，匹配旧版）
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']  # 注意：groups[0] 是 backbone LR
                print(f"当前学习率 (Backbone): {current_lr:.6f}")
            
            # 验证阶段
            val_metrics = {}
            if 'val' in self.dataloaders:
                val_metrics = self.train_epoch('val')
                for metric, value in val_metrics.items():
                    self.metrics_history[f'val_{metric}'].append(value)
            
            epoch_time = time.time() - epoch_start_time
            
            # 保存最佳模型
            if val_metrics and 'macro_f1' in val_metrics:
                current_val_f1 = val_metrics['macro_f1']
                is_best = current_val_f1 > best_val_f1
                if is_best:
                    best_val_f1 = current_val_f1
                    best_epoch = epoch
                    print(f"✓ 新的最佳模型，验证F1: {current_val_f1:.4f}")
            else:
                is_best = False
            
            # 保存检查点
            if epoch % 10 == 0 or is_best:
                save_model(
                    self.model, self.optimizer, self.scheduler,
                    self.metrics_history, self.config, epoch,
                    self.model_dir, is_best=is_best
                )
            
            # 绘制曲线
            plot_training_curves(self.metrics_history, self.model_dir, self.num_labels)
            
            # 打印进度
            print(f"\nEpoch {epoch} 总结:")
            print(f"  训练损失: {train_metrics['loss']:.4f}, 准确率: {train_metrics['sample_accuracy']:.4f}")
            if val_metrics:
                print(f"  验证损失: {val_metrics['loss']:.4f}, 准确率: {val_metrics['sample_accuracy']:.4f}")
                print(f"  验证F1分数: {val_metrics['macro_f1']:.4f}")
            print(f"  耗时: {epoch_time:.2f}s")
        
        total_time = time.time() - start_time
        print('=' * 60)
        print(f'训练完成! 总耗时: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s')
        if best_val_f1 > 0:
            print(f'最佳验证F1: {best_val_f1:.4f} (Epoch {best_epoch})')
        print('=' * 60)
    
    def _init_metrics_history(self):
        """初始化指标记录（保持不变）"""
        self.metrics_history = {
            'train_loss': [], 'train_sample_accuracy': [],
            'train_exact_match_acc': [], 'train_macro_f1': [],
            'train_micro_f1': [],
        }
        
        if 'val' in self.dataloaders:
            self.metrics_history.update({
                'val_loss': [], 'val_sample_accuracy': [],
                'val_exact_match_acc': [], 'val_macro_f1': [],
                'val_micro_f1': [],
            })
        
        for i in range(self.num_labels):
            self.metrics_history[f'train_label{i}_acc'] = []
            if 'val' in self.dataloaders:
                self.metrics_history[f'val_label{i}_acc'] = []

def parse_args():
    """解析命令行参数（保持不变，建议测试用 batch=32）"""
    parser = argparse.ArgumentParser(description='多标签分类训练脚本')
    
    # 数据参数
    parser.add_argument('--data-root', required=True, help='数据根目录')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')  # 默认 32，匹配旧版测试
    parser.add_argument('--num-workers', type=int, default=4, help='工作线程数')
    
    # 模型参数
    parser.add_argument('--backbone', default='resnet50', 
                       choices=['resnet50', 'resnet34', 'resnet18', 'densenet121'], 
                       help='骨干网络')
    
    # 训练参数
    parser.add_argument('--num-epoch', type=int, default=100, help='训练轮数')
    parser.add_argument('--device', default='cuda', help='训练设备')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='学习率')
    parser.add_argument('--val-size', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--random-seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no-stratify', action='store_true', help='不使用分层抽样')
    parser.add_argument('--loss-type', default='bce', choices=['bce', 'bce_with_logits', 'focal'],
                       help='损失函数类型')
    
    # 预训练和恢复训练
    parser.add_argument('--pretrain', help='预训练模型路径')
    parser.add_argument('--auto-pretrain', action='store_true', help='自动下载预训练模型')
    parser.add_argument('--resume', help='恢复训练检查点路径')
    
    return parser.parse_args()

def main():
    """主函数（保持不变）"""
    args = parse_args()
    
    # 创建配置
    config = MultiLabelTrainingConfig(
        data_root=args.data_root,
        backbone=args.backbone,
        batch_size=args.batch_size,
        num_epoch=args.num_epoch,
        num_workers=args.num_workers,
        device=args.device,
        val_size=args.val_size,
        random_seed=args.random_seed,
        stratify=not args.no_stratify,
        learning_rate=args.learning_rate,
        pretrain=args.pretrain,
        auto_pretrain=args.auto_pretrain,
        resume=args.resume,
        loss_type=args.loss_type
    )
    
    # 创建训练器并开始训练
    trainer = MultiLabelTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
