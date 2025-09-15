import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

class MultiLabelDataset(Dataset):
    """多标签分类数据集"""
    
    def __init__(self, data_root, txt_file=None, transform=None, is_train=True):
        """
        初始化多标签数据集
        
        Args:
            data_root: 数据根目录
            txt_file: 包含图像路径和标签的txt文件（如果为None则自动扫描）
            transform: 数据增强变换
            is_train: 是否为训练模式
        """
        self.data_root = data_root
        self.transform = transform
        self.is_train = is_train
        self.image_paths = []
        self.labels = []
        
        if txt_file:
            # 从txt文件加载
            self._load_from_txt_file(txt_file)
        else:
            # 自动扫描目录（备用方案）
            self._scan_data_directory()
        
        print(f"加载数据集: {data_root}")
        print(f"样本数量: {len(self.image_paths)}")
        if self.labels:
            print(f"标签维度: {len(self.labels[0])}")
    
    def _load_from_txt_file(self, txt_file):
        """从txt文件加载数据"""
        txt_path = os.path.join(self.data_root, txt_file)
        
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"找不到标签文件: {txt_path}")
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) != 2:
                continue
                
            img_path, label_str = parts
            # 构建完整的图像路径
            full_img_path = os.path.join(self.data_root, img_path)
            
            # 解析标签（格式: 0,1,0,1）
            try:
                labels = [int(x) for x in label_str.split(',')]
                labels = torch.tensor(labels, dtype=torch.float32)
                
                self.image_paths.append(full_img_path)
                self.labels.append(labels)
            except ValueError:
                print(f"跳过无效的标签行: {line}")
                continue
    
    def _scan_data_directory(self):
        """扫描数据目录，自动解析标签（备用方案）"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for root, _, files in os.walk(self.data_root):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    img_path = os.path.join(root, file)
                    labels = self._parse_labels_from_path(img_path)
                    
                    if labels is not None:
                        self.image_paths.append(img_path)
                        self.labels.append(labels)
    
    def _parse_labels_from_path(self, img_path):
        """从文件路径解析标签（备用方案）"""
        # 这里可以根据你的文件名格式实现标签解析
        # 示例实现，需要根据实际数据调整
        return torch.tensor([0, 0, 0], dtype=torch.float32)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
                
            return image, label
            
        except Exception as e:
            print(f"加载图像失败: {img_path}, 错误: {e}")
            blank_image = torch.zeros(3, 224, 224)
            return blank_image, label

def get_train_transforms():
    """训练数据增强"""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    """验证数据增强"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_dataloaders(data_root, batch_size=32, num_workers=4, 
                      val_size=0.2, random_seed=42, stratify=True):
    """
    创建训练和验证数据加载器（自动划分数据集）
    
    Args:
        data_root: 数据根目录
        batch_size: 批次大小
        num_workers: 工作线程数
        val_size: 验证集比例 (0.0-1.0)
        random_seed: 随机种子
        stratify: 是否进行分层抽样
    
    Returns:
        train_loader, val_loader, num_labels
    """
    # 首先检查是否存在train_list.txt
    train_txt_path = os.path.join(data_root, 'train_list.txt')
    
    if os.path.exists(train_txt_path):
        print("使用train_list.txt文件加载数据...")
        # 从train_list.txt加载完整数据集
        full_dataset = MultiLabelDataset(
            data_root=data_root,
            txt_file='train_list.txt',
            transform=None,
            is_train=True
        )
    else:
        print("未找到train_list.txt，自动扫描目录...")
        # 自动扫描目录
        full_dataset = MultiLabelDataset(
            data_root=data_root,
            transform=None,
            is_train=True
        )
    
    if len(full_dataset) == 0:
        raise ValueError(f"在目录 {data_root} 中没有找到有效数据")
    
    num_labels = len(full_dataset.labels[0]) if full_dataset.labels else 0
    
    # 划分训练集和验证集
    if val_size > 0:
        if stratify and num_labels > 0:
            # 分层抽样
            labels = [label.numpy() for label in full_dataset.labels]
            train_idx, val_idx = train_test_split(
                range(len(full_dataset)),
                test_size=val_size,
                random_state=random_seed,
                stratify=labels
            )
            
            # 创建子数据集
            train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
            val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
            
        else:
            # 随机划分
            dataset_size = len(full_dataset)
            val_size_int = int(dataset_size * val_size)
            train_size_int = dataset_size - val_size_int
            
            train_dataset, val_dataset = random_split(
                full_dataset,
                [train_size_int, val_size_int],
                generator=torch.Generator().manual_seed(random_seed)
            )
        
        # 为子数据集设置不同的变换
        # 注意：这里需要修改子数据集的基础数据集的变换
        full_dataset.transform = get_train_transforms()
        full_dataset.is_train = True
        
        # 创建验证集副本并设置不同的变换
        val_full_dataset = MultiLabelDataset(
            data_root=data_root,
            txt_file='train_list.txt' if os.path.exists(train_txt_path) else None,
            transform=get_val_transforms(),
            is_train=False
        )
        val_dataset = torch.utils.data.Subset(val_full_dataset, val_idx)
        
    else:
        # 不使用验证集
        full_dataset.transform = get_train_transforms()
        train_dataset = full_dataset
        val_dataset = None
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
    else:
        val_loader = None
    
    print(f"数据集统计:")
    print(f"  总样本数: {len(full_dataset)}")
    print(f"  训练集: {len(train_dataset)}")
    if val_dataset:
        print(f"  验证集: {len(val_dataset)}")
    print(f"  标签数量: {num_labels}")
    
    return train_loader, val_loader, num_labels

def test_dataloader():
    """测试数据加载器"""
    data_root = '/mnt/jx/3_NANING/00_DATASETS/04_H_Glass_Mask_baseFace_MUL_CLA/00_helt_glass_mask_BASE_face'
    
    try:
        train_loader, val_loader, num_labels = create_dataloaders(
            data_root, 
            batch_size=4, 
            val_size=0.2
        )
        
        print(f"标签数量: {num_labels}")
        
        # 测试一个批次
        for images, labels in train_loader:
            print(f"图像形状: {images.shape}")
            print(f"标签形状: {labels.shape}")
            print(f"标签示例: {labels[0]}")
            break
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_dataloader()
