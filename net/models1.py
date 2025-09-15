import torch
from torch import nn
from torch.nn import init
from torchvision import models
from torch.nn import functional as F

class Backbone_nFC(nn.Module):
    def __init__(self, class_num, model_name='resnet50_nfc'):
        super(Backbone_nFC, self).__init__()
        self.class_num = class_num
        model_ft = getattr(models,'resnet50')(pretrained=False)
        print("dff")
        
        model_ft = nn.Sequential(*list(model_ft.children())[:-1])
        self.features = model_ft
        
        # 动态创建分类头
        for c in range(self.class_num):
            self.__setattr__(f'class_{c}', nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, 1),
                nn.Sigmoid()
            ))
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        # 动态获取所有分类头的输出
        pred_labels = []
        for c in range(self.class_num):
            class_head = self.__getattr__(f'class_{c}')
            pred_label = class_head(x)
            pred_labels.append(pred_label)
        
        # 拼接所有预测结果
        pred_label = torch.cat(pred_labels, dim=1)
        
        return pred_label

class Backbone_nFC_Id(nn.Module):
    def __init__(self, class_num, id_num, model_name='resnet50_nfc_id'):
        super(Backbone_nFC_Id, self).__init__()
        self.model_name = model_name
        self.backbone_name = model_name.split('_')[0]
        self.class_num = class_num
        self.id_num = id_num
        
        model_ft = getattr(models, self.backbone_name)(pretrained=True)
        if 'resnet' in self.backbone_name:
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft
            self.num_ftrs = 2048
        elif 'densenet' in self.backbone_name:
            model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft.features
            self.num_ftrs = 1024
        else:
            raise NotImplementedError
        
        # 动态创建分类头
        for c in range(self.class_num):
            self.__setattr__(f'class_{c}', nn.Sequential(
                nn.Linear(self.num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, 1),
                nn.Sigmoid()
            ))
        
        # ID分类头
        self.id_head = nn.Sequential(
            nn.Linear(self.num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, self.id_num)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        # 多标签分类预测
        pred_labels = []
        for c in range(self.class_num):
            class_head = self.__getattr__(f'class_{c}')
            pred_label = class_head(x)
            pred_labels.append(pred_label)
        
        pred_label = torch.cat(pred_labels, dim=1)
        
        # ID预测
        pred_id = self.id_head(x)
        
        return pred_label, pred_id
