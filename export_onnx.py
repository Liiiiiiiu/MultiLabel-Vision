import torch
import torch.nn as nn
import argparse
import os

# 假设您的模型定义在models1.py中，根据您的代码导入
from net import get_model  # 您的get_model函数

def load_model_for_export(checkpoint_path, backbone='resnet50', num_labels=3, device='cpu', use_dynamic=False):
    """
    加载训练好的模型，用于ONNX导出。
    处理DataParallel包装，并返回纯模型。
    
    Args:
        use_dynamic: 是否使用动态模型（例如动态输入形状支持）。如果True，ONNX导出时启用更多动态轴。
    """
    # 创建模型实例（与训练时相同）
    model = get_model(backbone, num_label=num_labels, use_id=False)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # 处理DataParallel的前缀
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # 设置为评估模式
    
    print(f"模型已从 {checkpoint_path} 加载完成，输出维度: {num_labels}")
    print(f"动态模型模式: {use_dynamic}")
    
    return model

def export_to_onnx(model, onnx_path, input_size=(1, 3, 224, 224), opset_version=12, use_dynamic=False):
    """
    将PyTorch模型导出为ONNX格式。
    
    Args:
        use_dynamic: 是否启用动态轴（例如batch_size、H、W动态）。如果True，支持可变输入形状。
    """
    # 创建dummy input
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size, device=device)
    
    if use_dynamic:
        dynamic_axes = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size'}
        }
        print("启用动态轴: batch_size, height, width")
    else:
        dynamic_axes = {}  # 无动态轴，batch_size固定为1
        print("无动态轴，固定batch_size=1") 
    # 导出ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,  # 导出训练好的参数
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    print(f"模型已导出为ONNX: {onnx_path}")

def main():
    parser = argparse.ArgumentParser(description='将PyTorch模型转换为ONNX')
    parser.add_argument('--checkpoint', required=True, help='训练好的检查点路径 (e.g., model_best.pth)')
    parser.add_argument('--backbone', default='resnet50', help='骨干网络 (default: resnet50)')
    parser.add_argument('--num_labels', type=int, default=3, help='标签数量 (default: 3)')
    parser.add_argument('--onnx_path', default='model.onnx', help='ONNX输出路径 (default: model.onnx)')
    parser.add_argument('--device', default='cpu', help='设备 (cpu or cuda)')
    parser.add_argument('--input_size', nargs=4, type=int, default=[1, 3, 224, 224], help='输入形状 (batch, C, H, W)')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset版本 (default: 12)')
    parser.add_argument('--use_dynamic', action='store_true', help='启用动态模型（支持可变batch_size和图像尺寸）')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    
    # 加载模型（传递use_dynamic参数）
    model = load_model_for_export(args.checkpoint, args.backbone, args.num_labels, device, args.use_dynamic)
    
    # 导出（根据use_dynamic调整动态轴）
    export_to_onnx(model, args.onnx_path, tuple(args.input_size), args.opset, args.use_dynamic)

if __name__ == '__main__':
    main()
