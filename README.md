# LinkDome Multi-Label Classification Project

## Overview
This project, developed by **Brick** at **LinkDome**, implements a multi-label classification framework for image data using PyTorch. It supports training a ResNet50-based model (extensible to other backbones like ResNet34, ResNet18, or DenseNet121) to classify images with 3 binary labels. The framework includes data loading, model training, evaluation, and ONNX model export for deployment. It is designed to handle large datasets, supports multi-GPU training, and provides comprehensive metrics (loss, accuracy, F1 scores) with training curve visualization.

The codebase is modular, with utilities for data preprocessing, model setup, loss calculation, and checkpoint management. This README provides instructions for environment setup, data preparation, training, and ONNX export.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training Process](#training-process)
- [ONNX Export](#onnx-export)
- [Directory Structure](#directory-structure)
- [Troubleshooting](#troubleshooting)
- [Contact](#contact)

## Environment Setup
### Prerequisites
- **Operating System**: Linux (recommended) or Windows
- **Python**: 3.8+
- **Hardware**: GPU (NVIDIA CUDA-compatible) recommended for faster training; CPU supported

### Dependencies
Install the required Python packages using `pip`:

```bash
pip install torch torchvision numpy matplotlib scikit-learn pillow
```

For ONNX export:
```bash
pip install onnx onnxruntime
```

Optional for advanced data augmentation:
```bash
pip install albumentations
```

### Recommended Environment
Use a virtual environment (e.g., `venv` or `conda`) to avoid conflicts:
```bash
python -m venv linkdome_env
source linkdome_env/bin/activate  # Linux
linkdome_env\Scripts\activate  # Windows
pip install -r requirements.txt
```

Create a `requirements.txt` with:
```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
pillow>=8.0.0
onnx>=1.9.0
onnxruntime>=1.8.0
```

## Data Preparation
The framework expects a dataset with images and corresponding multi-label annotations (3 binary labels: 0 or 1). The data should be organized as follows:

### Dataset Structure
- **Root Directory**: Specify a root directory (e.g., `/path/to/data`) containing images and a `train_list.txt` file.
- **Images**: Stored in subdirectories or directly under the root (e.g., `data/images/img1.jpg`).
- **Labels File**: A `train_list.txt` file with the format:
  ```
  images/img1.jpg	0,1,0
  images/img2.jpg	1,0,1
  ```
  - Each line: `<relative_image_path>\t<label1>,<label2>,<label3>`
  - Labels are comma-separated binary values (0 or 1).
  - Paths are relative to the data root.

Example:
```
/path/to/data/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── train_list.txt
```

### Notes
- Ensure images are in formats like JPG, PNG, or BMP.
- If `train_list.txt` is missing, the framework attempts to scan the directory and parse labels from filenames (not implemented by default; customize `_parse_labels_from_path` in `dataload.py` if needed).
- Dataset size in example: ~95k training samples, ~10k validation (20% split).

### Data Augmentation
- **Training**: RandomResizedCrop (224x224), RandomHorizontalFlip, RandomRotation (10°), ColorJitter, Normalize (ImageNet mean/std).
- **Validation**: Resize (256), CenterCrop (224), Normalize.

To customize, modify `get_train_transforms` or `get_val_transforms` in `dataload.py`.

## Training Process
The training script (`LD_TRAIN_MUL_CLA.py`) supports multi-label classification with ResNet50, configurable via command-line arguments.

### Running Training
```bash
python LD_TRAIN_MUL_CLA.py \
  --data-root /path/to/data \
  --backbone resnet50 \
  --batch-size 32 \
  --num-epoch 10 \
  --num-workers 4 \
  --device cuda \
  --learning-rate 0.01 \
  --val-size 0.2 \
  --random-seed 42 \
  --loss-type bce
```

### Key Arguments
- `--data-root`: Path to dataset directory (required).
- `--backbone`: Model architecture (default: `resnet50`; options: `resnet34`, `resnet18`, `densenet121`).
- `--batch-size`: Batch size (default: 32, adjust based on GPU memory).
- `--num-epoch`: Number of epochs (default: 10).
- `--device`: `cuda` (GPU) or `cpu` (default: `cuda`).
- `--loss-type`: Loss function (`bce`, `bce_with_logits`, or `focal`; default: `bce`).
- `--pretrain`: Path to pretrained weights (optional).
- `--resume`: Path to checkpoint for resuming training (optional).

### Training Features
- **Model**: ResNet50 with 3 separate classification heads (Linear(2048→512)→ReLU→Dropout(0.5)→Linear(512→1)→Sigmoid).
- **Loss**: Binary Cross-Entropy (BCE) by default, with support for BCEWithLogits or Focal Loss.
- **Optimizer**: SGD with momentum (0.9), weight decay (5e-4), and step learning rate decay (step_size=40, gamma=0.1).
- **Metrics**: Sample accuracy, exact match accuracy, per-label accuracy, macro/micro F1 scores.
- **Outputs**: Checkpoints saved every 10 epochs or when validation F1 improves (`checkpoints/multilabel/resnet50_multilabel/`), plus training curves (`training_curves.png`).

### Example Output
```
Epoch 6/10
--------------------------------------------------
train: [  128/94448] (  0%) | 标签损失: 0.5833 | 准确率: 0.7083 | 耗时: 2.01s
...
val: [  128/10495] (  1%) | 标签损失: 0.5637 | 准确率: 0.7344 | 耗时: 1.17s
✓ 新的最佳模型，验证F1: 0.2543
模型已保存至: ./checkpoints/multilabel/resnet50_multilabel/model_epoch_006.pth
```

### Optimization Tips
- **Label Imbalance**: If F1 is low (<0.3), check label distribution in `_setup_data` (add `Counter` for labels). Use `--loss-type bce_with_logits` with `pos_weight` in `MultiLabelLoss`.
- **Threshold Tuning**: Adjust threshold (default 0.5) in `calculate_metrics` (e.g., 0.3) to boost F1.
- **Hyperparameters**: Try `--learning-rate 0.001`, `--loss-type focal`, or AdamW optimizer for faster convergence.

## ONNX Export
The `export_to_onnx.py` script converts the trained model to ONNX format for deployment.

### Running Export
```bash
python export_to_onnx.py \
  --checkpoint ./checkpoints/multilabel/resnet50_multilabel/model_best.pth \
  --backbone resnet50 \
  --num_labels 3 \
  --onnx_path resnet50_multilabel.onnx \
  --device cpu
```

### Key Arguments
- `--checkpoint`: Path to trained model checkpoint (required).
- `--onnx_path`: Output ONNX file path (default: `model.onnx`).
- `--use_dynamic`: Enable dynamic axes for batch_size, height, width (default: False, fixes batch_size=1).
- `--input_size`: Input shape (default: `[1, 3, 224, 224]`).

### Notes
- Non-dynamic mode (`--use_dynamic` not set) fixes batch_size=1, suitable for single-image inference.
- Dynamic mode (`--use_dynamic`) supports variable batch sizes and image dimensions.
- Verify the ONNX model with Netron or test inference with `onnxruntime`.

## Directory Structure
```
LinkDome_MultiLabel/
├── LD_TRAIN_MUL_CLA.py       # Main training script
├── export_to_onnx.py         # ONNX export script
├── common/
│   └── utils.py             # Utilities (device setup, model saving, etc.)
├── LD_dataloader/
│   └── dataload.py          # Data loading and augmentation
├── net/
│   ├── __init__.py          # Model factory
│   └── models1.py           # Model definitions (Backbone_nFC)
├── checkpoints/             # Saved models and curves
└── requirements.txt         # Dependencies
```

## Troubleshooting
- **Data Errors**: Ensure `train_list.txt` exists and follows the correct format. If missing, implement `_parse_labels_from_path` in `dataload.py`.
- **Low F1 Score**: Check label distribution for imbalance. Add `pos_weight` to `MultiLabelLoss` or use `--loss-type focal`.
- **ONNX Issues**: Verify checkpoint compatibility and input shapes. Test ONNX model with `onnxruntime`.
- **GPU Memory**: Reduce `--batch-size` (e.g., 16) if CUDA out-of-memory occurs.

For further issues, contact Brick with error logs.

## Contact
- **Author**: Brick
- **Company**: LinkDome
- **Email**: ljx903010@gmail.com
- **Date**: September 15, 2025

Thank you for using the LinkDome Multi-Label Classification Framework!
