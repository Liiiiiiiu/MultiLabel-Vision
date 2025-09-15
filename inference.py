#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Age and gender prediction script with evaluation and Markdown report generation.
Supports PyTorch (.pth) and ONNX (.onnx) models with automatic type detection.
"""
import argparse
import glob
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from torchvision import transforms as T
from tqdm import tqdm

from net import get_model  # Assuming net.py defines get_model

# Configure logging
logging.basicConfig(
    filename="age_gender_predictor.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
AGE_RANGES = ["0-20", "21-40", "41-60", "60>"]
GENDER_LABELS = ["male", "female"]
CLASS_NAMES = ["baby", "child", "Teenager", "Youth", "Middle-age", "old", "gender"]
AGE_MAPPING = {
    "child": "0-20",
    "young": "21-40",
    "middle-age": "41-60",
    "middle": "41-60",
    "old": "60>",
    "elderly": "60>",
    "0-20": "0-20",
    "21-40": "21-40",
    "41-60": "41-60",
    "60>": "60>",
}
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
TEST_INFO_DEFAULTS = {
    "test_date": "2025-05-23",
    "tester": "Brick",
    "test_machine": "ruoyu@192.168.3.91",
}

# Test configuration
TEST_INFO = {
    "测试日期": TEST_INFO_DEFAULTS["test_date"],
    "测试人员": TEST_INFO_DEFAULTS["tester"],
    "测试机器": TEST_INFO_DEFAULTS["test_machine"],
    "模型路径": "",
    "数据路径": "",
    "评估标准": {
        "年龄分段": AGE_RANGES,
        "性别分类": GENDER_LABELS,
        "年龄合并规则": {
            "0-20": ["baby", "child", "Teenager"],
            "21-40": ["Youth"],
            "41-60": ["Middle-age"],
            "60>": ["old"],
        },
    },
}

# Image preprocessing
TRANSFORMS = T.Compose(
    [
        T.Resize(size=(288, 144)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Age and gender prediction with evaluation")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model file (.pth or .onnx)",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to input image or folder",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet50_nfc",
        help="Model name (for PyTorch only, default: resnet50_nfc)",
    )
    parser.add_argument(
        "--num-label",
        type=int,
        default=7,
        help="Number of output labels (for PyTorch only, default: 7)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Enable evaluation mode (requires labeled data)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)",
    )
    parser.add_argument(
        "--test-date",
        type=str,
        default=TEST_INFO_DEFAULTS["test_date"],
        help="Test date (e.g., '2025-05-23 14:00:00')",
    )
    parser.add_argument(
        "--tester",
        type=str,
        default=TEST_INFO_DEFAULTS["tester"],
        help="Name of the tester",
    )
    parser.add_argument(
        "--test-machine",
        type=str,
        default=TEST_INFO_DEFAULTS["test_machine"],
        help="Name or ID of the test machine",
    )
    parser.add_argument(
	"--male",
	type=float,
	default=0.5,
    )
    return parser.parse_args()


def validate_date(date_str: str) -> bool:
    """Validate date string format."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        return True
    except ValueError:
        logger.warning(f"Invalid date format: {date_str}. Expected 'YYYY-MM-DD HH:MM:SS'")
        return False


def get_image_paths(input_path: str) -> List[str]:
    """Retrieve list of image paths from input path."""
    if os.path.isdir(input_path):
        paths = []
        for ext in IMAGE_EXTENSIONS:
            paths.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
        return sorted(paths)
    elif os.path.isfile(input_path) and os.path.splitext(input_path)[1].lower() in IMAGE_EXTENSIONS:
        return [input_path]
    raise ValueError(f"Invalid input path: {input_path}. Must be an image or directory.")


def preprocess_image(image_path: str) -> torch.Tensor:
    """Preprocess a single image."""
    try:
        img = Image.open(image_path).convert("RGB")
        return TRANSFORMS(img)
    except Exception as e:
        logger.error(f"Failed to preprocess {image_path}: {str(e)}")
        raise


def save_markdown_report(results: Dict, output_path: str = "evaluation_report.md") -> None:
    """Save evaluation results as a Markdown file with results-first structure."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 年龄性别识别模型评估报告\n\n")

        # 1. 测试结果
        f.write("## 1. 测试结果\n")
        f.write(f"**总测试图片数量**: {results['total_images']}\n\n")

        f.write("### 1.1 性别预测\n")
        f.write(f"- **准确率**: {results['gender_accuracy']:.4f}\n")
        f.write(f"- 正确预测: {results['gender_correct']}/{results['total_images']}\n")
        f.write(f"- 错误预测: {results['gender_errors_count']}/{results['total_images']}\n\n")
        f.write("**混淆矩阵**:\n")
        f.write("| | Pred Male | Pred Female |\n")
        f.write("|------|------|------|\n")
        cm = results["gender_confusion_matrix"]
        f.write(f"| True Male | {cm[0][0]} | {cm[0][1]} |\n")
        f.write(f"| True Female | {cm[1][0]} | {cm[1][1]} |\n\n")

        if results["gender_errors"]:
            f.write("**错误案例**:\n")
            f.write("| 图片名称 | 真实性别 | 预测性别 |\n")
            f.write("|------|------|------|\n")
            for error in results["gender_errors"]:
                f.write(f"| {os.path.basename(error['image'])} | {error['true']} | {error['pred']} |\n")
            f.write("\n")

        f.write("### 1.2 年龄预测\n")
        f.write(f"- **准确率**: {results['age_accuracy']:.4f}\n")
        f.write(f"- 正确预测: {results['age_correct']}/{results['total_images']}\n")
        f.write(f"- 错误预测: {results['age_errors_count']}/{results['total_images']}\n\n")
        f.write("**混淆矩阵**:\n")
        f.write(f"| 真实\\预测 | {' | '.join(AGE_RANGES)} |\n")
        f.write(f"|------|{'|'.join(['------'] * len(AGE_RANGES))}|\n")
        for i, row in enumerate(results["age_confusion_matrix"]):
            f.write(f"| {AGE_RANGES[i]} | {' | '.join(map(str, row))} |\n")
        f.write("\n")

        if results["age_errors"]:
            f.write("**错误案例**:\n")
            f.write("| 图片名称 | 真实年龄 | 预测年龄 |\n")
            f.write("|------|------|------|\n")
            for error in results["age_errors"]:
                f.write(f"| {os.path.basename(error['image'])} | {error['true']} | {error['pred']} |\n")
            f.write("\n")

        # 2. 评估标准
        f.write("## 2. 评估标准\n")
        f.write("### 2.1 年龄分段\n")
        f.write(f"{', '.join(AGE_RANGES)}\n\n")
        f.write("### 2.2 性别分类\n")
        f.write(f"{', '.join(GENDER_LABELS)}\n\n")

        # 3. 测试基本信息
        f.write("## 3. 测试基本信息\n")
        f.write("| 项目 | 内容 |\n|------|------|\n")
        for key, value in TEST_INFO.items():
            if isinstance(value, dict):
                continue
            f.write(f"| {key} | {value} |\n")

    logger.info(f"Evaluation report saved to: {os.path.abspath(output_path)}")
    print(f"Evaluation report saved to: {os.path.abspath(output_path)}")


def print_evaluation_results(results: Dict) -> None:
    """Print evaluation results to console."""
    print("\n==================== 模型评估报告 ====================")
    print("\n[测试基本信息]")
    for key, value in TEST_INFO.items():
        if isinstance(value, dict):
            continue
        print(f"{key}: {value}")

    print("\n[评估标准]")
    print(f"年龄分段: {', '.join(AGE_RANGES)}")
    print(f"性别分类: {', '.join(GENDER_LABELS)}")

    print("\n[评估结果概览]")
    print(f"总测试图片数量: {results['total_images']}")

    print("\n[性别预测]")
    print(f"准确率: {results['gender_accuracy']:.4f}")
    print(f"正确预测: {results['gender_correct']}/{results['total_images']}")
    print(f"错误预测: {results['gender_errors_count']}/{results['total_images']}")
    print("\n混淆矩阵:")
    print("        Pred Male  Pred Female")
    cm = results["gender_confusion_matrix"]
    print(f"True Male    {cm[0][0]:<10} {cm[0][1]}")
    print(f"True Female  {cm[1][0]:<10} {cm[1][1]}")

    if results["gender_errors"]:
        print("\n错误案例:")
        for error in results["gender_errors"]:
            print(f"图片: {os.path.basename(error['image'])} | 真实: {error['true']} | 预测: {error['pred']}")

    print("\n[年龄预测]")
    print(f"准确率: {results['age_accuracy']:.4f}")
    print(f"正确预测: {results['age_correct']}/{results['total_images']}")
    print(f"错误预测: {results['age_errors_count']}/{results['total_images']}")
    print("\n混淆矩阵:")
    print("        " + "  ".join(f"{r:<6}" for r in AGE_RANGES))
    for i, row in enumerate(results["age_confusion_matrix"]):
        print(f"{AGE_RANGES[i]:<7} " + "  ".join(f"{v:<6}" for v in row))

    if results["age_errors"]:
        print("\n错误案例:")
        for error in results["age_errors"]:
            print(f"图片: {os.path.basename(error['image'])} | 真实: {error['true']} | 预测: {error['pred']}")


class AgeGenderPredictor:
    """Predictor for age and gender classification using PyTorch or ONNX models."""

    def __init__(self, args: argparse.Namespace):
        """Initialize predictor with model and configuration."""
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_type = self._detect_model_type(args.model_path)
        self.model = None
        self.ort_session = None

        if self.model_type == "pytorch":
            self._load_pytorch_model()
        else:
            self._load_onnx_model()

        # Update TEST_INFO
        if not validate_date(args.test_date):
            logger.warning(f"Using test_date as provided: {args.test_date}")
        TEST_INFO.update(
            {
                "测试日期": args.test_date,
                "测试人员": args.tester,
                "测试机器": args.test_machine,
                "模型路径": args.model_path,
                "数据路径": args.input_path,
            }
        )
        logger.info(f"Initialized predictor with model: {args.model_path}")

    def _detect_model_type(self, model_path: str) -> str:
        """Detect model type from file extension."""
        ext = os.path.splitext(model_path)[1].lower()
        if ext == ".pth":
            return "pytorch"
        elif ext == ".onnx":
            return "onnx"
        raise ValueError(f"Unsupported model format: {ext}. Use .pth or .onnx")

    def _load_pytorch_model(self) -> None:
        """Load PyTorch model."""
        if not os.path.exists(self.args.model_path):
            raise FileNotFoundError(f"Model file not found: {self.args.model_path}")
        try:
            self.model = get_model(self.args.model_name, self.args.num_label, use_id=False)
            state_dict = torch.load(self.args.model_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded PyTorch model from {self.args.model_path}")
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {str(e)}")
            raise

    def _load_onnx_model(self) -> None:
        """Load ONNX model."""
        if not os.path.exists(self.args.model_path):
            raise FileNotFoundError(f"ONNX model file not found: {self.args.model_path}")
        try:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if torch.cuda.is_available()
                else ["CPUExecutionProvider"]
            )
            self.ort_session = onnxruntime.InferenceSession(self.args.model_path, providers=providers)
            logger.info(f"Loaded ONNX model from {self.args.model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {str(e)}")
            raise

    def predict_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Predict on a batch of images."""
        if self.model_type == "pytorch":
            batch = batch.to(self.device)
            with torch.no_grad():
                outputs = self.model(batch)
            return outputs.cpu().numpy()
        else:
            ort_inputs = {self.ort_session.get_inputs()[0].name: batch.numpy()}
            return self.ort_session.run(None, ort_inputs)[0]

    def process_outputs(self, outputs: np.ndarray) -> List[Dict]:
        """Process model outputs to age and gender predictions."""
        results = []
        for out in outputs:
            # Age prediction
            age_logits = out[:6]
            age_idx = np.argmax(age_logits)
            age_prob = float(F.softmax(torch.tensor(age_logits), dim=0).numpy()[age_idx])

            age_range = (
                "0-20"
                if age_idx in [0, 1, 2]  # baby, child, Teenager
                else "21-40"
                if age_idx == 3  # Youth
                else "41-60"
                if age_idx == 4  # Middle-age
                else "60>"  # old
            )

            # Gender prediction
            gender_logit = float(out[6])
            gender = "Male" if gender_logit >= self.args.male else "Female"
            gender_conf = gender_logit if gender == "Male" else 1 - gender_logit

            results.append(
                {
                    "gender": gender,
                    "gender_conf": float(gender_conf),
                    "gender_logit": gender_logit,
                    "age_range": age_range,
                    "age_conf": age_prob,
                    "raw_output": out.tolist(),
                }
            )
        return results

    def process_input(self, input_path: str) -> List[Dict]:
        """Process input images and return predictions."""
        image_paths = get_image_paths(input_path)
        if not image_paths:
            raise ValueError(f"No valid images found in {input_path}")

        all_results = []
        for i in tqdm(
            range(0, len(image_paths), self.args.batch_size), desc="Processing images"
        ):
            batch_paths = image_paths[i : i + self.args.batch_size]
            batch_images = []

            for img_path in batch_paths:
                try:
                    img = preprocess_image(img_path)
                    batch_images.append(img)
                except Exception:
                    continue

            if not batch_images:
                continue

            batch = torch.stack(batch_images)
            outputs = self.predict_batch(batch)
            batch_results = self.process_outputs(outputs)

            for img_path, result in zip(batch_paths, batch_results):
                result["image_path"] = img_path
                all_results.append(result)

        logger.info(f"Processed {len(all_results)} images")
        return all_results

    def evaluate(self, input_path: str) -> Dict:
        """Evaluate model on labeled dataset."""
        if not os.path.isdir(input_path):
            raise ValueError("Evaluation requires a directory of labeled images")

        image_paths = get_image_paths(input_path)
        if not image_paths:
            raise ValueError(f"No images found in {input_path}")

        y_true_gender, y_true_age, valid_image_paths = [], [], []
        gender_errors, age_errors = [], []

        for img_path in image_paths:
            try:
                filename = os.path.basename(img_path)
                parts = filename.split("_")
                if len(parts) < 3:
                    logger.warning(f"Skipping {filename} - invalid format")
                    continue

                gender_gt = parts[-2].lower()
                age_gt = parts[-1].split(".")[0].lower()

                if gender_gt not in ["male", "female"]:
                    logger.warning(f"Skipping {filename} - invalid gender: {gender_gt}")
                    continue

                if age_gt not in AGE_MAPPING:
                    logger.warning(f"Skipping {filename} - invalid age label: {age_gt}")
                    continue

                age_range = AGE_MAPPING[age_gt]
                y_true_gender.append(0 if gender_gt == "male" else 1)
                y_true_age.append(AGE_RANGES.index(age_range))
                valid_image_paths.append(img_path)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                continue

        if not valid_image_paths:
            raise ValueError("No valid labeled images found")

        self.args.batch_size = min(32, self.args.batch_size)
        results = self.process_input(input_path)

        if len(results) != len(valid_image_paths):
            logger.warning(
                f"Processed {len(results)} out of {len(valid_image_paths)} valid images"
            )

        y_pred_gender, y_pred_age = [], []
        for i, result in enumerate(results):
            pred_gender = result["gender"]
            pred_age = result["age_range"]
            true_gender = "male" if y_true_gender[i] == 0 else "female"
            true_age = AGE_RANGES[y_true_age[i]]

            y_pred_gender.append(0 if pred_gender == "Male" else 1)
            y_pred_age.append(AGE_RANGES.index(pred_age))

            if pred_gender.lower() != true_gender:
                gender_errors.append(
                    {"image": result["image_path"], "true": true_gender, "pred": pred_gender}
                )
            if pred_age != true_age:
                age_errors.append(
                    {"image": result["image_path"], "true": true_age, "pred": pred_age}
                )

        gender_acc = accuracy_score(y_true_gender, y_pred_gender)
        age_acc = accuracy_score(y_true_age, y_pred_age)
        gender_cm = confusion_matrix(y_true_gender, y_pred_gender, labels=[0, 1])
        age_cm = confusion_matrix(y_true_age, y_pred_age, labels=list(range(len(AGE_RANGES))))

        result_dict = {
            "total_images": len(valid_image_paths),
            "gender_accuracy": gender_acc,
            "age_accuracy": age_acc,
            "gender_confusion_matrix": gender_cm.tolist(),
            "age_confusion_matrix": age_cm.tolist(),
            "gender_correct": int(gender_acc * len(valid_image_paths)),
            "gender_errors_count": len(valid_image_paths) - int(gender_acc * len(valid_image_paths)),
            "age_correct": int(age_acc * len(valid_image_paths)),
            "age_errors_count": len(valid_image_paths) - int(age_acc * len(valid_image_paths)),
            "gender_errors": gender_errors,
            "age_errors": age_errors,
        }

        print_evaluation_results(result_dict)
        save_markdown_report(result_dict)
        return result_dict


def main():
    """Main function to run prediction or evaluation."""
    args = parse_args()
    predictor = AgeGenderPredictor(args)

    if args.eval:
        predictor.evaluate(args.input_path)
    else:
        results = predictor.process_input(args.input_path)
        for result in results:
            print(f"\nImage: {result['image_path']}")
            print(f"Predicted Gender: {result['gender']} (confidence: {result['gender_conf']:.4f})")
            print(f"Predicted Age Range: {result['age_range']} (confidence: {result['age_conf']:.4f})")
            print(f"Raw outputs: {result['raw_output']}")


if __name__ == "__main__":
    main()
