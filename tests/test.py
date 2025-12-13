#!/usr/bin/env python
"""
Test Set Evaluation Script for ViT Classification Model
對測試集進行評估，計算 accuracy, precision, recall, F1 等指標

Usage:
    python tests/test.py --model-dir outputs/classifier --config configs/cls.yaml
    python tests/test.py --model-dir outputs/classifier --config configs/cls.yaml --split test
"""

import argparse
import os
import yaml
from tqdm import tqdm

import sys
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

# 添加專案根目錄到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.classifier import ViTMAEForImageClassification
from data.dataset import Synth90kDataset, synth90k_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained classifier on test set")
    parser.add_argument("--model-dir", required=True, help="Directory with saved model")
    parser.add_argument("--config", required=True, help="Training YAML config file")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate (default: test)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    return parser.parse_args()


def evaluate(model, dataloader, device):
    """評估模型在資料集上的表現"""
    model.eval()
    
    total_samples = 0
    correct_predictions = 0
    total_loss = 0.0
    
    # Top-K accuracy tracking
    top5_correct = 0
    top10_correct = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # synth90k_collate_fn 返回 (images, labels) tuple
            pixel_values, labels = batch
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels, return_dict=True)
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Top-1 predictions
            predictions = logits.argmax(dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            
            # Top-5 predictions
            _, top5_preds = logits.topk(5, dim=-1)
            top5_correct += (top5_preds == labels.unsqueeze(-1)).any(dim=-1).sum().item()
            
            # Top-10 predictions
            _, top10_preds = logits.topk(10, dim=-1)
            top10_correct += (top10_preds == labels.unsqueeze(-1)).any(dim=-1).sum().item()
            
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    # 計算指標
    accuracy = correct_predictions / total_samples
    top5_accuracy = top5_correct / total_samples
    top10_accuracy = top10_correct / total_samples
    avg_loss = total_loss / total_samples
    
    # 計算 weighted precision, recall, F1
    print("\n📊 Computing precision, recall, F1 (this may take a moment)...")
    precision_weighted = precision_score(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    recall_weighted = recall_score(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    f1_weighted = f1_score(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "top5_accuracy": top5_accuracy,
        "top10_accuracy": top10_accuracy,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "loss": avg_loss,
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def main():
    args = parse_args()
    
    # 設定裝置
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🖥️  Using device: {device}")
    
    # 載入配置
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    data_cfg = cfg.get("data", {})
    img_h = data_cfg.get("img_height", 48)
    img_w = data_cfg.get("img_width", 96)
    
    print(f"📐 Image size: {img_h} × {img_w}")
    
    # 載入模型
    print(f"📂 Loading model from: {args.model_dir}")
    model = ViTMAEForImageClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Total parameters: {total_params:,}")
    
    # 載入資料集
    print(f"📁 Loading {args.split} dataset...")
    dataset = Synth90kDataset(
        root_dir=data_cfg["root"],
        mode=args.split,
        img_height=img_h,
        img_width=img_w,
    )
    print(f"   Dataset size: {len(dataset):,} samples")
    
    # 創建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=synth90k_collate_fn,
        pin_memory=True,
    )
    
    # 評估
    print(f"\n🔍 Evaluating on {args.split} set...")
    results = evaluate(model, dataloader, device)
    
    # 輸出結果
    print("\n" + "=" * 60)
    print(f"📊 Evaluation Results ({args.split} set)")
    print("=" * 60)
    print(f"   Total Samples: {results['total_samples']:,}")
    print(f"   Correct Predictions: {results['correct_predictions']:,}")
    print(f"   ")
    print(f"   📈 Top-1 Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"   📈 Top-5 Accuracy:  {results['top5_accuracy']*100:.2f}%")
    print(f"   📈 Top-10 Accuracy: {results['top10_accuracy']*100:.2f}%")
    print(f"   ")
    print(f"   🎯 Precision (weighted): {results['precision_weighted']*100:.2f}%")
    print(f"   🎯 Recall (weighted):    {results['recall_weighted']*100:.2f}%")
    print(f"   🎯 F1-Score (weighted):  {results['f1_weighted']*100:.2f}%")
    print(f"   ")
    print(f"   📉 Average Loss: {results['loss']:.4f}")
    print("=" * 60)
    
    # 儲存結果到 test.py 所在的資料夾 (tests/)
    import json
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(tests_dir, f"results_{args.split}.json")
    
    save_results = {
        "split": args.split,
        "total_samples": results["total_samples"],
        "correct_predictions": results["correct_predictions"],
        "accuracy": results["accuracy"],
        "top5_accuracy": results["top5_accuracy"],
        "top10_accuracy": results["top10_accuracy"],
        "precision_weighted": results["precision_weighted"],
        "recall_weighted": results["recall_weighted"],
        "f1_weighted": results["f1_weighted"],
        "loss": results["loss"],
    }
    
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n✅ Results saved to: {results_path}")


if __name__ == "__main__":
    main()
