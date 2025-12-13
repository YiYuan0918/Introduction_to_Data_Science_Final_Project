#!/usr/bin/env python
"""
Inference Script for ViT Classification Model
對單張或多張圖片進行推理預測

Usage:
    # 單張圖片推理
    python infer.py --model-dir outputs/classifier --config configs/cls.yaml --input path/to/image.jpg
    
    # 資料夾批次推理
    python infer.py --model-dir outputs/classifier --config configs/cls.yaml --input path/to/images/
    
    # 顯示 Top-5 預測
    python infer.py --model-dir outputs/classifier --config configs/cls.yaml --input path/to/image.jpg --top-k 5
"""

import argparse
import os
import glob
import yaml
from typing import List

from PIL import Image
import numpy as np
import torch

from models.classifier import ViTMAEForImageClassification


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for trained ViT classifier")
    parser.add_argument("--model-dir", required=True, help="Directory with saved model")
    parser.add_argument("--input", required=True, help="Image file or directory for inference")
    parser.add_argument("--config", required=True, help="Training YAML config file")
    parser.add_argument("--top-k", type=int, default=1, help="Show top-k predictions (default: 1)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    parser.add_argument("--show-probs", action="store_true", help="Show prediction probabilities")
    return parser.parse_args()


def preprocess_image(path: str, img_h: int, img_w: int) -> torch.Tensor:
    """預處理圖片，與訓練時一致"""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    
    # 保持長寬比縮放
    scale = min(img_w / w, img_h / h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.BILINEAR)
    
    # 置中填充黑邊
    canvas = Image.new("RGB", (img_w, img_h), (0, 0, 0))
    left = (img_w - nw) // 2
    top = (img_h - nh) // 2
    canvas.paste(img, (left, top))
    
    # 轉換為 tensor 並正規化
    arr = np.array(canvas).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    
    return torch.from_numpy(arr)


def load_lexicon(root_dir: str) -> List[str]:
    """載入詞彙表"""
    lexicon_path = os.path.join(root_dir, "lexicon.txt")
    if os.path.exists(lexicon_path):
        with open(lexicon_path, "r") as f:
            return [line.strip() for line in f]
    return []


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
    
    # 載入詞彙表
    lexicon = load_lexicon(data_cfg.get("root", "dataset/minDataset"))
    if not lexicon:
        print("⚠️  Warning: Could not load lexicon.txt, will show label indices instead")
    else:
        print(f"📖 Loaded lexicon with {len(lexicon)} words")
    
    # 載入模型
    print(f"📂 Loading model from: {args.model_dir}")
    model = ViTMAEForImageClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Total parameters: {total_params:,}")
    
    # 收集圖片
    if os.path.isdir(args.input):
        exts = ("png", "jpg", "jpeg", "bmp", "PNG", "JPG", "JPEG")
        imgs = []
        for ext in exts:
            imgs.extend(glob.glob(os.path.join(args.input, f"**/*.{ext}"), recursive=True))
        imgs = sorted(imgs)
    else:
        imgs = [args.input]
    
    if len(imgs) == 0:
        raise RuntimeError("❌ No input images found for inference")
    
    print(f"\n🖼️  Found {len(imgs)} image(s) to process\n")
    print("=" * 70)
    
    # 批次處理
    batch_tensors = [preprocess_image(p, img_h, img_w) for p in imgs]
    batch_tensor = torch.stack(batch_tensors).to(device)
    
    with torch.no_grad():
        outputs = model(pixel_values=batch_tensor, return_dict=True)
        logits = outputs.logits  # (B, num_classes)
        probs = torch.softmax(logits, dim=-1)
    
    # 輸出預測結果
    for i, (path, prob) in enumerate(zip(imgs, probs)):
        filename = os.path.basename(path)
        
        # 取得 top-k 預測
        top_probs, top_indices = prob.topk(args.top_k)
        
        print(f"📄 {filename}")
        
        for rank, (idx, p) in enumerate(zip(top_indices.cpu().tolist(), top_probs.cpu().tolist()), 1):
            if lexicon and idx < len(lexicon):
                word = lexicon[idx]
            else:
                word = f"LABEL_{idx}"
            
            if args.show_probs or args.top_k > 1:
                print(f"   #{rank}: {word} ({p*100:.2f}%)")
            else:
                print(f"   Prediction: {word}")
        
        if i < len(imgs) - 1:
            print("-" * 70)
    
    print("=" * 70)
    print(f"\n✅ Inference completed for {len(imgs)} image(s)")


if __name__ == "__main__":
    main()
