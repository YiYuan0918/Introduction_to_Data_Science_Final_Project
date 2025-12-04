import argparse
import os
import glob
import yaml

from typing import List, Optional

from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from models.classifier import ViTMAEForCTC
from data.dataset import Synth90kDataset


def parse_args():
    p = argparse.ArgumentParser(description="Inference + learning-curve plot for trained ViT-MAE CTC classifier")
    p.add_argument("--model-dir", required=True, help="Directory with saved model (HF save_pretrained style)")
    p.add_argument("--input", required=True, help="Image file or directory for inference")
    p.add_argument("--config", default=None, help="Optional training YAML config (used to display hyperparams and image size)")
    p.add_argument("--log-csv", default=None, help="Optional CSV training log (overrides detection in model-dir)")
    p.add_argument("--device", default=None, help="cuda or cpu (default auto)")
    return p.parse_args()


def preprocess_image(path: str, img_h: int, img_w: int):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(img_w / w, img_h / h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (img_w, img_h), (0, 0, 0))
    left = (img_w - nw) // 2
    top = (img_h - nh) // 2
    canvas.paste(img, (left, top))
    arr = np.array(canvas).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return torch.from_numpy(arr)


def ctc_greedy_decode(logits: torch.Tensor, label_map: dict) -> List[str]:
    # logits: (B, T, C)
    preds = logits.argmax(-1).cpu().numpy()
    outputs: List[str] = []
    for seq in preds:
        prev = None
        chars: List[str] = []
        for idx in seq:
            if idx == prev:
                prev = idx
                continue
            prev = idx
            if int(idx) == 0:
                continue  # blank
            chars.append(label_map[int(idx)])
        outputs.append("".join(chars))
    return outputs


def plot_learning_curves(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 4))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plotted = False
    if "loss" in df.columns:
        plt.plot(df["loss"], label="train_loss")
        plotted = True
    if "eval_loss" in df.columns:
        plt.plot(df["eval_loss"], label="val_loss")
        plotted = True
    if plotted:
        plt.xlabel("record")
        plt.ylabel("loss")
        plt.title("Loss")
        plt.legend()

    # Accuracy subplot (search common names)
    plt.subplot(1, 2, 2)
    acc_candidates = [c for c in df.columns if "accuracy" in c or c.endswith("acc") or "acc" in c]
    acc_plotted = False
    for c in acc_candidates[:4]:
        plt.plot(df[c], label=c)
        acc_plotted = True
    if acc_plotted:
        plt.xlabel("record")
        plt.ylabel("accuracy")
        plt.title("Accuracy")
        plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_h, img_w = 224, 224
    cfg = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        data_cfg = cfg.get("data", {})
        if data_cfg:
            img_h = int(data_cfg.get("img_height", img_h))
            img_w = int(data_cfg.get("img_width", img_w))

    # load model (assumes HF save_pretrained style)
    model = ViTMAEForCTC.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    total, trainable = count_params(model)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")

    # label map provided by dataset (CTC blank = 0, chars start from 1)
    LABEL2CHAR = Synth90kDataset.LABEL2CHAR

    # collect images
    if os.path.isdir(args.input):
        exts = ("png", "jpg", "jpeg", "bmp")
        imgs = []
        for e in exts:
            imgs.extend(glob.glob(os.path.join(args.input, f"**/*.{e}"), recursive=True))
        imgs = sorted(imgs)
    else:
        imgs = [args.input]

    if len(imgs) == 0:
        raise RuntimeError("No input images found for inference")

    batch = [preprocess_image(p, img_h, img_w) for p in imgs]
    batch_tensor = torch.stack(batch).to(device)

    with torch.no_grad():
        out = model(pixel_values=batch_tensor, return_dict=True)
        logits = out.logits  # expected (B, T, C)

    preds = ctc_greedy_decode(logits, LABEL2CHAR)
    for path, text in zip(imgs, preds):
        print(f"{os.path.basename(path)} -> {text}")

    # find csv log: explicit override else first csv in model-dir
    csv_path: Optional[str] = args.log_csv
    if csv_path is None:
        csv_candidates = glob.glob(os.path.join(args.model_dir, "*.csv"))
        csv_path = csv_candidates[0] if csv_candidates else None

    if csv_path:
        out_fig = os.path.join(args.model_dir, "learning_curve.png")
        plot_learning_curves(csv_path, out_fig)
        print("Saved learning curve to", out_fig)

        # print numeric summary
        df = pd.read_csv(csv_path)
        summary = {}
        if "loss" in df.columns:
            summary["min_train_loss"] = float(df["loss"].min())
            summary["last_train_loss"] = float(df["loss"].iloc[-1])
        if "eval_loss" in df.columns:
            summary["min_val_loss"] = float(df["eval_loss"].min())
            summary["last_val_loss"] = float(df["eval_loss"].iloc[-1])
        # accuracy candidates
        acc_cols = [c for c in df.columns if "accuracy" in c or c.endswith("acc") or "acc" in c]
        for c in acc_cols[:4]:
            summary[f"max_{c}"] = float(df[c].max())
            summary[f"last_{c}"] = float(df[c].iloc[-1])
        print("Log summary:", summary)
    else:
        print("No CSV training log found in model-dir; pass --log-csv to specify")

    # print hyperparams summary if config provided
    if cfg:
        training_cfg = cfg.get("training", {})
        data_cfg = cfg.get("data", {})
        model_cfg = cfg.get("model", {})
        print("Config summary:")
        print("  data keys:", list(data_cfg.keys()))
        print("  training keys:", list(training_cfg.keys()))
        print("  model keys:", list(model_cfg.keys()))
        # show common hyperparams if exist
        hp = {}
        if training_cfg:
            for k in ("learning_rate", "lr", "batch_size", "epochs", "weight_decay"):
                if k in training_cfg:
                    hp[k] = training_cfg[k]
        if hp:
            print("  example hyperparams:", hp)


if __name__ == "__main__":
    main()