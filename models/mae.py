from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
    ViTMAEForPreTraining,
)
from transformers.trainer_utils import set_seed
from transformers.utils import logging

from data.dataset import Synth90kDataset
from utils.logging_callbacks import CsvLoggingCallback

logger = logging.get_logger(__name__)


class Synth90kImageDictDataset(Synth90kDataset):
    """Wrap Synth90kDataset outputs into the dict format expected by HF data collators."""

    def __getitem__(self, idx: int):
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                path = self.paths[idx]
                image = Image.open(path).convert('RGB')
                
                # resize
                w, h = image.size
                scale = min(self.img_width / w, self.img_height / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = image.resize((new_w, new_h), resample=Image.BILINEAR)
                
                # padding
                canvas = Image.new("RGB", (self.img_width, self.img_height), (0, 0, 0))
                left = (self.img_width - new_w) // 2
                top = (self.img_height - new_h) // 2
                canvas.paste(image, (left, top))
                
                # Normalization
                img_np = np.array(canvas).astype("float32") / 255.0
                img_np = (img_np - self.mean) / self.std
                img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
                image = torch.FloatTensor(img_np)
                
                return {"pixel_values": image}
            except (IOError, OSError):
                logger.warning(f'Corrupted image at index {idx}, attempting next image...')
                idx = (idx + 1) % len(self.paths)
                if attempt == max_attempts - 1:
                    # If all attempts fail, raise an error
                    raise RuntimeError(f"Failed to load a valid image after {max_attempts} attempts")


def _build_datasets(data_cfg: dict):
    train_ds = Synth90kImageDictDataset(
        root_dir=data_cfg["root"],
        mode="train",
        img_height=data_cfg.get("img_height", 224),
        img_width=data_cfg.get("img_width", 224),
    )

    eval_ds = None
    if data_cfg.get("val_split") is not None:
        eval_ds = Synth90kImageDictDataset(
            root_dir=data_cfg["root"],
            mode="val",
            img_height=data_cfg.get("img_height", 224),
            img_width=data_cfg.get("img_width", 224),
        )
    return train_ds, eval_ds


def run_mae_pretraining(cfg: dict, resume_from: Optional[str] = None) -> None:
    seed = cfg.get("seed")
    if seed is not None:
        set_seed(seed)

    model_cfg = cfg["model"]
    model_name = model_cfg.get("mae_checkpoint", "facebook/vit-mae-base")

    data_cfg = cfg["data"]
    img_h = int(data_cfg.get("img_height", 224))
    img_w = int(data_cfg.get("img_width", 224))

    # ViT-MAE expects square inputs. If a rectangular size is provided, upscale
    # the smaller side so that both dimensions match the largest one.
    target_image_size = max(img_h, img_w)
    if img_h != img_w:
        logger.warning(
            f"MAE expects square inputs; received {img_h}x{img_w}. "
            f"Padding/resizing to square {target_image_size}x{target_image_size}."
        )
        img_h = img_w = target_image_size
        data_cfg = {**data_cfg, "img_height": img_h, "img_width": img_w}

    image_processor = AutoImageProcessor.from_pretrained(model_name)

    # Update image_processor size to match training dimensions
    # This ensures consistency when using the saved processor for inference
    if hasattr(image_processor, "size"):
        image_processor.size = {"height": img_h, "width": img_w}
    if hasattr(image_processor, "crop_size"):
        image_processor.crop_size = {"height": img_h, "width": img_w}

    train_ds, eval_ds = _build_datasets(data_cfg)

    # Check if using custom dimensions (different from pretrained default 224)
    using_custom_dimensions = target_image_size != 224

    if using_custom_dimensions:
        # For custom dimensions, train MAE from scratch (no pretrained weights)
        # This avoids position embeddings mismatch and input size validation issues
        logger.info(
            f"Using custom image dimensions {img_h}×{img_w} (target_image_size={target_image_size}). "
            f"Training MAE from scratch with randomly initialized weights."
        )

        # Load config from pretrained to get architecture, but initialize model from scratch
        from transformers import ViTMAEConfig
        config = ViTMAEConfig.from_pretrained(model_name)
        config.image_size = target_image_size

        # Set mask_ratio if specified
        if "mask_ratio" in model_cfg:
            mask_ratio = float(model_cfg["mask_ratio"])
            if not 0.0 <= mask_ratio <= 1.0:
                raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
            config.mask_ratio = mask_ratio
            logger.info(f"Setting MAE mask_ratio to {mask_ratio} from config")

        # Initialize model from scratch with custom config
        model = ViTMAEForPreTraining(config)
        logger.info(f"Initialized MAE model from scratch with image_size={model.config.image_size}, mask_ratio={model.config.mask_ratio}")

    else:
        # Standard dimensions (224×224) - load pretrained weights
        model_kwargs = {}

        # Set mask_ratio if specified
        if "mask_ratio" in model_cfg:
            mask_ratio = float(model_cfg["mask_ratio"])
            if not 0.0 <= mask_ratio <= 1.0:
                raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
            model_kwargs["mask_ratio"] = mask_ratio
            logger.info(f"Setting MAE mask_ratio to {mask_ratio} from config")

        model = ViTMAEForPreTraining.from_pretrained(model_name, **model_kwargs)
        logger.info(f"Loaded pretrained MAE model with image_size={model.config.image_size}, mask_ratio={model.config.mask_ratio}")

    data_collator = DefaultDataCollator(return_tensors="pt")

    training_cfg = cfg["training"]["mae"]
    eval_strategy = "steps" if eval_ds else "no"
    load_best = eval_strategy != "no" and bool(training_cfg.get("load_best_model_at_end", True))

    training_args = TrainingArguments(
        output_dir=training_cfg["output_dir"],
        per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(training_cfg.get("per_device_eval_batch_size", 16)),
        num_train_epochs=int(training_cfg.get("num_train_epochs", 1)),
        learning_rate=float(training_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.05)),
        warmup_steps=int(training_cfg.get("warmup_steps", 500)),
        logging_steps=int(training_cfg.get("logging_steps", 50)),
        eval_steps=int(training_cfg.get("eval_steps", 500)),
        save_steps=int(training_cfg.get("save_steps", 1000)),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
        eval_strategy=eval_strategy,
        save_strategy=training_cfg.get("save_strategy", "steps"),
        load_best_model_at_end=load_best,
        metric_for_best_model=training_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=bool(training_cfg.get("greater_is_better", False)),
        save_total_limit=training_cfg.get("save_total_limit"),
        dataloader_num_workers=int(data_cfg.get("num_workers", 4)),
        remove_unused_columns=False,
        fp16=bool(training_cfg.get("fp16", False)),
        report_to=training_cfg.get("report_to", "none"),
        optim=training_cfg.get("optim", "adamw_torch"),
        adam_beta1=float(training_cfg.get("adam_beta1", 0.9)),
        adam_beta2=float(training_cfg.get("adam_beta2", 0.999)),
        label_names=["pixel_values"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=[CsvLoggingCallback(training_cfg["output_dir"])],
    )

    trainer.train(resume_from_checkpoint=resume_from or training_cfg.get("resume_from_checkpoint"))
    trainer.save_model(training_cfg["output_dir"])
    image_processor.save_pretrained(training_cfg["output_dir"])


__all__ = ["run_mae_pretraining"]
