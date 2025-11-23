import os
from typing import Optional

from transformers import (
    AutoImageProcessor,
    DataCollatorForImageMasking,
    Trainer,
    TrainingArguments,
    ViTMAEForPreTraining,
)
from transformers.trainer_utils import set_seed

from utils.dataset import Synth90kDataset


def _build_datasets(data_cfg: dict, image_processor: AutoImageProcessor):
    train_ds = Synth90kDataset(
        root_dir=data_cfg["root"],
        mode=data_cfg.get("train_split", "train"),
        img_height=data_cfg.get("img_height", 224),
        img_width=data_cfg.get("img_width", 224),
        image_processor=image_processor,
    )

    eval_ds = None
    if data_cfg.get("val_split"):
        eval_ds = Synth90kDataset(
            root_dir=data_cfg["root"],
            mode=data_cfg.get("val_split"),
            img_height=data_cfg.get("img_height", 224),
            img_width=data_cfg.get("img_width", 224),
            image_processor=image_processor,
        )
    return train_ds, eval_ds


def run_mae_pretraining(cfg: dict, resume_from: Optional[str] = None) -> None:
    seed = cfg.get("seed")
    if seed is not None:
        set_seed(seed)

    model_name = cfg["model"].get("mae_checkpoint", "facebook/vit-mae-base")
    image_processor = AutoImageProcessor.from_pretrained(model_name)

    data_cfg = cfg["data"]
    train_ds, eval_ds = _build_datasets(data_cfg, image_processor)

    model = ViTMAEForPreTraining.from_pretrained(model_name)

    mask_ratio = cfg["model"].get("mask_ratio", 0.75)
    data_collator = DataCollatorForImageMasking(
        image_processor=image_processor,
        mask_ratio=mask_ratio,
    )

    training_cfg = cfg["training"]["mae"]
    evaluation_strategy = "steps" if eval_ds else "no"

    training_args = TrainingArguments(
        output_dir=training_cfg["output_dir"],
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 16),
        num_train_epochs=training_cfg.get("num_train_epochs", 1),
        learning_rate=training_cfg.get("learning_rate", 1e-4),
        weight_decay=training_cfg.get("weight_decay", 0.05),
        warmup_steps=training_cfg.get("warmup_steps", 500),
        logging_steps=training_cfg.get("logging_steps", 50),
        eval_steps=training_cfg.get("eval_steps", 500),
        save_steps=training_cfg.get("save_steps", 1000),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        evaluation_strategy=evaluation_strategy,
        save_strategy=training_cfg.get("save_strategy", "steps"),
        dataloader_num_workers=data_cfg.get("num_workers", 4),
        remove_unused_columns=False,
        fp16=training_cfg.get("fp16", False),
        report_to=training_cfg.get("report_to", "none"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=resume_from or training_cfg.get("resume_from_checkpoint"))
    trainer.save_model(training_cfg["output_dir"])
    image_processor.save_pretrained(training_cfg["output_dir"])


__all__ = ["run_mae_pretraining"]
