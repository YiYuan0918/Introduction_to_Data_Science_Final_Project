from typing import Optional

from transformers import (
    AutoImageProcessor,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
    ViTMAEForPreTraining,
)
from transformers.trainer_utils import set_seed

from data.dataset import Synth90kDataset


class Synth90kImageDictDataset(Synth90kDataset):
    """Wrap Synth90kDataset outputs into the dict format expected by HF data collators."""

    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        image = sample[0] if isinstance(sample, tuple) else sample
        return {"pixel_values": image}


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

    model_name = cfg["model"].get("mae_checkpoint", "facebook/vit-mae-base")
    image_processor = AutoImageProcessor.from_pretrained(model_name)

    data_cfg = cfg["data"]
    train_ds, eval_ds = _build_datasets(data_cfg)

    model = ViTMAEForPreTraining.from_pretrained(model_name)

    data_collator = DefaultDataCollator(return_tensors="pt")

    training_cfg = cfg["training"]["mae"]
    eval_strategy = "steps" if eval_ds else "no"

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
        dataloader_num_workers=int(data_cfg.get("num_workers", 4)),
        remove_unused_columns=False,
        fp16=bool(training_cfg.get("fp16", False)),
        report_to=training_cfg.get("report_to", "none"),
        optim=training_cfg.get("optim", "adamw_torch"),
        adam_beta1=float(training_cfg.get("adam_beta1", 0.9)),
        adam_beta2=float(training_cfg.get("adam_beta2", 0.999)),
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
