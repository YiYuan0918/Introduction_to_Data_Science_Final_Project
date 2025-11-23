from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    ViTMAEConfig,
    ViTMAEModel,
)
from transformers.modeling_outputs import ModelOutput
from transformers.trainer_utils import set_seed
from transformers.utils import logging

from utils.dataset import Synth90kCTCCollator, Synth90kDataset

logger = logging.get_logger(__name__)


@dataclass
class CTCOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class ViTMAEForCTC(PreTrainedModel):
    """Thin classification head on top of ViT-MAE encoder for CTC OCR."""

    config_class = ViTMAEConfig

    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__(config)
        self.encoder = ViTMAEModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
        self.post_init()

    def forward(self, pixel_values, labels=None, label_lengths=None, return_dict: Optional[bool] = None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.encoder(pixel_values=pixel_values, return_dict=return_dict)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None and label_lengths is not None:
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # (T, B, C)
            input_lengths = torch.full(
                size=(log_probs.size(1),),
                fill_value=log_probs.size(0),
                dtype=torch.long,
                device=log_probs.device,
            )
            loss = self.ctc_loss(log_probs, labels, input_lengths, label_lengths)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CTCOutput(loss=loss, logits=logits)


def _load_vitmae_config(checkpoint: str, vocab_size: int) -> ViTMAEConfig:
    config = ViTMAEConfig.from_pretrained(checkpoint)
    config.vocab_size = vocab_size
    return config


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


def run_classification_training(cfg: dict, resume_from: Optional[str] = None) -> None:
    seed = cfg.get("seed")
    if seed is not None:
        set_seed(seed)

    vocab_size = len(Synth90kDataset.CHARS) + 1  # +1 for CTC blank

    training_cfg = cfg["training"]["classifier"]
    mae_init = training_cfg.get("mae_checkpoint_for_init") or cfg["model"].get("mae_checkpoint")
    if mae_init is None:
        raise ValueError("mae_checkpoint_for_init or model.mae_checkpoint must be provided")

    image_processor = AutoImageProcessor.from_pretrained(mae_init)
    data_cfg = cfg["data"]
    train_ds, eval_ds = _build_datasets(data_cfg, image_processor)

    config = _load_vitmae_config(mae_init, vocab_size)
    model = ViTMAEForCTC(config=config)

    if mae_init:
        try:
            state = ViTMAEModel.from_pretrained(mae_init).state_dict()
            missing, unexpected = model.encoder.load_state_dict(state, strict=False)
            if unexpected:
                logger.warning("Unexpected keys when loading MAE weights: %s", unexpected)
            if missing:
                logger.info("Missing encoder keys when loading MAE weights: %s", missing)
        except OSError:
            logger.warning("Could not load MAE weights from %s, training encoder from scratch", mae_init)

    collator = Synth90kCTCCollator()
    evaluation_strategy = "steps" if eval_ds else "no"

    training_args = TrainingArguments(
        output_dir=training_cfg["output_dir"],
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 16),
        num_train_epochs=training_cfg.get("num_train_epochs", 1),
        learning_rate=training_cfg.get("learning_rate", 3e-4),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        warmup_steps=training_cfg.get("warmup_steps", 250),
        logging_steps=training_cfg.get("logging_steps", 50),
        eval_steps=training_cfg.get("eval_steps", 500),
        save_steps=training_cfg.get("save_steps", 1000),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        evaluation_strategy=evaluation_strategy,
        save_strategy=training_cfg.get("save_strategy", "steps"),
        dataloader_num_workers=data_cfg.get("num_workers", 4),
        remove_unused_columns=False,
        fp16=training_cfg.get("fp16", False),
        label_names=["labels", "label_lengths"],
        report_to=training_cfg.get("report_to", "none"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=resume_from or training_cfg.get("resume_from_checkpoint"))
    trainer.save_model(training_cfg["output_dir"])
    image_processor.save_pretrained(training_cfg["output_dir"])


__all__ = ["run_classification_training", "ViTMAEForCTC"]
