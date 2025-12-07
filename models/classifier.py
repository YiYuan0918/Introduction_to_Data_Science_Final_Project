from dataclasses import dataclass
from typing import Optional
import os

import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    ViTConfig,
    ViTModel,
    ViTMAEConfig,
    ViTMAEModel,
)
from transformers.modeling_outputs import ModelOutput
from transformers.trainer_utils import set_seed
from transformers.utils import logging

from data.dataset import Synth90kDataset, synth90k_collate_fn
from utils.logging_callbacks import CsvLoggingCallback

logger = logging.get_logger(__name__)


@dataclass
class ImageClassificationOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class ViTMAEForImageClassification(PreTrainedModel):
    """Vision Transformer with MAE-pretrained backbone for image classification.

    Uses standard ViT encoder (no masking) for classification. Can load weights
    from MAE-pretrained checkpoints.
    """

    config_class = ViTConfig

    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)
        self.encoder = ViTModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, pixel_values, labels=None, return_dict: Optional[bool] = None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get encoder outputs
        # Use interpolate_pos_encoding=True to handle non-square or non-standard image sizes
        outputs = self.encoder(
            pixel_values=pixel_values,
            interpolate_pos_encoding=True,
            return_dict=return_dict
        )

        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Apply dropout and classification head
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # [batch_size, num_classes]

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return ImageClassificationOutput(loss=loss, logits=logits)


class Synth90kClassificationCollator:
    """
    Collator for image classification training.
    Compatible with the tuple format returned by Synth90kDataset.
    """

    def __call__(self, batch):
        """
        Collate batch of samples from Synth90kDataset.

        Args:
            batch: List of tuples (image, label) or list of images

        Returns:
            dict with:
                - pixel_values: Tensor of shape [batch_size, 3, height, width]
                - labels: Tensor of shape [batch_size] with class indices (0-88171)
        """
        result = synth90k_collate_fn(batch)

        if isinstance(result, tuple):
            # Training case: (images, labels)
            pixel_values, labels = result
            return {
                "pixel_values": pixel_values,
                "labels": labels,
            }
        else:
            # MAE pretraining case: just images
            return {
                "pixel_values": result,
            }


def _load_vit_config_from_mae(
    checkpoint: str,
    num_labels: int,
    img_height: int = 224,
    img_width: int = 224,
) -> ViTConfig:
    """Load ViTConfig from a MAE checkpoint, preserving encoder architecture.

    Args:
        checkpoint: Path to MAE checkpoint or HuggingFace model ID
        num_labels: Number of classification labels (88172 for Synth90k)
        img_height: Target image height
        img_width: Target image width

    Returns:
        ViTConfig with same architecture as MAE but for classification
    """
    # Load MAE config to get architecture hyperparameters
    mae_config = ViTMAEConfig.from_pretrained(checkpoint)

    # ViTConfig.image_size is an int, use max dimension for non-square images
    # Position embeddings will be interpolated to match actual input size
    target_image_size = max(img_height, img_width)

    # Create ViTConfig with same architecture
    config = ViTConfig(
        hidden_size=mae_config.hidden_size,
        num_hidden_layers=mae_config.num_hidden_layers,
        num_attention_heads=mae_config.num_attention_heads,
        intermediate_size=mae_config.intermediate_size,
        hidden_dropout_prob=mae_config.hidden_dropout_prob,
        attention_probs_dropout_prob=mae_config.attention_probs_dropout_prob,
        image_size=target_image_size,
        patch_size=mae_config.patch_size,
        num_channels=mae_config.num_channels,
        layer_norm_eps=mae_config.layer_norm_eps,
        num_labels=num_labels,
    )

    return config


def _build_datasets(data_cfg: dict):
    train_ds = Synth90kDataset(
        root_dir=data_cfg["root"],
        mode="train",
        img_height=data_cfg.get("img_height", 224),
        img_width=data_cfg.get("img_width", 224),
    )
    eval_ds = None
    if data_cfg.get("val_split") is not None:
        eval_ds = Synth90kDataset(
            root_dir=data_cfg["root"],
            mode="val",
            img_height=data_cfg.get("img_height", 224),
            img_width=data_cfg.get("img_width", 224),
        )
    return train_ds, eval_ds


def run_classification_training(cfg: dict, resume_from: Optional[str] = None) -> None:
    seed = cfg.get("seed")
    if seed is not None:
        set_seed(seed)

    vocab_size = Synth90kDataset.NUM_CLASSES  # 88172 word classes

    training_cfg = cfg["training"]["classifier"]
    mae_init = training_cfg.get("mae_checkpoint_for_init") or cfg["model"].get("mae_checkpoint")
    if mae_init is None:
        raise ValueError("mae_checkpoint_for_init or model.mae_checkpoint must be provided")

    data_cfg = cfg["data"]
    img_h = int(data_cfg.get("img_height", 224))
    img_w = int(data_cfg.get("img_width", 224))

    image_processor = AutoImageProcessor.from_pretrained(mae_init)

    # Update image_processor size to match training dimensions
    # This ensures consistency when using the saved processor for inference
    if hasattr(image_processor, "size"):
        image_processor.size = {"height": img_h, "width": img_w}
    if hasattr(image_processor, "crop_size"):
        image_processor.crop_size = {"height": img_h, "width": img_w}

    train_ds, eval_ds = _build_datasets(data_cfg)

    config = _load_vit_config_from_mae(mae_init, vocab_size, img_height=img_h, img_width=img_w)
    model = ViTMAEForImageClassification(config=config)

    # Load MAE encoder weights into ViT encoder
    if mae_init:
        try:
            mae_encoder_state = ViTMAEModel.from_pretrained(mae_init).state_dict()

            # Remove position embeddings if size mismatch (happens with custom dimensions)
            # The model will use its own initialized position embeddings
            # and interpolate them dynamically via interpolate_pos_encoding=True
            if "embeddings.position_embeddings" in mae_encoder_state:
                pretrained_pos_embed_shape = mae_encoder_state["embeddings.position_embeddings"].shape
                current_pos_embed_shape = model.encoder.embeddings.position_embeddings.shape

                if pretrained_pos_embed_shape != current_pos_embed_shape:
                    logger.info(
                        f"Skipping position embeddings due to size mismatch: "
                        f"pretrained {pretrained_pos_embed_shape} vs current {current_pos_embed_shape}. "
                        f"Model will use randomly initialized position embeddings with dynamic interpolation."
                    )
                    del mae_encoder_state["embeddings.position_embeddings"]

            missing, unexpected = model.encoder.load_state_dict(mae_encoder_state, strict=False)

            if unexpected:
                logger.warning("Unexpected keys when loading MAE weights: %s", unexpected)
            if missing:
                # Filter out expected missing keys (pooler and position_embeddings)
                pooler_keys = [k for k in missing if 'pooler' in k]
                pos_embed_keys = [k for k in missing if 'position_embeddings' in k]
                other_missing = [k for k in missing if 'pooler' not in k and 'position_embeddings' not in k]

                if other_missing:
                    logger.warning("Unexpectedly missing encoder keys: %s", other_missing)
                if pooler_keys:
                    logger.info("Missing pooler keys (expected, not used): %s", pooler_keys)
                if pos_embed_keys:
                    logger.info("Missing position embeddings (expected with custom dimensions): %s", pos_embed_keys)

            logger.info("Successfully loaded MAE pretrained weights into ViT encoder")
        except OSError as e:
            logger.warning("Could not load MAE weights from %s, training encoder from scratch. Error: %s", mae_init, e)

    collator = Synth90kClassificationCollator()
    eval_strategy = "steps" if eval_ds else "no"
    load_best = eval_strategy != "no" and bool(training_cfg.get("load_best_model_at_end", True))

    training_args = TrainingArguments(
        output_dir=training_cfg["output_dir"],
        per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(training_cfg.get("per_device_eval_batch_size", 16)),
        num_train_epochs=int(training_cfg.get("num_train_epochs", 1)),
        learning_rate=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        optim=training_cfg.get("optim", "adamw_hf"),
        adam_beta1=float(training_cfg.get("adam_beta1", 0.9)),
        adam_beta2=float(training_cfg.get("adam_beta2", 0.999)),
        adam_epsilon=float(training_cfg.get("adam_epsilon", 1e-8)),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=int(training_cfg.get("warmup_steps", 0)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.0)),
        logging_steps=int(training_cfg.get("logging_steps", 50)),
        eval_steps=int(training_cfg.get("eval_steps", 500)),
        save_steps=int(training_cfg.get("save_steps", 1000)),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
        max_grad_norm=float(training_cfg.get("max_grad_norm", 1.0)),
        eval_strategy=eval_strategy,
        save_strategy=training_cfg.get("save_strategy", "steps"),
        load_best_model_at_end=load_best,
        metric_for_best_model=training_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=bool(training_cfg.get("greater_is_better", False)),
        save_total_limit=training_cfg.get("save_total_limit"),
        dataloader_num_workers=int(data_cfg.get("num_workers", 4)),
        remove_unused_columns=False,
        fp16=bool(training_cfg.get("fp16", False)),
        label_names=["labels"],
        report_to=training_cfg.get("report_to", "none"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[CsvLoggingCallback(training_cfg["output_dir"])],
    )

    trainer.train(resume_from_checkpoint=resume_from or training_cfg.get("resume_from_checkpoint"))

    # Save best model to separate directory if evaluation was enabled
    if eval_ds is not None and load_best:
        best_checkpoint_dir = os.path.join(training_cfg["output_dir"], "best_checkpoint")
        logger.info(f"Saving best model to {best_checkpoint_dir}")
        trainer.save_model(best_checkpoint_dir)
        image_processor.save_pretrained(best_checkpoint_dir)

    # Save final/last model to root output directory
    trainer.save_model(training_cfg["output_dir"])
    image_processor.save_pretrained(training_cfg["output_dir"])


__all__ = ["run_classification_training", "ViTMAEForImageClassification"]
