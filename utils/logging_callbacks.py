import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from transformers import TrainerCallback
from transformers.utils import logging

logger = logging.get_logger(__name__)


class CsvLoggingCallback(TrainerCallback):
    """
    Logs training/evaluation metrics to a CSV file and prints them with controlled precision.

    Precision rules:
    - Any metric containing "loss" is rounded to 4 decimal places.
    - Learning rate metrics ("learning_rate" or "lr") are rounded to 10 decimal places.
    All annotations remain in English as requested.
    """

    def __init__(self, output_dir: str, filename: str = "train_eval_log.csv") -> None:
        self.output_dir = Path(output_dir)
        self.filepath = self.output_dir / filename

    @staticmethod
    def _round_metric(metric: str, value: Any) -> Any:
        """Round selected metrics; fallback to the raw value if casting fails."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return value

        if "loss" in metric:
            return round(numeric, 4)
        if metric in {"learning_rate", "lr"}:
            return round(numeric, 10)
        return numeric

    def _write_rows(self, rows: Iterable[Dict[str, Any]]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        file_exists = self.filepath.exists()
        with self.filepath.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "epoch", "split", "metric", "value"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    def on_log(self, args, state, control, logs: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        if not logs or not state.is_local_process_zero:
            return

        # Apply rounding rules directly to the log dict so downstream outputs also respect precision.
        for key, value in list(logs.items()):
            logs[key] = self._round_metric(key, value)

        split = "eval" if any(key.startswith("eval_") for key in logs.keys()) else "train"
        rows = []
        for metric, value in logs.items():
            metric_name = metric[5:] if split == "eval" and metric.startswith("eval_") else metric
            rows.append(
                {
                    "step": state.global_step,
                    "epoch": round(state.epoch, 4) if state.epoch is not None else None,
                    "split": split,
                    "metric": metric_name,
                    "value": value,
                }
            )

        self._write_rows(rows)

        summary = ", ".join(f"{row['metric']}={row['value']}" for row in rows)
        logger.info("Step %s [%s] %s", state.global_step, split, summary)

    def on_train_end(self, args, state, control, **kwargs) -> None:
        if not state.is_local_process_zero:
            return

        best_metric = getattr(state, "best_metric", None)
        metric_name = getattr(args, "metric_for_best_model", None) or "eval_loss"
        if best_metric is None:
            return

        value = self._round_metric(metric_name, best_metric)
        row = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch is not None else None,
            "split": "best",
            "metric": metric_name,
            "value": value,
        }
        self._write_rows([row])
        logger.info("Best model tracked: %s=%s", metric_name, value)


__all__ = ["CsvLoggingCallback"]
