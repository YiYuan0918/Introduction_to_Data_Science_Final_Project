import argparse
import yaml

from models.classifier import run_classification_training
from models.mae import run_mae_pretraining


TASKS = {
    "mae": run_mae_pretraining,
    "classifier": run_classification_training,
}


def parse_args():
    parser = argparse.ArgumentParser(description="HuggingFace training entrypoint")
    parser.add_argument("--config", default="configs/mae.yaml", help="Path to YAML config")
    parser.add_argument(
        "--resume_from", default=None, help="Optional checkpoint path to resume training"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    task = cfg.get("task")
    if task not in TASKS:
        raise ValueError(f"Config must include a valid 'task' key. Expected one of {list(TASKS.keys())}, got: {task}")

    TASKS[task](cfg, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
