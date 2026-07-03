"""Inspect CSV, checkpoints and MLflow run ID before choosing a training action."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_yaml, resolve_path  # noqa: E402


MODELS = ["unet", "deeplabv3plus", "pspnet"]


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def inspect_training_state(config_path: str | Path, model_name: str) -> dict:
    config_file = Path(config_path).expanduser().resolve()
    config = load_yaml(config_file)
    project_root = config_file.parent.parent
    training = config["training"]
    epochs = int(training["epochs"])
    checkpoint_dir = resolve_path(training["checkpoint_dir"], project_root)
    history_dir = resolve_path(
        training.get("history_dir", "outputs/metrics"), project_root
    )

    best_path = checkpoint_dir / f"{model_name}_best.pt"
    last_path = checkpoint_dir / f"{model_name}_last.pt"
    history_path = history_dir / f"training_history_{model_name}.csv"
    run_id_path = history_dir / f"mlflow_run_id_{model_name}.txt"

    last_epoch = None
    checkpoint_model = None
    if last_path.is_file():
        checkpoint = load_checkpoint(last_path)
        last_epoch = int(checkpoint.get("epoch", -1))
        checkpoint_model = checkpoint.get("model_name")

    history_rows = 0
    history_last_epoch = None
    if history_path.is_file():
        history = pd.read_csv(history_path)
        history_rows = len(history)
        if not history.empty and "epoch" in history.columns:
            history_last_epoch = int(history["epoch"].max())

    run_id = None
    if run_id_path.is_file():
        run_id = run_id_path.read_text(encoding="utf-8").strip() or None

    problems = []
    if checkpoint_model not in (None, model_name):
        problems.append(
            f"last checkpoint относится к {checkpoint_model}, а не {model_name}"
        )
    if last_epoch is not None and history_last_epoch is not None:
        if last_epoch != history_last_epoch:
            problems.append(
                f"checkpoint epoch={last_epoch}, CSV epoch={history_last_epoch}"
            )

    completed = (
        last_epoch is not None
        and last_epoch >= epochs
        and history_last_epoch is not None
        and history_last_epoch >= epochs
        and best_path.is_file()
        and not problems
    )
    resumable = (
        last_epoch is not None
        and 0 < last_epoch < epochs
        and history_last_epoch == last_epoch
        and not problems
    )
    if completed:
        state = "completed"
        recommended_action = "skip"
    elif resumable:
        state = "partial"
        recommended_action = "resume"
    elif last_epoch is None and history_rows == 0:
        state = "not_started"
        recommended_action = "fresh"
    else:
        state = "inconsistent"
        recommended_action = "inspect_before_fresh"

    return {
        "model_name": model_name,
        "state": state,
        "recommended_action": recommended_action,
        "target_epochs": epochs,
        "last_checkpoint": str(last_path),
        "last_checkpoint_exists": last_path.is_file(),
        "last_epoch": last_epoch,
        "best_checkpoint": str(best_path),
        "best_checkpoint_exists": best_path.is_file(),
        "history_csv": str(history_path),
        "history_rows": history_rows,
        "history_last_epoch": history_last_epoch,
        "mlflow_run_id_file": str(run_id_path),
        "mlflow_run_id": run_id,
        "problems": problems,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--json", action="store_true", help="Вывести только JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        state = inspect_training_state(args.config, args.model)
    except (FileNotFoundError, ValueError, OSError) as error:
        raise SystemExit(f"Ошибка инспектора: {error}") from error
    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return
    print(json.dumps(state, ensure_ascii=False, indent=2))
    print(
        f"Рекомендуемое действие для {args.model}: "
        f"{state['recommended_action']}"
    )


if __name__ == "__main__":
    main()
