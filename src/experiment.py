"""Configuration and isolated paths for one model run."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils import load_yaml, resolve_path


@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoints: Path
    metrics: Path
    figures: Path
    predictions: Path
    best_checkpoint: Path
    last_checkpoint: Path
    history: Path
    run_id: Path
    evaluations: Path
    per_class: Path

    def create(self) -> None:
        for directory in (
            self.root,
            self.checkpoints,
            self.metrics,
            self.figures,
            self.predictions,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def make_run_paths(config: dict[str, Any], project_root: str | Path) -> RunPaths:
    project_root = Path(project_root).expanduser().resolve()
    run_name = str(config["run"].get("name", "")).strip()
    if not run_name:
        raise ValueError("run.name не должен быть пустым")
    run_root = resolve_path(config["run"]["output_dir"], project_root)
    return RunPaths(
        root=run_root,
        checkpoints=run_root / "checkpoints",
        metrics=run_root / "metrics",
        figures=run_root / "figures",
        predictions=run_root / "predictions",
        best_checkpoint=run_root / "checkpoints" / "best.pt",
        last_checkpoint=run_root / "checkpoints" / "last.pt",
        history=run_root / "metrics" / "training_history.csv",
        run_id=run_root / "mlflow_run_id.txt",
        evaluations=run_root / "metrics" / "evaluation_results.csv",
        per_class=run_root / "metrics" / "per_class_iou.csv",
    )


def load_run(config_path: str | Path) -> tuple[dict[str, Any], Path, RunPaths]:
    config_file = Path(config_path).expanduser().resolve()
    config = load_yaml(config_file)
    for section in ("run", "data", "model", "training", "evaluation", "tracking"):
        if section not in config:
            raise ValueError(f"В конфигурации отсутствует раздел {section}")

    project_root = config_file.parent.parent
    if "project_root" in config:
        project_root = resolve_path(config["project_root"], project_root)
    paths = make_run_paths(config, project_root)
    return config, project_root, paths
