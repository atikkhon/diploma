"""Create a corruption reference manifest and one example grid without caching images."""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.corruptions import (  # noqa: E402
    create_corruption_manifest,
    load_corruption_config,
    save_corruption_examples,
)
from src.utils import load_yaml, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--corruptions", default="configs/corruptions.yaml")
    parser.add_argument("--split", choices=["train", "dev", "val"], default="dev")
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def resolve_manifest_file(value: str, dataset_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (dataset_root / path).resolve()


def create_assets(
    experiment_config_path: str | Path,
    corruption_config_path: str | Path,
    split: str,
    sample_index: int,
) -> tuple[Path, Path]:
    experiment_path = Path(experiment_config_path).expanduser().resolve()
    project_root = experiment_path.parent.parent
    experiment = load_yaml(experiment_path)
    corruption_config = load_corruption_config(
        resolve_path(corruption_config_path, project_root)
    )
    data = experiment.get("data", {})
    evaluation = experiment.get("evaluation", {})
    dataset_root = resolve_path(data["root"], project_root)
    clean_manifest_path = resolve_path(data["split_file"], project_root)
    metrics_dir = resolve_path(
        evaluation.get("metrics_dir", "outputs/metrics"), project_root
    )
    figures_dir = resolve_path(
        evaluation.get("figures_dir", "outputs/figures"), project_root
    )

    corruption_manifest_path = create_corruption_manifest(
        clean_manifest=clean_manifest_path,
        output_path=metrics_dir / "corruption_manifest.csv",
        config=corruption_config,
        split=split,
    )
    clean_frame = pd.read_csv(clean_manifest_path)
    split_frame = clean_frame.loc[clean_frame["split"] == split].sort_values(
        "image_id"
    ).reset_index(drop=True)
    if sample_index < 0 or sample_index >= len(split_frame):
        raise IndexError(
            f"sample-index={sample_index} вне диапазона 0..{len(split_frame) - 1}"
        )
    sample = split_frame.iloc[sample_index]
    image_path = resolve_manifest_file(str(sample["image_path"]), dataset_root)
    if not image_path.is_file():
        raise FileNotFoundError(f"Изображение для corruption grid не найдено: {image_path}")
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV не удалось прочитать изображение: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    figure_path = save_corruption_examples(
        image=image_rgb,
        image_id=str(sample["image_id"]),
        output_path=figures_dir / "corruption_examples.png",
        config=corruption_config,
    )
    print(f"Corruption manifest: {corruption_manifest_path}")
    print(f"Corruption examples: {figure_path}")
    print("Искажённые изображения целиком на диск не сохранялись.")
    return corruption_manifest_path, figure_path


def main() -> None:
    args = parse_args()
    try:
        create_assets(
            args.config,
            args.corruptions,
            args.split,
            args.sample_index,
        )
    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        IndexError,
        RuntimeError,
        OSError,
    ) as error:
        raise SystemExit(f"Ошибка создания corruption assets: {error}") from error


if __name__ == "__main__":
    main()
