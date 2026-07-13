"""Create a reproducible group-aware train/dev manifest from Cityscapes train."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import find_cityscapes_pairs, read_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Создать внутреннее train/dev-разбиение Cityscapes."
    )
    parser.add_argument(
        "--config", required=True, help="Путь к YAML-конфигурации данных"
    )
    parser.add_argument(
        "--output",
        help="Путь к CSV; по умолчанию используется data.split_file из YAML",
    )
    parser.add_argument(
        "--skip-mask-validation",
        action="store_true",
        help="Не читать все маски при создании manifest (быстрее, но менее безопасно)",
    )
    return parser.parse_args()


def load_config(config_path: str | Path) -> tuple[dict, Path]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Файл конфигурации не найден: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError("В YAML должен быть словарь data с путями к Cityscapes")
    return config, path


def project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def require_official_train_directory(path_value: str | Path, field: str) -> None:
    """Prevent accidental creation of the development split from official val."""
    parts = [part.lower() for part in Path(path_value).parts]
    if "val" in parts or not parts or parts[-1] != "train":
        raise ValueError(
            f"{field} должен указывать на официальный каталог train, не val: "
            f"{path_value}"
        )


def validate_manifest(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError("Получено пустое разбиение")
    if frame["image_id"].duplicated().any():
        duplicated = frame.loc[frame["image_id"].duplicated(), "image_id"].iloc[0]
        raise ValueError(f"Повторяющийся image_id в manifest: {duplicated}")

    train_ids = set(frame.loc[frame["split"] == "train", "image_id"])
    dev_ids = set(frame.loc[frame["split"] == "dev", "image_id"])
    overlap = train_ids & dev_ids
    if overlap:
        raise ValueError(f"image_id попал и в train, и в dev: {sorted(overlap)[0]}")
    if not train_ids or not dev_ids:
        raise ValueError("Разбиение должно содержать непустые train и dev")

    train_groups = set(
        frame.loc[frame["split"] == "train", "city"].astype(str)
        + "_"
        + frame.loc[frame["split"] == "train", "sequence"].astype(str)
    )
    dev_groups = set(
        frame.loc[frame["split"] == "dev", "city"].astype(str)
        + "_"
        + frame.loc[frame["split"] == "dev", "sequence"].astype(str)
    )
    if train_groups & dev_groups:
        raise ValueError("Одна city/sequence-группа попала и в train, и в dev")


def create_manifest(
    config: dict,
    config_path: Path,
    output_arg: str | None,
    skip_mask_validation: bool = False,
) -> Path:
    data = config["data"]
    required = ["root", "train_images", "train_masks", "split_file"]
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"В data отсутствуют параметры: {missing}")

    require_official_train_directory(data["train_images"], "data.train_images")
    require_official_train_directory(data["train_masks"], "data.train_masks")

    project_root = config_path.parent.parent
    dataset_root = project_path(data["root"], project_root)
    pairs = find_cityscapes_pairs(
        dataset_root, data["train_images"], data["train_masks"]
    )
    frame = pd.DataFrame(pairs)

    dev_size = float(data.get("internal_val_size", 0.2))
    if not 0.0 < dev_size < 1.0:
        raise ValueError("data.internal_val_size должен быть между 0 и 1")
    groups = frame["city"].astype(str) + "_" + frame["sequence"].astype(str)
    if groups.nunique() < 2:
        raise ValueError("Для группового разбиения нужны минимум две city/sequence-группы")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=dev_size,
        random_state=int(config.get("seed", 42)),
    )
    train_indices, dev_indices = next(splitter.split(frame, groups=groups))
    frame["split"] = ""
    frame.loc[train_indices, "split"] = "train"
    frame.loc[dev_indices, "split"] = "dev"
    frame = frame.sort_values(["split", "city", "sequence", "image_id"])
    frame = frame.reset_index(drop=True)
    validate_manifest(frame)

    if not skip_mask_validation:
        print(f"Проверка значений в {len(frame)} масках...")
        for relative_path in frame["mask_path"]:
            read_mask(dataset_root / relative_path)

    output_value = output_arg or data["split_file"]
    output_path = project_path(output_value, project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")

    train_count = int((frame["split"] == "train").sum())
    dev_count = int((frame["split"] == "dev").sum())
    print(f"Manifest сохранён: {output_path}")
    print(f"train: {train_count}, dev: {dev_count}, всего: {len(frame)}")
    print("Пересечение image_id и city/sequence-групп между train/dev отсутствует.")
    return output_path


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    create_manifest(
        config,
        config_path,
        args.output,
        skip_mask_validation=args.skip_mask_validation,
    )


if __name__ == "__main__":
    main()
