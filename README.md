# Независимые эксперименты U-Net на Cityscapes

Проект обучает и оценивает один запуск одной модели. Сейчас доступна U-Net и одно
искажение `darkness`. Каждый запуск имеет собственные параметры, checkpoint, CSV,
изображения и MLflow run, поэтому эксперименты не перезаписывают друг друга.

Основной сценарий находится в `notebooks/run_all_colab.ipynb`. Разделы 1–8
подготавливают GPU, Google Drive, Cityscapes, split и проверяют dataset. После них
ноутбук переключает Colab-клон на ветку `codex/unet-modular-pipeline`, а параметры
U-Net задаются вручную.

## Каталог одного запуска

```text
runs/<run_name>/
├── run_config.yaml
├── mlflow_run_id.txt
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── metrics/
│   ├── training_history.csv
│   ├── evaluation_results.csv
│   ├── per_class_iou.csv
│   └── evaluations/<evaluation_id>/
│       ├── summary.csv
│       ├── per_class_iou.csv
│       └── confusion_matrix.csv
└── figures/
    └── segmentation_<condition>_index_<index>.png
```

Новое имя запуска создаёт новый эксперимент. То же имя с `--resume` продолжает
его из `last.pt` и пишет метрики в тот же MLflow run.

## Локальный запуск

Полная пошаговая инструкция приведена в `docs/RUNBOOK.md`.

```powershell
python scripts/create_split.py --config configs/experiment.yaml
python scripts/train_model.py --config configs/experiment.yaml
python scripts/train_model.py --config configs/experiment.yaml --resume
python scripts/visualize_checkpoint.py --config configs/experiment.yaml --index 0
python scripts/evaluate_model.py --config configs/experiment.yaml --condition clean
python scripts/evaluate_model.py --config configs/experiment.yaml --condition darkness --severity 1
```

## MLflow UI

MLflow обязателен: ошибки подключения не скрываются. Перед запуском задайте SQLite
backend и каталог artifacts.

```powershell
$root = (Get-Location).Path.Replace('\', '/')
$env:MLFLOW_TRACKING_URI = "sqlite:///$root/mlflow.db"
$env:MLFLOW_ARTIFACT_ROOT = "file:///$root/mlartifacts"
python -m mlflow server --backend-store-uri $env:MLFLOW_TRACKING_URI
```

Откройте `http://127.0.0.1:5000`. Родительский run содержит обучение и preview,
а каждая clean/darkness-оценка хранится отдельным дочерним run.

## Добавление новой модели

1. Создайте новый файл в `src/models/` с функцией построения модели.
2. Добавьте эту функцию в `MODEL_BUILDERS` в `src/models/__init__.py`.
3. Добавьте отдельный блок параметров модели в `MODEL_SETTINGS` ноутбука.
4. Используйте для каждого обучения новое `RUN_NAME`.

Обучение и оценку изменять не требуется: они получают модель и её параметры из
конфигурации конкретного запуска.
