# Независимые эксперименты сегментации на Cityscapes

Проект обучает и оценивает один запуск одной модели. Сейчас доступны U-Net,
DeepLabV3+ и PSPNet. Для ручной проверки устойчивости доступны `darkness`,
`brightness`, `gaussian_blur`, `gaussian_noise`, `jpeg_compression` и `fog`. Каждый запуск имеет
собственные параметры, checkpoint, CSV, qualitative export и MLflow run, поэтому
эксперименты не перезаписывают друг друга.

Основной сценарий находится в `notebooks/run_all_colab.ipynb`. Разделы 1–8
подготавливают GPU, Google Drive, Cityscapes, split и проверяют dataset. После них
ноутбук переключает Colab-клон на ветку `codex/modular-segmentation-pipeline`. После этого
каждая модель имеет собственную сворачиваемую секцию с параметрами, обучением,
clean-оценкой и отдельными ручными corruption-проверками.

## Каталоги одного запуска

```text
runs/<run_name>/
├── run_config.yaml
├── mlflow_run_id.txt
├── metrics/
│   ├── training_history.csv
│   ├── evaluation_results.csv
│   ├── per_class_iou.csv
│   └── evaluations/<evaluation_id>/
│       ├── summary.csv
│       ├── per_class_iou.csv
│       └── confusion_matrix.csv
└── predictions/qualitative/              # создаётся только явным export
    ├── manifest.csv
    ├── class_schema.json
    └── index_<index>__<image_id>/
        ├── ground_truth_trainid.png
        ├── clean/
        │   ├── input.png
        │   ├── prediction_trainid.png
        │   ├── overlay.png
        │   └── metadata.json
        └── <condition>/severity_<severity>/
            ├── input.png
            ├── prediction_trainid.png
            ├── overlay.png
            └── metadata.json

models/<run_name>/
├── best.pt
└── last.pt
```

`runs/<run_name>/` — лёгкая папка результатов: её удобно скачивать с Google Drive
без весов модели. `models/<run_name>/` хранит только checkpoint-файлы того же run.
MLflow artifacts не хранят `.pt` веса и qualitative PNG. В MLflow остаются
config, history CSV, environment JSON и таблицы оценок. Кривые обучения MLflow
строит из числовых метрик, записанных по эпохам.

Четырёхпанельные segmentation preview и три графика обучения создаются только в
памяти и показываются в Colab. Папка `figures/` для них не создаётся. Постоянные
изображения появляются только после явного qualitative export выбранных индексов.

Новое имя запуска создаёт новый эксперимент. То же имя с `--resume` продолжает
его из `last.pt` после обрыва runtime и пишет метрики в тот же MLflow run.
Дообучение уже завершённой модели делается отдельным новым запуском:
`--continue-from-run <старый_run>`. В новом `run_config.yaml` сохраняются путь к
исходному запуску, папка исходной модели, исходный checkpoint, число добавленных
эпох и общий target epochs.

## Локальный запуск

Полная пошаговая инструкция приведена в `docs/RUNBOOK.md`.

```powershell
python scripts/create_split.py --config configs/experiment.yaml
python scripts/train_model.py --config configs/experiment.yaml
python scripts/train_model.py --config configs/experiment.yaml --resume
python scripts/train_model.py --config configs/experiment.yaml --continue-from-run unet_example --init-checkpoint last
python scripts/visualize_checkpoint.py --config configs/experiment.yaml --index 0
python scripts/evaluate_model.py --config configs/experiment.yaml --replace-existing --condition clean
python scripts/evaluate_model.py --config configs/experiment.yaml --replace-existing --condition darkness --severity 1
python scripts/evaluate_model.py --config configs/experiment.yaml --replace-existing --condition brightness --severity 1
python scripts/evaluate_model.py --config configs/experiment.yaml --replace-existing --condition gaussian_blur --severity 1
python scripts/evaluate_model.py --config configs/experiment.yaml --replace-existing --condition gaussian_noise --severity 1
python scripts/evaluate_model.py --config configs/experiment.yaml --replace-existing --condition jpeg_compression --severity 1
python scripts/evaluate_model.py --config configs/experiment.yaml --replace-existing --condition fog --severity 1
python scripts/export_qualitative.py --config configs/experiment.yaml --indices 0 17 42 --conditions clean darkness brightness gaussian_blur gaussian_noise jpeg_compression fog --severities 1 2 3
```

## Baseline и robust augmentation

Baseline-run использует только resize, horizontal flip, ImageNet normalize и tensor conversion.
Robust-run создаётся отдельным новым `RUN_NAME` и обучается с нуля от ImageNet-весов, без
`--continue-from-run` от baseline. Отличие robust-run от baseline-run — только блок
`augmentation`.

Для robust-run поставьте:

```yaml
run:
  kind: robust
  source_baseline_run: deeplabv3plus_01_ep16_lr0003

augmentation:
  policy: robust
  horizontal_flip_probability: 0.5
  robust_one_of_probability: 0.5
```

При `robust_one_of_probability: 0.5` половина train-изображений остаётся без тяжёлого
искажения, а оставшаяся половина равномерно выбирает один из пяти seen-искажений:
`darkness`, `brightness`, `gaussian_blur`, `gaussian_noise`, `jpeg_compression`.
То есть примерно по 10% train-сэмплов на каждый вид. `fog` остаётся unseen-искажением
для проверки обобщения на evaluation.

`visualize_checkpoint.py` и notebook-preview показывают изображение из official
Cityscapes validation, то есть из той же выборки, на которой считаются
clean/corruption метрики. Preview не создаёт файлы. `export_qualitative.py`
использует тот же `best.pt`, те же искажения и ту же выборку, но сохраняет
компоненты для внешнего генератора дипломных иллюстраций.

## MLflow UI

MLflow обязателен: ошибки подключения не скрываются. Перед запуском задайте SQLite
backend и каталог artifacts.

```powershell
$root = (Get-Location).Path.Replace('\', '/')
$env:MLFLOW_TRACKING_URI = "sqlite:///$root/mlflow.db"
$env:MLFLOW_ARTIFACT_ROOT = "file:///$root/mlartifacts"
python -m mlflow server --backend-store-uri $env:MLFLOW_TRACKING_URI
```

Откройте `http://127.0.0.1:5000`. Родительский run содержит обучение, а каждая
clean/corruption-оценка хранится отдельным дочерним run. Preview и qualitative
export в MLflow не копируются.

## Добавление новой модели

1. Создайте новый файл в `src/models/` с функцией построения модели.
2. Добавьте эту функцию в `MODEL_BUILDERS` в `src/models/__init__.py`.
3. Добавьте отдельный блок параметров модели в `MODEL_SETTINGS` ноутбука.
4. Добавьте отдельную секцию модели в ноутбуке по образцу U-Net, DeepLabV3+ или
   PSPNet.
5. Используйте для каждого обучения новое `RUN_NAME`.

Обучение и оценку изменять не требуется: они получают модель и её параметры из
конфигурации конкретного запуска.
