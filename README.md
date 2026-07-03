# Исследование устойчивости сегментации Cityscapes

Минимальный каркас проекта для сравнения U-Net, DeepLabV3+ и PSPNet на задаче
семантической сегментации. Модели создаются через
`segmentation_models_pytorch`, эксперименты отслеживаются в MLflow.

Обучение, возобновление из checkpoint и визуальная проверка запускаются
обычными Python-скриптами.

## Установка

Рекомендуется Python 3.10 или 3.11. Выполните из корня проекта:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Для Linux, macOS или Google Colab команда активации окружения выглядит так:

```bash
source .venv/bin/activate
```

## Подготовка данных

Распакуйте Cityscapes в `data/cityscapes` или измените пути в
`configs/experiment.yaml`. В каталоге данных ожидаются папки `leftImg8bit` и
`gtFine`.

В Google Colab notebook загружает набор командой
`kagglehub.dataset_download("electraawais/cityscape-dataset")`, автоматически
находит вложенные каталоги `leftImg8bit` и `gtFine`. Если в Kaggle-архиве есть
только маски `*_gtFine_labelIds.png`, они один раз преобразуются в обязательные
19-классовые `*_gtFine_labelTrainIds.png`; кэш сохраняется в Google Drive.

## Последовательность запуска

Все команды выполняются из корня проекта.

```bash
# 1. Создать воспроизводимое внутреннее train/validation-разбиение.
python scripts/create_split.py --config configs/experiment.yaml

# 2. Обучить базовые модели.
python scripts/train_baselines.py --config configs/experiment.yaml

# Продолжить модель из last checkpoint; без него обучение начнётся с epoch 1.
python -u scripts/train_baselines.py --config configs/experiment.yaml --models pspnet --resume

# Сразу проверить best checkpoint на одном internal-dev изображении.
python scripts/visualize_checkpoint.py --config configs/experiment.yaml --model pspnet --index 0

# 3. Оценить модели на чистой официальной validation-выборке.
python scripts/evaluate_clean.py --config configs/experiment.yaml

# Создать corruption_manifest.csv и сетку примеров без сохранения corrupted-копий.
python scripts/create_corruption_assets.py --config configs/experiment.yaml --corruptions configs/corruptions.yaml --split dev

# 4. Оценить устойчивость к искажениям.
python scripts/evaluate_corruptions.py --config configs/experiment.yaml --corruptions configs/corruptions.yaml

# 5. Указать лучшую модель в robust_training.model_name и обучить её с аугментациями.
python scripts/train_robust.py --config configs/experiment.yaml --corruptions configs/corruptions.yaml

# 6. Построить таблицы и рисунки из сохранённых результатов.
python scripts/build_report_assets.py --config configs/experiment.yaml

# 7. Запустить тесты.
python -m pytest tests -q

# 8. Сохранить визуальную проверку восьми dev-примеров.
python scripts/smoke_test_dataset.py --config configs/experiment.yaml --split dev

# 9. Открыть интерфейс MLflow.
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Перед обучением задайте SQLite backend и отдельный каталог artifacts. Legacy
FileStore не используется. Если MLflow недоступен, обучение продолжится, а
CSV и checkpoints сохранятся.

```powershell
$root = (Get-Location).Path.Replace('\', '/')
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
$env:MLFLOW_ARTIFACT_ROOT = "file:///$root/mlartifacts"
python scripts/train_baselines.py --config configs/experiment.yaml --models unet
```

Занести завершённую модель в MLflow после обучения, выполненного без MLflow:

```powershell
python scripts/backfill_mlflow.py --config configs/experiment.yaml --model unet
```

## Результаты

- `checkpoints/` — веса моделей;
- `outputs/metrics/training_history_<model>.csv` — история каждой эпохи;
- `outputs/metrics/training_environment.json` — версии ПО и сведения о GPU;
- `outputs/metrics/clean_summary.csv` — итоговые clean-метрики моделей;
- `outputs/metrics/clean_per_class_iou.csv` — IoU 19 классов;
- `outputs/metrics/confusion_matrix_<model>.csv` — общая confusion matrix;
- `outputs/metrics/resource_summary.csv` — inference time, параметры и GPU memory;
- `outputs/metrics/corruption_manifest.csv` — ссылки на clean-пары, corruption,
  severity и детерминированный SHA256 seed;
- `outputs/metrics/corruption_results.csv` — clean и 24 corruption-условия для
  каждой модели; `delta_miou = clean_miou - corrupted_miou`;
- `outputs/metrics/corruption_per_class.csv` — IoU 19 классов для каждого условия;
- `outputs/metrics/robustness_summary.csv` — агрегаты, robustness rank и выбранная
  лучшая модель;
- `outputs/figures/` — графики и иллюстрации, включая
  `segmentation_preview_<model>.png` с ground truth и prediction;
- `outputs/figures/corruption_examples.png` — clean и три severity для каждого
  из восьми искажений;
- `outputs/figures/robustness_heatmap.png`, `degradation_curves.png`,
  `retention_comparison.png`, `corruption_family_comparison.png` и
  `worst_case_comparison.png` — графики из corruption CSV;
- `outputs/tables/` — итоговые таблицы;
- `outputs/predictions/` — примеры предсказаний;
- `mlflow.db` — SQLite metadata MLflow;
- `mlartifacts/` — artifacts MLflow;
- `outputs/metrics/mlflow_run_id_<model>.txt` — идентификатор MLflow run.

Ноутбук `notebooks/run_all_colab.ipynb` предназначен для последовательного
запуска тех же скриптов в Google Colab без дублирования основной логики.
