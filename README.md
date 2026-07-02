# Исследование устойчивости сегментации Cityscapes

Минимальный каркас проекта для сравнения U-Net, DeepLabV3+ и PSPNet на задаче
семантической сегментации. Модели создаются через
`segmentation_models_pytorch`, эксперименты отслеживаются в MLflow.

Сейчас Python-файлы содержат только docstring с описанием назначения. Логику
следует добавлять постепенно, сохраняя простой скриптовый интерфейс.

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

## Последовательность запуска

Все команды выполняются из корня проекта.

```bash
# 1. Создать воспроизводимое внутреннее train/validation-разбиение.
python scripts/create_split.py --config configs/experiment.yaml

# 2. Обучить базовые модели.
python scripts/train_baselines.py --config configs/experiment.yaml

# 3. Оценить модели на чистой официальной validation-выборке.
python scripts/evaluate_clean.py --config configs/experiment.yaml

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
python -m mlflow ui --backend-store-uri ./mlruns
```

При необходимости задайте URI MLflow через переменную окружения перед
обучением. Если MLflow недоступен, обучение продолжится, а история останется в
CSV.

```powershell
$env:MLFLOW_TRACKING_URI = "file:./mlruns"
python scripts/train_baselines.py --config configs/experiment.yaml
```

Продолжить отдельную модель из её `last.pt`:

```powershell
python scripts/train_baselines.py --config configs/experiment.yaml --models unet --resume
```

## Результаты

- `checkpoints/` — веса моделей;
- `outputs/metrics/training_history_<model>.csv` — история каждой эпохи;
- `outputs/metrics/training_environment.json` — версии ПО и сведения о GPU;
- `outputs/figures/` — графики и иллюстрации;
- `outputs/tables/` — итоговые таблицы;
- `outputs/predictions/` — примеры предсказаний;
- `mlruns/` — локальные данные MLflow.

Ноутбук `notebooks/run_all_colab.ipynb` предназначен для последовательного
запуска тех же скриптов в Google Colab без дублирования основной логики.
