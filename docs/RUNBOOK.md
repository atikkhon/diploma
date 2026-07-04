# Пошаговый запуск

Код намеренно не перехватывает исключения. Если шаг выполнен неправильно,
выполнение остановится и Python покажет исходную ошибку.

## Google Colab

1. Откройте `notebooks/run_all_colab.ipynb` и включите GPU.
2. Выполните разделы 1–8 по порядку.
3. Выполните ячейку выбора ветки `codex/unet-modular-pipeline`.
4. В разделе 9 задайте `RUN_NAME` и параметры U-Net.
5. Для нового обучения задайте `RESUME_TRAINING = False`.
6. Запустите раздел 10. Результаты сохранятся в
   `Google Drive/cityscapes_robustness/runs/<RUN_NAME>`.
7. В разделе 11 задайте индекс и посмотрите четыре панели сегментации.
8. Выполните clean evaluation в разделе 12.
9. Выберите `DARKNESS_SEVERITY` 1, 2 или 3 и выполните раздел 13.
10. Посмотрите результат на искажённом изображении в разделе 14.
11. Откройте CSV в разделе 15 или запустите MLflow UI в разделе 16.

## Новая тренировка той же модели

Задайте другое уникальное `RUN_NAME`, оставьте `RESUME_TRAINING = False` и снова
запустите разделы 9–16. Старые результаты не изменятся.

## Resume после обрыва

1. Снова выполните разделы 1–8.
2. Укажите прежний `RUN_NAME`.
3. Задайте `RESUME_TRAINING = True`.
4. Запустите раздел 10. Будут загружены `last.pt`, optimizer, scaler, история CSV
   и прежний MLflow run ID.

Не меняйте модель, encoder или размер изображения внутри незавершённого запуска.
Для других параметров создайте новый `RUN_NAME`.

## Правильный порядок оценки

1. Сначала выполните clean evaluation.
2. Затем запускайте darkness evaluation с любыми нужными уровнями.
3. Каждый повтор оценки получает новый `evaluation_id`; строки CSV не стираются.

## MLflow локально

Из корня проекта:

```powershell
$root = (Get-Location).Path.Replace('\', '/')
$env:MLFLOW_TRACKING_URI = "sqlite:///$root/mlflow.db"
$env:MLFLOW_ARTIFACT_ROOT = "file:///$root/mlartifacts"
python -m mlflow server --backend-store-uri $env:MLFLOW_TRACKING_URI
```

Интерфейс откроется по адресу `http://127.0.0.1:5000`.

## Частые причины остановки

- `run.name` уже содержит результаты, но `--resume` не указан;
- для resume отсутствует `last.pt`, CSV истории или `mlflow_run_id.txt`;
- darkness запускается раньше clean evaluation;
- не заданы `MLFLOW_TRACKING_URI` или `MLFLOW_ARTIFACT_ROOT`;
- путь Cityscapes не содержит ожидаемые `leftImg8bit` и `gtFine`;
- в Colab не выбран GPU.
