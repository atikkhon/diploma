# Пошаговый запуск

Код намеренно не перехватывает исключения. Если шаг выполнен неправильно,
выполнение остановится и Python покажет исходную ошибку.

## Google Colab

1. Откройте `notebooks/run_all_colab.ipynb` и включите GPU.
2. Выполните разделы 1–8 по порядку.
3. Выполните ячейку выбора ветки `codex/unet-modular-pipeline`.
4. Откройте секцию нужной модели: U-Net, DeepLabV3+ или PSPNet.
5. В параметрах выбранной модели задайте `RUN_NAME` и гиперпараметры.
6. Для нового обучения задайте `RESUME_TRAINING = False` и
   `CONTINUE_FROM_RUN = None`.
7. Запустите train/resume/continue ячейку этой модели. Результаты сохранятся в
   `Google Drive/cityscapes_robustness/runs/<RUN_NAME>`, а checkpoint-модели — в
   `Google Drive/cityscapes_robustness/models/<RUN_NAME>`.
8. В clean preview ячейке задайте индекс official validation изображения и посмотрите
   четыре панели сегментации.
9. Выполните clean evaluation.
10. Выберите `DARKNESS_SEVERITY` 1, 2 или 3 и выполните darkness evaluation.
11. Посмотрите result preview на искажённом изображении.
12. Откройте CSV в saved results ячейке или запустите MLflow UI.

## Новая тренировка той же модели

В секции нужной модели задайте другое уникальное `RUN_NAME`, оставьте
`RESUME_TRAINING = False` и снова запустите её train/evaluation ячейки. Старые
результаты не изменятся.

## Resume после обрыва runtime

1. Снова выполните разделы 1–8.
2. Укажите прежний `RUN_NAME`.
3. Задайте `RESUME_TRAINING = True` и `CONTINUE_FROM_RUN = None`.
4. Запустите train/resume/continue ячейку выбранной модели. Будут загружены
   `last.pt`, optimizer, scaler, история CSV и прежний MLflow run ID.
   Существующий `run_config.yaml` не перезаписывается.

Не меняйте модель, encoder или размер изображения внутри незавершённого запуска.
Для других параметров создайте новый `RUN_NAME`.

## Дообучение завершённой модели в новый run

1. В секции нужной модели задайте новый уникальный `RUN_NAME`.
2. Задайте `RESUME_TRAINING = False`.
3. Задайте `CONTINUE_FROM_RUN = 'имя_старого_run'` или абсолютный путь к старому
   каталогу результатов run.
4. Оставьте `INIT_CHECKPOINT = 'last'`, если хотите продолжать обучение честно с
   последней эпохи. Используйте `'best'`, если хотите стартовать с лучшего
   checkpoint старого run.
5. В settings-словаре выбранной модели укажите `training.epochs` — число новых
   эпох, которые надо добавить. Например, если старый run завершился на 8 эпохе
   и тут указать `8`, новый run будет обучаться до target epoch 16.

Новый `run_config.yaml` сохранит `init_from_run`, `init_from_model_dir`,
`init_checkpoint`, `initial_checkpoint_epoch`, `additional_epochs` и общий
`training.epochs`, поэтому архив будет воспроизводимым.

## Где лежат результаты и модели

- `runs/<RUN_NAME>/` содержит только лёгкие результаты: `run_config.yaml`,
  `mlflow_run_id.txt`, CSV, графики, preview и таблицы оценок.
- `models/<RUN_NAME>/` содержит только веса модели: `best.pt` и `last.pt`.

Если нужно скачать с Google Drive только метрики конкретного эксперимента,
скачивайте `runs/<RUN_NAME>/` или `runs/<RUN_NAME>/metrics/`; checkpoint-файлы
останутся отдельно в `models/<RUN_NAME>/`.

## Правильный порядок оценки

1. Сначала выполните clean evaluation.
2. Затем запускайте darkness evaluation с любыми нужными уровнями.
3. Каждый повтор оценки получает новый `evaluation_id`; строки CSV не стираются.

## MLflow UI в Colab

В разделе 16 ноутбука есть четыре простых шага:

1. Запустить сервер MLflow UI. Ячейка сама задаёт SQLite backend, artifact root,
   Colab proxy flags, показывает ссылку и iframe.
2. Проверить соединение через `curl -I http://127.0.0.1:5000`.
3. Повторно открыть ссылку или iframe, если нужно.
4. Остановить сервер через `pkill -f "mlflow server"`.

Лог сервера сохраняется в
`Google Drive/cityscapes_robustness/logs/mlflow_ui_server.log`.

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
- для дообучения выбран старый `RUN_NAME` вместо нового пустого run;
- для resume отсутствует `last.pt`, CSV истории или `mlflow_run_id.txt`;
- darkness запускается раньше clean evaluation;
- не заданы `MLFLOW_TRACKING_URI` или `MLFLOW_ARTIFACT_ROOT`;
- путь Cityscapes не содержит ожидаемые `leftImg8bit` и `gtFine`;
- в Colab не выбран GPU.
