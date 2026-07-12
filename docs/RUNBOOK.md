# Пошаговый запуск

Код намеренно не перехватывает исключения. Если шаг выполнен неправильно,
выполнение остановится и Python покажет исходную ошибку.

## Google Colab

1. Откройте `notebooks/run_all_colab.ipynb` и включите GPU.
2. Выполните разделы 1–8 по порядку.
3. Выполните ячейку выбора ветки `codex/modular-segmentation-pipeline`.
4. Откройте секцию нужной модели: U-Net, DeepLabV3+ или PSPNet.
5. В параметрах выбранной модели задайте `RUN_NAME` и гиперпараметры.
6. Для нового обучения задайте `RESUME_TRAINING = False` и
   `CONTINUE_FROM_RUN = None`.
7. Запустите train/resume/continue ячейку этой модели. Результаты сохранятся в
   `Google Drive/cityscapes_robustness/runs/<RUN_NAME>`, а checkpoint-модели — в
   `Google Drive/cityscapes_robustness/models/<RUN_NAME>`.
8. Выполните clean evaluation и задайте список индексов для clean-preview.
9. Выберите severity 1, 2 или 3 в нужном corruption-блоке и выполните evaluation.
10. Посмотрите preview выбранных индексов на искажённом изображении.
11. Откройте CSV и временные графики обучения в saved results ячейке или
    запустите MLflow UI.
12. После отбора сцен при необходимости выполните отдельный qualitative export.

## Новая тренировка той же модели

В секции нужной модели задайте другое уникальное `RUN_NAME`, оставьте
`RESUME_TRAINING = False` и снова запустите её train/evaluation ячейки. Старые
результаты не изменятся.

## Robust augmentation run

Robust-run делайте как отдельную новую тренировку с нуля, а не как дообучение baseline.
В секции нужной модели:

1. задайте новый `RUN_NAME`, например `deeplabv3plus_robust_01_ep16_lr0003`;
2. оставьте `RESUME_TRAINING = False`;
3. оставьте `CONTINUE_FROM_RUN = None`;
4. в settings-секции поставьте:

```python
'run': {
    'kind': 'robust',
    'source_baseline_run': 'deeplabv3plus_01_ep16_lr0003',
},
'augmentation': ROBUST_AUGMENTATION,
```

Baseline settings должны оставаться:

```python
'run': {
    'kind': 'baseline',
    'source_baseline_run': None,
},
'augmentation': BASELINE_AUGMENTATION,
```

При `ROBUST_AUGMENTATION` train-transform с вероятностью 0.5 применяет ровно одно
искажение из пяти seen-вариантов: darkness, brightness, gaussian blur, gaussian noise
или JPEG compression. Выбор вида равномерный, поэтому каждое seen-искажение получает
примерно 10% train-сэмплов. Fog в train не используется и остаётся unseen corruption
для оценки.

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

- `runs/<RUN_NAME>/` содержит `run_config.yaml`, `mlflow_run_id.txt`, CSV и
  таблицы оценок. После явного qualitative export там также появляется папка
  `predictions/qualitative/` с выбранными изображениями.
- `models/<RUN_NAME>/` содержит только веса модели: `best.pt` и `last.pt`.
- `mlartifacts/` содержит config, CSV, environment JSON и таблицы оценок. Веса,
  preview, training PNG и qualitative export туда не логируются.

Если нужно скачать с Google Drive только метрики конкретного эксперимента,
скачивайте `runs/<RUN_NAME>/` или `runs/<RUN_NAME>/metrics/`; checkpoint-файлы
останутся отдельно в `models/<RUN_NAME>/`.

## Правильный порядок оценки

1. Сначала выполните clean evaluation.
2. Затем запускайте нужные corruption evaluation с любыми нужными уровнями.
3. Evaluation-ячейки ноутбука запускаются с `--replace-existing`: повтор оценки
   удаляет старые строки CSV для той же пары `condition`/`severity` и записывает
   новую финальную оценку.

Для полного чистого набора одной модели заново выполните 19 оценок: clean и
шесть corruption-эффектов с severity 1, 2 и 3. После этого
`evaluation_results.csv` должен содержать 19 строк, а `per_class_iou.csv` —
361 строку.

В ноутбуке это можно сделать без ручного перебора: сначала выполните обычную
`clean evaluation` ячейку выбранной модели, затем optional-блок
`batch corruption evaluation for all severities`.

```python
BENCHMARK_SEVERITIES = [1, 2, 3]
```

Этот блок запускает только 18 corruption-оценок. Clean-оценка остаётся отдельной
ячейкой выше. Логи пишутся в те же файлы, что и при ручном запуске:
`logs/evaluate_<RUN_NAME>_<condition>_s<severity>.log`.

Если run был создан до добавления новых искажений, после обновления ветки
заново выполните `Model run helper` и ячейку параметров нужной модели с тем же
`RUN_NAME` и `RESUME_TRAINING = True`. Ноутбук допишет в старый
`run_config.yaml` недостающие corruption-настройки, но не будет менять
параметры обучения, модель, encoder, пути и веса.

Если вы изменили значения в `CORRUPTION_CONFIG` и хотите применить их к уже
существующему run, в вызове `prepare_model_run(...)` выбранной модели поставьте:

```python
update_corruption_settings=True
```

После запуска ячейки параметров модели ноутбук перезапишет только блок
`corruptions` в `runs/<RUN_NAME>/run_config.yaml`. Параметры обучения, модель,
encoder, пути и веса не меняются.

## Preview изображений

Для ручного просмотра одного или нескольких изображений в секции нужной модели
задайте список, например:

```python
CLEAN_PREVIEW_INDICES = [17, 42]
```

Затем запускайте нужные preview-ячейки: clean, darkness, brightness, blur,
noise, JPEG или fog. Четырёхпанельные изображения строятся в памяти и только
показываются в Colab. PNG, папка `figures/` и MLflow artifact не создаются.

Если нужны clean preview и все severity 1, 2, 3 для каждого выбранного индекса,
используйте optional-блок `batch preview selected image indices for all
severities`:

```python
BENCHMARK_PREVIEW_INDICES = [17, 42, 108]
BENCHMARK_PREVIEW_SEVERITIES = [1, 2, 3]
```

Для каждого индекса этот блок покажет clean preview и все corruption preview
для всех severity. Результаты остаются только в выводе Colab-ячейки.

Три графика обучения также не сохраняются как PNG. Ячейка сохранённых
результатов каждый раз строит их из `metrics/training_history.csv` и показывает
в Colab. Числовая история продолжает храниться в CSV и отображаться в MLflow.

## Экспорт исходников для дипломных иллюстраций

После отбора окончательных сцен откройте блок `экспорт исходников для дипломных
иллюстраций` выбранной модели:

```python
QUALITATIVE_EXPORT_INDICES = [17, 42, 108, 221, 305, 411]
QUALITATIVE_EXPORT_CONDITIONS = [
    'clean',
    'darkness',
    'brightness',
    'gaussian_blur',
    'gaussian_noise',
    'jpeg_compression',
    'fog',
]
QUALITATIVE_EXPORT_SEVERITIES = [1, 2, 3]
```

Это единственный механизм, сохраняющий изображения. Он создаёт:

```text
runs/<RUN_NAME>/predictions/qualitative/
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
```

`input.png` — фактический RGB-вход модели после resize и corruption, но до
ImageNet-нормализации. Маски сохраняются как raw trainId, поэтому внешний проект
может сам применить Cityscapes-палитру и собрать любые сравнения. Повторный
экспорт той же комбинации `image_id/condition/severity` заменяет строку manifest,
не создавая дубликатов. Эти файлы не копируются в MLflow.

## MLflow UI в Colab

В разделе MLflow UI в конце ноутбука есть четыре простых шага:

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
- corruption evaluation запускается раньше clean evaluation;
- не заданы `MLFLOW_TRACKING_URI` или `MLFLOW_ARTIFACT_ROOT`;
- путь Cityscapes не содержит ожидаемые `leftImg8bit` и `gtFine`;
- в Colab не выбран GPU.
