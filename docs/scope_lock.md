# Текущий объём проекта

- Датасет: Cityscapes, 19 trainId-классов, `ignore_index=255`.
- Модели: U-Net, DeepLabV3+ и PSPNet с настраиваемым encoder.
- Чистая оценка: официальный Cityscapes validation.
- Искажения: `darkness`, `brightness`, `gaussian_blur`, `gaussian_noise`,
  `jpeg_compression` и `fog`; уровень выбирается вручную.
- Метрики: mIoU, Dice, pixel accuracy, IoU классов, confusion matrix,
  время inference, память, delta mIoU и retention.
- Результаты: отдельные checkpoint, CSV, PNG и MLflow runs для каждого запуска.
- Robust retraining и автоматический выбор лучшей архитектуры не входят в проект.

Эта версия измеряет устойчивость вручную выбранной модели к выбранному искажению. Она не
запускает автоматический бенчмарк архитектур и не утверждает, что модель обучена
с robust augmentation.
