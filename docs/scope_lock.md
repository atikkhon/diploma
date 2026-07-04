# Текущий объём проекта

- Датасет: Cityscapes, 19 trainId-классов, `ignore_index=255`.
- Модель: U-Net с настраиваемым encoder.
- Чистая оценка: официальный Cityscapes validation.
- Искажение: только `darkness`, уровень выбирается вручную.
- Метрики: mIoU, Dice, pixel accuracy, IoU классов, confusion matrix,
  время inference, память, delta mIoU и retention.
- Результаты: отдельные checkpoint, CSV, PNG и MLflow runs для каждого запуска.
- Robust retraining и автоматический выбор лучшей архитектуры не входят в проект.

Эта версия измеряет устойчивость U-Net к затемнению. Она не сравнивает разные
архитектуры и не утверждает, что модель обучена с robust augmentation.
