# Текущий объём проекта

- Датасет: Cityscapes, 19 trainId-классов, `ignore_index=255`.
- Модели: U-Net, DeepLabV3+ и PSPNet с настраиваемым encoder.
- Чистая оценка: официальный Cityscapes validation.
- Искажения: `darkness`, `brightness`, `gaussian_blur`, `gaussian_noise`,
  `jpeg_compression` и `fog`; уровень выбирается вручную.
- Метрики: mIoU, Dice, pixel accuracy, IoU классов, confusion matrix,
  время inference, память, delta mIoU и retention.
- Результаты: отдельные checkpoint, CSV и qualitative export для каждого запуска.
- Robust-обучение поддерживается как отдельный run с детерминированным планом.
  Автоматический выбор лучшей архитектуры не входит в проект.

Эта версия обучает и оценивает вручную выбранную модель. Она не запускает
автоматический выбор или ранжирование архитектур.
