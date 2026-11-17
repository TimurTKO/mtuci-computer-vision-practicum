# Модели и закреплённые версии

## Пакеты

| Пакет | Версия | Работы |
|---|---|---|
| torch / torchvision | 2.11.0 / 0.26.0 (CPU-сборка) | Все |
| numpy / scipy / scikit-learn | 2.1.1 / 1.14.1 / 1.5.2 | Все |
| matplotlib / pillow / pandas / tqdm | 3.9.2 / 10.4.0 / 2.3.3 / 4.67.1 | Все |
| segment-anything | коммит `dca509fe793f601edb92606367a655c15ac00fdf` | ЛР5 |
| ultralytics / pycocotools | 8.4.33 / 2.0.11 | ЛР6, ЛР7 |
| transformers | 5.16.1 | ЛР7 |

Полный набор версий зависимостей — `constraints.txt`.

## Модели torchvision (ЛР2–ЛР4)

ResNet18 (`IMAGENET1K_V1`), MobileNetV3 и LR-ASPP MobileNetV3-Large с весами ImageNet
загружаются средствами torchvision 0.26.0 при первом обращении.

## Segment Anything (ЛР5)

| Файл | Размер, байт | SHA-256 |
|---|---|---|
| `sam_vit_b_01ec64.pth` | 375 042 383 | `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912` |

Источник: `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`
(ссылка приведена в README репозитория segment-anything).

## Ultralytics (ЛР6, ЛР7)

`yolo11n.pt` и `rtdetr-l.pt` загружаются пакетом ultralytics 8.4.33 автоматически при первом
вызове `YOLO('yolo11n.pt')` и `RTDETR('rtdetr-l.pt')`.

## Модели Hugging Face (ЛР7)

| Модель | Ревизия (commit) |
|---|---|
| `openai/clip-vit-base-patch32` | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| `facebook/dinov2-small` | `ed25f3a31f01632728cabb09d1542f84ab7b0056` |
| `IDEA-Research/grounding-dino-tiny` | `a2bb814dd30d776dcf7e30523b00659f4f141c71` |

Ревизия передаётся в загрузку: `from_pretrained(name, revision=...)`. Контрольные суммы файлов
весов и особенности загрузки CLIP приведены в `GET_DATA.md`. Для работы без сети заполните
кэш заранее (переменная `HF_HOME`) и установите `HF_HUB_OFFLINE=1`.
