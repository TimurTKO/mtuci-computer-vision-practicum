# Получение данных и весов

Наборы данных и веса моделей в репозиторий не входят: этого не позволяют условия их
распространения либо объём. Ниже — официальные источники, контрольные суммы и команды
получения. Значения MD5 взяты из torchvision 0.26.0; значения SHA-256 вычислены авторами
при подготовке материалов (ноябрь 2026 г.).

## Fashion-MNIST (ЛР1)

- Лицензия: MIT (Zalando SE).
- Получение: `torchvision.datasets.FashionMNIST(root, train=True/False, download=True)`.
- Размер: около 30 МБ.

## Oxford-IIIT Pet (ЛР2–ЛР5)

- Лицензия: CC BY-SA 4.0. Авторские права на изображения сохраняются за их владельцами.
- Официальный источник (зеркал в torchvision нет):
  - `https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz` — 792 МБ, MD5 `5c4f3ee8e5d25df40f4fd59a7f44e54c`;
  - `https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz` — MD5 `95a8c909bbe2e81eed6a22bccdf3f68f`.
- Получение: `torchvision.datasets.OxfordIIITPet(root, split='trainval'/'test', target_types=('category', 'segmentation'), download=True)`.
- Адрес `www.robots.ox.ac.uk` перенаправляет на `https://thor.robots.ox.ac.uk/pets/`. При проверке сервер отдавал данные со скоростью около 0,13–0,18 МБ/с (архив изображений — 73 мин) и не поддерживал докачку: curl с ключом `-C -` завершался ошибкой 33. Скачайте набор заранее одним сеансом и раздайте группе локальную копию. Если положить архивы в `<root>/oxford-iiit-pet/`, вызов torchvision с `download=True` проверит MD5 и распакует их без повторной загрузки.
- После распаковки: trainval 3680, test 3669 изображений, 37 пород. В `annotations/trimaps` рядом с каждой маской лежит служебный файл macOS `._<имя>.png`; скрипт `trimap_to_classids.py` их пропускает, собственный цикл по маске `*.png` должен делать то же. Разметка `Persian_259` (test) ошибочна: trimap целиком помечен как фон, хотя кошка занимает весь кадр.
- Классы и разбиение курса: `data/pets_classes.txt` и `data/pets_split.json`. Список `train` записан в сбалансированном порядке: любой его префикс содержит породы поровну.

## Pascal VOC 2007 (основа course edition ЛР6 и ЛР7)

- Условия: часть изображений предоставлена Flickr, использовать их нужно по условиям Flickr (страница VOC 2007). Отдельного разрешения на перераспространение нет, поэтому в комплект входят только идентификаторы, правила и скрипт сборки.
- Официальный источник:
  - `https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar` (460 032 000 байт) — MD5 `c52e279531787c972589f7e41ab4ae64`;
  - `https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar` (451 020 800 байт) — MD5 `b6e924de25625d8de591ea690078ad9f`.
  Значения MD5 совпадают с torchvision 0.26.0 (`torchvision/datasets/voc.py`). Старый адрес `http://host.robots.ox.ac.uk/pascal/VOC/voc2007/` перенаправляет на новый (проверено 11.11.2026). Сервер сообщает `Accept-Ranges: bytes`, т. е. докачка возможна.
- Зеркало Ultralytics: `https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtrainval_06-Nov-2007.zip` и `.../VOCtest_06-Nov-2007.zip`. SHA-256 архивов зеркала, загруженных авторами: `VOCtrainval_06-Nov-2007.zip` (445 914 070 байт) — `82a7f58ca92950792ee086875f9238abbbc8434b8f463f6dc0962ff18f2f8642`; `VOCtest_06-Nov-2007.zip` (438 316 827 байт) — `a5b13ab807f50802ac4651daac9d526d0236116be4b3e09626bfc35ca1338d9a`. Аннотации и списки test-части зеркала побайтно совпадают с официальным `VOCtestnoimgs_06-Nov-2007.tar`.
- Сборка редакций: `python data/build_voc_editions.py --voc <путь>/VOCdevkit/VOC2007 --out <каталог> lr6 lr7` после распаковки обоих архивов в один каталог `VOCdevkit/VOC2007`. Для unlabeled pool ЛР7 метки и эталон не создаются; ключ `--teacher-gold` предназначен для преподавателя и записывает эталон пула в отдельный каталог `<каталог>/teacher_only/`, который студентам не выдаётся. Скрипт создаёт COCO JSON, списки и метки YOLO. Состав частей задан в `data/lr6_course_edition/manifest.json` и `data/lr7_course_edition/manifest.json`.

## Checkpoint SAM ViT-B (ЛР5)

- Лицензия: Apache 2.0 (репозиторий facebookresearch/segment-anything).
- `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth` — 375 042 383 байт, SHA-256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`.

## Модели Hugging Face (ЛР7)

Закреплённые ревизии:

| Модель | Лицензия | Ревизия |
|--------|----------|---------|
| `openai/clip-vit-base-patch32` | В карточке не указана; репозиторий openai/CLIP — MIT | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| `facebook/dinov2-small` | Apache 2.0 | `ed25f3a31f01632728cabb09d1542f84ab7b0056` |
| `IDEA-Research/grounding-dino-tiny` | Apache 2.0 | `a2bb814dd30d776dcf7e30523b00659f4f141c71` |

Чтобы закрепить ревизию, передайте её в загрузку: `from_pretrained(name, revision=...)`.
Ревизии проверены по Hugging Face Hub 17.11.2026; загрузка по ним в transformers 5.16.1
проверена в автономном режиме (`HF_HUB_OFFLINE=1`) из заполненного кэша.

Файлы весов в кэше эталонного прогона (SHA-256):

| Модель | Файл | Размер, байт | SHA-256 |
|--------|------|--------------|---------|
| CLIP | `model.safetensors` (служебная ветка `refs/pr/66`, commit `c237dc49a33fc61debc9276459120b7eac67e7ef`) | 605 157 884 | `99d28a652e6ec46629ab7047a0ac82c69b1fe11e0ce672c43af65d3a9a3fc05d` |
| DINOv2-small | `model.safetensors` | 88 249 960 | `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` |
| Grounding DINO tiny | `model.safetensors` | 689 359 096 | `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3` |

В закреплённой ревизии CLIP веса хранятся только как `pytorch_model.bin`; transformers 5.16.1
загружает их копию в формате safetensors из служебной ветки репозитория — это штатное
поведение библиотеки. Для занятий без сети заполните кэш заранее (переменная `HF_HOME`) и
раздайте его группе; при `HF_HUB_OFFLINE=1` модели загружаются только из кэша.

## Веса Ultralytics (ЛР6, ЛР7)

- Лицензия: AGPL-3.0.
- `yolo11n.pt` и `rtdetr-l.pt` загружаются пакетом ultralytics 8.4.33 автоматически при первом вызове `YOLO('yolo11n.pt')` и `RTDETR('rtdetr-l.pt')`.

## Роли изображений ЛР5 и bulk outputs

- `data/lr5_roles.json`: изображение-образец, gold validation (60), gold test (40) и рабочий
  пул (150) из разбиения `data/pets_split.json`.
- Bulk outputs ЛР5 и ЛР7 — предвычисленные результаты тяжёлого инференса — описаны в
  [`docs/BULK_OUTPUTS.md`](docs/BULK_OUTPUTS.md). Embeddings SAM для ЛР5 вычисляются
  сценарием `tools/make_lr5_bulk.py` (около 18 мин на 8-поточном CPU для 250 изображений).
