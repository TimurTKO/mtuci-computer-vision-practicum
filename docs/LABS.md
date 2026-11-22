# Лабораторные работы ЛР1–ЛР7: материалы и подготовка

Задания, теория, порядок выполнения, критерии проверки и требования к отчёту приведены в
учебном пособии; здесь перечислено, какие материалы репозитория и внешние ресурсы нужны для
каждой работы и как их подготовить. Решений самостоятельных частей работ репозиторий не
содержит. Работы выполняются по порядку: данные, модели и инструменты оценки переходят из одной
работы в следующую.

Общие материалы для всех работ: окружение по `docs/INSTALL.md`, служебная ячейка notebook
(глава 2 пособия), шаблон отчёта `report_template.md`.

## ЛР1. Свёртка и первая свёрточная сеть

- Данные: Fashion-MNIST, загружается torchvision (`GET_DATA.md`).
- Пакеты: базовое окружение (`requirements.txt`).
- Материалы репозитория: пример формата разбиения `examples/split_example.json`.
- Собственные материалы студента: challenge set — реальные фотографии предметов тех же классов.

## ЛР2. Пространство признаков и embeddings

- Данные: Oxford-IIIT Pet (`GET_DATA.md`); 12 пород курса — `data/pets_classes.txt`, разбиение
  по идентификаторам — `data/pets_split.json` (seed 42; validation- и test-запросы, галерея).
- Модель: ResNet18 из torchvision.
- Собственные материалы студента: challenge-ядро — реальные фотографии пород курса.

## ЛР3. Backbone и классификация

- Данные: те же 12 пород и то же разбиение `data/pets_split.json`.
- Модель: ResNet18 или MobileNetV3 из torchvision.
- Checkpoint сохраняется со стандартными типами Python (`float`, `int`) — см. пример пособия.

## ЛР4. Сегментация и разметка

- Данные: изображения и trimap Oxford-IIIT Pet; список `train` в `data/pets_split.json`
  записан в сбалансированном порядке (пояснение — поле `train_order`): любой его префикс
  содержит породы поровну.
- Скрипт: `scripts/trimap_to_classids.py` — перекодирование trimap в class IDs 0/1/255:

  ```
  python scripts/trimap_to_classids.py <каталог trimaps> <каталог назначения>
  ```
- Модель: LR-ASPP MobileNetV3-Large из torchvision.
- Собственные материалы студента: размеченные challenge-кадры; итоговый checkpoint ЛР4 нужен в ЛР5.

## ЛР5. SAM и полуавтоматическая разметка

- Пакет: `requirements-lr5.txt` (segment-anything по закреплённому коммиту).
- Checkpoint: `sam_vit_b_01ec64.pth` (`docs/MODELS.md`).
- Роли изображений: `data/lr5_roles.json` — изображение-образец, gold validation, gold test,
  рабочий пул.
- Каркас сессии разметки: `scripts/sam_session.py`. Проверка режима:

  ```
  python scripts/sam_session.py
  ```
  Режим B (ввод координат с клавиатуры) работает в любой среде, включая Jupyter и Colab.
- Архив bulk outputs в публичный выпуск v1.0.1 не входит. При наличии локальных изображений
  Oxford-IIIT Pet и checkpoint SAM из `GET_DATA.md` студент может вычислить embeddings и
  маски-кандидаты сам:

  ```
  python tools/make_lr5_bulk.py --images <каталог изображений> --checkpoint <sam_vit_b_01ec64.pth> --out <каталог результатов>
  ```

  Для 250 изображений проверенный прогон занял около 18 минут на 8-поточном CPU; нужны также
  место для embeddings (около 0,46 ГБ) и время на загрузку изображений и checkpoint. GPU не
  обязателен. Формат результатов — `docs/BULK_OUTPUTS.md`.
- Формат записи о маске: `examples/mask_record_lr5.json`.
- Downstream-проверка выполняется из собственного checkpoint ЛР4.

## ЛР6. YOLO и RT-DETR

- Пакеты: `requirements-lr6.txt`.
- Данные: course edition на основе Pascal VOC 2007 (person, car, dog). После загрузки VOC 2007
  (`GET_DATA.md`):

  ```
  python data/build_voc_editions.py --voc <путь>/VOCdevkit/VOC2007 --out <каталог> lr6
  ```
  Скрипт создаёт аннотации COCO, метки и списки YOLO по `data/lr6_course_edition/manifest.json`.
- Общий evaluator: `scripts/to_coco.py` и `scripts/coco_eval_config.json`; учебный пример —
  `examples/toy_gt.json`, `examples/toy_results_records.json`.
- Модели: `yolo11n.pt` (дообучается студентом), `rtdetr-l.pt`. В учебном процессе checkpoint
  RT-DETR-L, дообученный на course train, выдаёт преподаватель. Вне учебного процесса его
  получают дообучением `rtdetr-l.pt` на той же train-части по шагу 4 ЛР6; в проверенной
  конфигурации 5 эпох при 480 px заняли около 53 минут на 8-поточном CPU. Время на другом
  компьютере может заметно отличаться; GPU ускорит обучение и полный инференс.

## ЛР7. Foundation models и open-vocabulary разметка

- Пакеты: `requirements-lr6.txt` и `requirements-lr7.txt`.
- Модели: CLIP, DINOv2, Grounding DINO по закреплённым ревизиям (`docs/MODELS.md`); student —
  YOLO11n.
- Данные: course edition на основе Pascal VOC 2007 (person, car, cat, dog):

  ```
  python data/build_voc_editions.py --voc <путь>/VOCdevkit/VOC2007 --out <каталог> lr7
  ```
  Для unlabeled pool разметка не создаётся. Классы и роли данных — `data/lr7_course_edition/ontology.json`.
- Публичный выпуск v1.0.1 не содержит готовых outputs Grounding DINO и embeddings CLIP/DINOv2.
  Их вычисляют по шагам ЛР7 из пособия на изображениях, собранных по manifest: Grounding DINO
  запускают для gold validation и unlabeled pool, а для gold test — только после фиксации policy;
  CLIP и DINOv2 применяют к crops размеченных manual train и gold validation. Ревизии моделей
  закреплены в `docs/MODELS.md`, форматы выходов — в `docs/BULK_OUTPUTS.md`.
- Полного сценария генерации архива ЛР7 в репозитории нет. Для инференса на всех изображениях
  потребуется значительное время; GPU рекомендуется. На CPU можно проверить небольшой объём,
  но время полного прогона заранее не гарантировано. Если у студента нет подходящих ресурсов,
  преподаватель может предоставить подготовленные outputs отдельно после согласования прав.
- Формат записи pseudo-label: `examples/pseudo_label_record_lr7.json`.

## Что доступно только в учебном процессе

Ряд материалов курса предназначен для преподавателя и в репозиторий не входит: значения
эталонных прогонов, которые выдаются студентам как ориентиры по мере выполнения шагов,
эталонные решения самостоятельных частей работ, эталонная разметка unlabeled pool для аудита,
а также обученные модели курса (дообученный checkpoint RT-DETR-L, резервный checkpoint ЛР4).
Вне учебного процесса работы выполнимы с собственными моделями, но без ориентиров эталонного
прогона. Без bulk outputs полный маршрут ЛР7 требует самостоятельного тяжёлого инференса;
наличие только CPU может сделать его практически неудобным.
