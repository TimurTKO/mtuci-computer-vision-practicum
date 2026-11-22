# Предвычисленные результаты (bulk outputs) для ЛР5 и ЛР7

Bulk outputs — результаты тяжёлого инференса, которые позволяют выполнить содержательную часть
ЛР5 и ЛР7 без собственного GPU. Они не заменяют самостоятельную работу: постановку prompts,
выбор кандидата, коррекцию, анализ scores, policy и экспорт студент выполняет сам. Обязательный
live inference на небольшом числе изображений сохраняется; на изображении-образце его результат
сверяют с bulk output.

Bulk outputs содержат только выходы моделей и метки размеченных частей, которые студент и так
получает (manual train, gold validation). Эталонной разметки unlabeled pool и test-разметки в
них нет.

## Архив `bulk_outputs_v1.0.0.zip`

Архив был подготовлен как отдельное приложение к выпуску v1.0.0, а не как часть Git-репозитория.
В публичный выпуск v1.0.1 он не включается. Маршруты выполнения работ без архива приведены в
`docs/LABS.md`; готовые результаты могут предоставляться преподавателем отдельно при наличии
прав на их распространение.
Для публичного распространения требуется отдельное подтверждение прав на производные
результаты моделей и наборов данных; MIT для кода репозитория к архиву не относится.
Если такое подтверждение не получено, архив не должен оставаться во вложениях общедоступного
Release, включая v1.0.0. Контрольная сумма старого архива указана в описании выпуска;
контрольные суммы его файлов — в `SHA256SUMS` внутри архива.

```
lr5/
    candidates_center_point.npz   кандидаты SAM ViT-B для prompt «точка в центре кадра»
    check_sample.json             сверка live inference и сохранённого embedding на образце
    manifest.json                 SHA-256 файлов, включая embeddings, версии пакетов, время
lr7/
    candidates_val.json           outputs Grounding DINO tiny: gold validation
    candidates_pool.json          то же: unlabeled pool
    candidates_test.json          то же: gold test (открывать после фиксации policy)
    bulk_embeddings_gold_crops.npz embeddings CLIP и DINOv2 gold crops
README.md                         описание форматов
SHA256SUMS
```

## ЛР5

Роли изображений — `data/lr5_roles.json` (250 изображений Oxford-IIIT Pet). Embeddings SAM
(float16, 1×256×64×64, около 0,46 ГБ для 250 изображений) в архив не входят: их вычисляет
сценарий

```
python tools/make_lr5_bulk.py --images <каталог images Oxford-IIIT Pet> ^
    --checkpoint <sam_vit_b_01ec64.pth> --out <каталог>
```

Время — около 18 мин на 8-поточном CPU. Сценарий записывает те же файлы `candidates_*`,
`check_sample.json` и `manifest.json`; контрольные суммы embeddings, вычисленных авторами, есть
в `lr5/manifest.json` архива. На другом процессоре младшие разряды embeddings могут отличаться,
поэтому для проверки используют `check_sample.json`: маски, полученные из сохранённого
embedding, должны совпадать с live inference (у авторов IoU не ниже 0,9988).

Загрузка embedding в `SamPredictor` вместо `set_image` — функция `load_embedding` того же
сценария; она совпадает с листингом пособия. Маски кандидатов упакованы `numpy.packbits` по
последней оси; исходная форма записана в ключе `<имя>/shape`.

## ЛР7

Файлы `candidates_<часть>.json`: модель `IDEA-Research/grounding-dino-tiny` (ревизия
`a2bb814d…`), single-class prompts `person.`, `car.`, `cat.`, `dog.`, `threshold` 0,05,
`text_threshold` 0,25. Поле `candidates` — список записей `image_id`, `box` (XYXY в пикселях
исходного кадра), `score`, `label` (номер класса запроса, начиная с 1); `image_ids` — соответствие
`image_id` и имени файла VOC; `call_seconds` — время вызовов на CPU.

Файл `bulk_embeddings_gold_crops.npz`: L2-нормированные embeddings crops по эталонным boxes —
CLIP ViT-B/32 и DINOv2-small для gold validation (`val_clip`, `val_dino`; 261 crop, включая 30
фоновых) и DINOv2-small для manual train (`train_dino`; по 20 crops на класс, для прототипов),
метки (`val_labels`, `train_labels`: 0..K−1 — классы, K — фоновый crop), координаты crops
(`val_crops`, `train_crops`: `image_id`, x1, y1, x2, y2) и список классов (`classes`).

`candidates_test.json` содержит предсказания Grounding DINO, а не эталонную разметку. Доступность
этих предсказаний не мешает воспроизвести работу, но сама по себе не обеспечивает правило
«открыть test один раз»: студент должен зафиксировать policy до просмотра test-выходов.
Аннотации Pascal VOC 2007 общедоступны, поэтому правило является методическим, а не
техническим ограничением доступа.

Среди кандидатов есть рамки со стороной 0–3 пикселя; crop для verifier строится по политике
листинга «Политика crop кандидата для verifier» пособия.
