# -*- coding: utf-8 -*-
"""ЛР6, ЛР7. Адаптер predictions к COCO detection results и общий evaluator.

Обе модели ЛР6 и все варианты ЛР7 приводятся к одному формату и оцениваются одним
evaluator. Метрики встроенных средств валидации пакетов в сравнение не допускаются:
их умолчания по постобработке и усечению могут не совпадать между моделями.

Согласование двух чисел:
  инференс — до max_det = 300 predictions на изображение при conf = 0.001,
             чтобы ранжированный список не обрезался искусственно;
  оценка   — COCOeval с maxDets = [1, 10, 100]; итоговая AP приводится
             при maxDets = 100, evaluator усекает список сам.

Каноническое окружение: Python 3.12.7, ultralytics 8.4.33, pycocotools 2.0.11.
"""

import json
import warnings
from pathlib import Path

import numpy as np

CONF_FOR_AP = 0.001
MAX_DET_FOR_AP = 300

# Ключи результата evaluate() и номер соответствующего элемента COCOeval.stats.
STAT_KEYS = (('AP', 0), ('AP50', 1), ('AP75', 2),
             ('AP_small', 3), ('AP_medium', 4), ('AP_large', 5),
             ('AR@1', 6), ('AR@10', 7), ('AR@100', 8),
             ('AR_small', 9), ('AR_medium', 10), ('AR_large', 11))


# --------------------------------------------------------------------------
# Приведение к каноническому виду
# --------------------------------------------------------------------------
def records_from_ultralytics(results, image_ids, contiguous_to_category):
    """Преобразовать результаты ultralytics в канонические записи.

    results  — то, что вернул model.predict(..., conf=CONF_FOR_AP, max_det=MAX_DET_FOR_AP)
    image_ids — image_id в том же порядке, что и results
    contiguous_to_category — {индекс класса модели: category_id в COCO}

    Boxes приводятся к XYXY в координатах исходного изображения. Boxes нулевой ширины
    или высоты (x2 <= x1 или y2 <= y1) не передаются дальше: при протоколе с низким
    порогом YOLO выдаёт такие low-score предсказания у границы кадра после обрезки
    letterbox-полей. Их число записывается в поле 'n_dropped' каждой записи, а итог
    по вызову сообщается предупреждением.
    """
    records = []
    dropped = 0
    for res, image_id in zip(results, image_ids):
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            records.append({'image_id': image_id,
                            'boxes': np.zeros((0, 4), dtype=np.float32),
                            'scores': np.zeros((0,), dtype=np.float32),
                            'labels': np.zeros((0,), dtype=np.int64),
                            'n_dropped': 0})
            continue
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)      # исходный масштаб
        scores = boxes.conf.cpu().numpy().astype(np.float32)
        cls = boxes.cls.cpu().numpy().astype(np.int64)
        keep = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
        n_bad = int((~keep).sum())
        dropped += n_bad
        labels = np.array([contiguous_to_category[int(c)] for c in cls[keep]],
                          dtype=np.int64).reshape(-1)
        records.append({'image_id': image_id, 'boxes': xyxy[keep].reshape(-1, 4),
                        'scores': scores[keep], 'labels': labels, 'n_dropped': n_bad})
    if dropped:
        warnings.warn('records_from_ultralytics: отброшено %d boxes нулевой площади '
                      '(поле n_dropped записей)' % dropped, stacklevel=2)
    return records


def check_records(records):
    """Проверки, которые должны пройти до передачи в evaluator.

    Ошибка формата — исключение ValueError с описанием первой найденной проблемы.
    """
    for r in records:
        b = np.asarray(r['boxes'])
        n = len(b)
        if b.shape != (n, 4):
            raise ValueError('image_id=%s: boxes формы %s, ожидается (n, 4)'
                             % (r['image_id'], b.shape))
        if len(r['scores']) != n or len(r['labels']) != n:
            raise ValueError('image_id=%s: длины boxes, scores и labels различаются'
                             % r['image_id'])
        if n:
            if not np.isfinite(b).all():
                raise ValueError('image_id=%s: нечисловые координаты' % r['image_id'])
            if not ((b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all()):
                raise ValueError('image_id=%s: нарушен формат XYXY или нулевая площадь'
                                 % r['image_id'])
    return True


def records_to_coco_results(records):
    """Канонические записи -> список COCO detection results.

    Формат COCO: bbox = [x, y, width, height], а не XYXY.
    """
    out = []
    for r in records:
        for (x1, y1, x2, y2), score, label in zip(r['boxes'], r['scores'], r['labels']):
            out.append({
                'image_id': int(r['image_id']),
                'category_id': int(label),
                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                'score': float(score),
            })
    return out


def dump_results(records, path):
    """Записать результаты в JSON (UTF-8; не-ASCII символы экранируются)."""
    path = Path(path)
    path.write_text(json.dumps(records_to_coco_results(records), ensure_ascii=True),
                    encoding='utf-8')
    return path


def load_json(path):
    """Прочитать JSON в кодировке UTF-8 независимо от кодировки системы."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


# --------------------------------------------------------------------------
# Общий evaluator
# --------------------------------------------------------------------------
def _coco_from_dict(dataset):
    from pycocotools.coco import COCO
    import contextlib
    import io
    coco = COCO()
    coco.dataset = dataset
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def evaluate(gt_json, results_json, config_path=None, verbose=True):
    """Прогнать COCOeval с конфигурацией курса.

    Возвращает словарь с ключами AP, AP50, AP75, AP_small, AP_medium, AP_large,
    AR@1, AR@10, AR@100, AR_small, AR_medium, AR_large (все — при maxDets = 100,
    кроме AR@1 и AR@10), maxDets, n_predictions и warnings.

    AP — AP@[.50:.95] при maxDets = 100, итоговое значение при сравнении моделей;
    AR@100 — recall, усреднённый по порогам IoU, при maxDets = 100.

    Значение None означает, что метрика не определена: в группе нет ни одного
    эталонного объекта (например, AP_small при отсутствии малых объектов). Это не
    ноль. Ноль означает, что эталонные объекты есть, а верных предсказаний нет;
    в частности, при пустом списке predictions все определённые метрики равны 0.

    Файлы читаются в UTF-8, поэтому названия категорий могут содержать кириллицу
    и на Windows.
    """
    from pycocotools.cocoeval import COCOeval
    import contextlib
    import io

    gt = load_json(gt_json)
    dets = load_json(results_json)
    if not isinstance(dets, list):
        raise ValueError('файл результатов должен содержать список detection results')
    notes = []
    coco_gt = _coco_from_dict(gt)
    if dets:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(dets)
    else:
        notes.append('пустой список predictions: определённые метрики равны 0')
        coco_dt = _coco_from_dict({'images': gt.get('images', []),
                                   'categories': gt.get('categories', []),
                                   'annotations': []})
    ev = COCOeval(coco_gt, coco_dt, iouType='bbox')

    if config_path:
        p = load_json(config_path)['coco_eval']
        ev.params.iouThrs = np.linspace(p['iou_start'], p['iou_end'],
                                        p['iou_steps'], endpoint=True)
        ev.params.maxDets = p['max_dets']
        ev.params.areaRng = [list(a) for a in p['area_rng']]
        ev.params.areaRngLbl = p['area_rng_lbl']

    out_buf = io.StringIO()
    with contextlib.redirect_stdout(out_buf):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    if verbose:
        print(out_buf.getvalue(), end='')
    res = {}
    for key, i in STAT_KEYS:
        v = float(ev.stats[i])
        res[key] = None if v < 0 else v       # -1 в COCOeval: группа без объектов
    undefined = [k for k, _ in STAT_KEYS if res[k] is None]
    if undefined:
        notes.append('не определены (нет эталонных объектов в группе): '
                     + ', '.join(undefined))
    for n in notes:
        warnings.warn('evaluate: ' + n, stacklevel=2)
    res['maxDets'] = list(ev.params.maxDets)
    res['n_predictions'] = len(dets)
    res['warnings'] = notes
    return res
