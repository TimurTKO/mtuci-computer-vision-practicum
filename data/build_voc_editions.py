# -*- coding: utf-8 -*-
"""Сборка course edition ЛР6 и ЛР7 из локальной копии Pascal VOC 2007 по manifest комплекта.

usage: python build_voc_editions.py --voc PATH/VOCdevkit/VOC2007 --out OUTDIR [--teacher-gold] lr6 lr7

Для каждой редакции создаются:
  annotations_<part>.json — COCO (bbox XYWH; difficult → iscrowd = 1, ignore-область COCOeval);
      для unlabeled pool ЛР7 не создаётся: эталона у студента нет; с --teacher-gold эталон
      пула пишется в OUTDIR/teacher_only/ для аудита преподавателем;
  yolo/images/ — жёсткие ссылки на изображения VOC (ultralytics ищет labels/ рядом с images/);
  yolo/labels/ — метки YOLO (difficult в обучающие метки не входят);
  yolo/<part>.txt — списки путей.
Изображения не копируются и не перераспространяются. Состав частей берётся из manifest.json."""
import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse(voc, img_id):
    root = ET.parse(voc / 'Annotations' / (img_id + '.xml')).getroot()
    size = root.find('size')
    w, h = int(size.find('width').text), int(size.find('height').text)
    objs = []
    for o in root.findall('object'):
        bb = o.find('bndbox')
        x1, y1 = float(bb.find('xmin').text) - 1, float(bb.find('ymin').text) - 1
        x2, y2 = float(bb.find('xmax').text), float(bb.find('ymax').text)
        objs.append({'name': o.find('name').text, 'difficult': int(o.find('difficult').text),
                     'xyxy': [x1, y1, x2, y2]})
    return w, h, objs


def build(voc, out_root, name, teacher_gold=False):
    man = json.loads((HERE / ('%s_course_edition' % name) / 'manifest.json')
                     .read_text(encoding='utf-8'))
    classes = man['classes']
    # unlabeled pool (ЛР7): меток YOLO нет; COCO-эталон пишется только для преподавателя
    unlabeled = man.get('unlabeled_parts', [])
    out = out_root / ('%s_course_edition' % name)
    for d in ('images', 'labels'):
        (out / 'yolo' / d).mkdir(parents=True, exist_ok=True)
    for part, ids in man['parts'].items():
        coco = {'info': {'description': '%s course edition (%s)' % (name, part)},
                'licenses': [], 'images': [], 'annotations': [],
                'categories': [{'id': k + 1, 'name': c} for k, c in enumerate(classes)]}
        lines = []
        for k, img_id in enumerate(ids, 1):
            w, h, objs = parse(voc, img_id)
            coco['images'].append({'id': k, 'file_name': img_id + '.jpg', 'width': w,
                                   'height': h})
            ylab = []
            for o in objs:
                if o['name'] not in classes:
                    continue
                x1, y1, x2, y2 = o['xyxy']
                bw, bh = x2 - x1, y2 - y1
                coco['annotations'].append({'id': len(coco['annotations']) + 1, 'image_id': k,
                                            'category_id': classes.index(o['name']) + 1,
                                            'bbox': [x1, y1, bw, bh], 'area': bw * bh,
                                            'iscrowd': o['difficult']})
                if not o['difficult']:
                    ylab.append('%d %.6f %.6f %.6f %.6f' % (
                        classes.index(o['name']), (x1 + bw / 2) / w, (y1 + bh / 2) / h,
                        bw / w, bh / h))
            if part not in unlabeled:
                (out / 'yolo' / 'labels' / (img_id + '.txt')).write_text('\n'.join(ylab))
            link = out / 'yolo' / 'images' / (img_id + '.jpg')
            if not link.exists():
                try:
                    os.link(voc / 'JPEGImages' / (img_id + '.jpg'), link)
                except OSError:
                    shutil.copy2(voc / 'JPEGImages' / (img_id + '.jpg'), link)
            lines.append(str(link.resolve()))
        # ASCII-безопасная запись: pycocotools читает JSON кодировкой системы
        if part not in unlabeled:
            (out / ('annotations_%s.json' % part)).write_text(json.dumps(coco), encoding='utf-8')
        elif teacher_gold:
            (out_root / 'teacher_only').mkdir(parents=True, exist_ok=True)
            (out_root / 'teacher_only' / ('%s_annotations_%s.json' % (name, part))).write_text(
                json.dumps(coco), encoding='utf-8')
        (out / 'yolo' / ('%s.txt' % part)).write_text('\n'.join(lines))
    shutil.copy2(HERE / ('%s_course_edition' % name) / 'manifest.json', out / 'manifest.json')
    print(name, {p: len(v) for p, v in man['parts'].items()}, '->', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--voc', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--teacher-gold', action='store_true',
                    help='записать эталон unlabeled pool в OUTDIR/teacher_only (для аудита '
                         'преподавателем; студентам не выдаётся)')
    ap.add_argument('editions', nargs='+', choices=['lr6', 'lr7'])
    a = ap.parse_args()
    for e in a.editions:
        build(a.voc, a.out, e, a.teacher_gold)
