# -*- coding: utf-8 -*-
"""ЛР4, ЛР5. Перекодирование trimap Oxford-IIIT Pet в class IDs курса.

Официальные значения trimap: 1 — объект, 2 — фон, 3 — неопределённая граница.
Схема курса:                 0 — фон,   1 — объект, 255 — ignore.

Загрузчик torchvision отдаёт trimap СЫРЫМ одноканальным изображением
со значениями 1/2/3 и никакого перекодирования не выполняет:
    OxfordIIITPet(root=..., split='trainval', target_types='segmentation')

Каноническое окружение: Python 3.12.7, numpy 2.1.1, pillow 10.4.0.

Использование:
    python trimap_to_classids.py <каталог trimaps> <каталог назначения>
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

TRIMAP_TO_CLASSID = {1: 1, 2: 0, 3: 255}
BACKGROUND, OBJECT, IGNORE = 0, 1, 255


def trimap_to_classids(trimap):
    """Перекодировать одну маску. Принимает PIL.Image или np.ndarray, возвращает PIL.Image."""
    src = np.asarray(trimap)
    if src.ndim != 2:
        raise ValueError(
            f'ожидается одноканальная разметка, получено {src.shape}. '
            'Похоже, это цветная palette-визуализация, а не class IDs.')
    unexpected = set(np.unique(src)) - set(TRIMAP_TO_CLASSID)
    if unexpected:
        raise ValueError(f'неожиданные значения trimap: {sorted(unexpected)}')

    dst = np.zeros_like(src, dtype=np.uint8)
    for old, new in TRIMAP_TO_CLASSID.items():
        dst[src == old] = new
    return Image.fromarray(dst, mode='L')


def check_classid_mask(mask):
    """Проверка готовой маски. Используйте её в шаге 1 ЛР4 и при экспорте в ЛР5."""
    arr = np.asarray(mask)
    assert arr.ndim == 2, f'маска должна быть одноканальной, получено {arr.shape}'
    values = set(np.unique(arr))
    assert values.issubset({BACKGROUND, OBJECT, IGNORE}), f'посторонние значения: {sorted(values)}'
    foreground_share = float((arr == OBJECT).mean())
    ignore_share = float((arr == IGNORE).mean())
    return {'shape': arr.shape, 'values': sorted(values),
            'foreground_share': round(foreground_share, 4),
            'ignore_share': round(ignore_share, 4)}


def convert_directory(src_dir, dst_dir, pattern='*.png'):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for path in sorted(src_dir.glob(pattern)):
        if path.name.startswith('.'):
            continue
        out = trimap_to_classids(Image.open(path))
        out.save(dst_dir / path.name)
        n_ok += 1
    return n_ok


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(1)
    total = convert_directory(sys.argv[1], sys.argv[2])
    print(f'перекодировано масок: {total} -> {sys.argv[2]}')
