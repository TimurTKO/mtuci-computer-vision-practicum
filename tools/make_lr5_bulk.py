# -*- coding: utf-8 -*-
"""ЛР5: подготовка bulk outputs — image embeddings SAM ViT-B и кандидатов масок.

Для изображений ролей из data/lr5_roles.json (gold validation, gold test, рабочий пул)
вычисляются image embeddings (float16, форма 1×256×64×64) и кандидаты для автоматического
prompt «точка в центре кадра» (multimask, три маски с predicted IoU). Изображения берутся из
локальной копии Oxford-IIIT Pet (GET_DATA.md), checkpoint — sam_vit_b_01ec64.pth.

Результат в каталоге OUT:
  emb/<имя>.npz                  embedding одного изображения (embedding, original_size,
                                 input_size)
  candidates_center_point.npz    маски (numpy.packbits по последней оси), их формы и scores
  check_sample.json              сверка live inference и сохранённого embedding на образце
  manifest.json                  SHA-256 файлов, время, версии пакетов

Повторный запуск пропускает уже вычисленные embeddings. Время на CPU — около 4–5 с на
изображение (250 изображений — около 18 мин на 8-поточном процессоре).

usage:
  python tools/make_lr5_bulk.py --images <каталог images Oxford-IIIT Pet>
         --checkpoint <sam_vit_b_01ec64.pth> --out <каталог> [--roles data/lr5_roles.json]
         [--limit N]
"""
import argparse
import hashlib
import json
import platform
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def load_embedding(predictor, path):
    """Подставить сохранённый embedding в SamPredictor вместо set_image (листинг пособия)."""
    d = np.load(path)
    predictor.reset_image()
    predictor.features = torch.from_numpy(d['embedding'].astype(np.float32))
    predictor.original_size = tuple(int(v) for v in d['original_size'])
    predictor.input_size = tuple(int(v) for v in d['input_size'])
    predictor.is_image_set = True


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def center_prompt(img):
    h, w = img.shape[:2]
    return np.array([[w / 2, h / 2]], np.float32), np.array([1])


def versions():
    out = {'python': platform.python_version()}
    for p in ('torch', 'numpy', 'pillow', 'segment-anything'):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, type=Path)
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--roles', type=Path, default=ROOT / 'data' / 'lr5_roles.json')
    ap.add_argument('--limit', type=int, default=None, help='обработать только первые N')
    a = ap.parse_args()
    from segment_anything import SamPredictor, sam_model_registry

    roles = json.loads(a.roles.read_text(encoding='utf-8'))
    names = list(dict.fromkeys(roles['gold_val'] + roles['gold_test'] + roles['pool']))
    if a.limit:
        names = names[:a.limit]
    (a.out / 'emb').mkdir(parents=True, exist_ok=True)
    sam = sam_model_registry['vit_b'](checkpoint=str(a.checkpoint)).eval()
    predictor = SamPredictor(sam)
    cand, t_enc, t0 = {}, [], time.perf_counter()
    for n in names:
        f = a.out / 'emb' / (n + '.npz')
        img = np.asarray(Image.open(a.images / (n + '.jpg')).convert('RGB'))
        with torch.inference_mode():
            if not f.exists():
                t1 = time.perf_counter()
                predictor.set_image(img)
                t_enc.append(time.perf_counter() - t1)
                np.savez_compressed(f, embedding=predictor.features.numpy().astype(np.float16),
                                    original_size=np.array(predictor.original_size),
                                    input_size=np.array(predictor.input_size))
            load_embedding(predictor, f)          # кандидаты — из сохранённого файла
            pc, pl = center_prompt(img)
            m, s, _ = predictor.predict(point_coords=pc, point_labels=pl, multimask_output=True)
        cand[n + '/masks_packed'] = np.packbits(m.astype(np.uint8), axis=-1)
        cand[n + '/shape'] = np.array(m.shape)
        cand[n + '/scores'] = s.astype(np.float32)
    np.savez_compressed(a.out / 'candidates_center_point.npz', **cand)
    # сверка на изображении-образце
    n = roles['sample']
    img = np.asarray(Image.open(a.images / (n + '.jpg')).convert('RGB'))
    pc, pl = center_prompt(img)
    with torch.inference_mode():
        predictor.set_image(img)
        m_live, s_live, _ = predictor.predict(point_coords=pc, point_labels=pl,
                                              multimask_output=True)
        load_embedding(predictor, a.out / 'emb' / (n + '.npz'))
        m_bulk, s_bulk, _ = predictor.predict(point_coords=pc, point_labels=pl,
                                              multimask_output=True)
    ious = [float((x & y).sum() / max((x | y).sum(), 1)) for x, y in zip(m_live, m_bulk)]
    check = {'image': n, 'prompt': 'center point, multimask', 'mask_iou_live_vs_bulk': ious,
             'max_abs_score_diff': float(np.abs(s_live - s_bulk).max())}
    (a.out / 'check_sample.json').write_text(json.dumps(check, indent=1), encoding='utf-8')
    files = sorted(p for p in a.out.rglob('*') if p.is_file() and p.name != 'manifest.json')
    man = {'versions': versions(), 'checkpoint': a.checkpoint.name,
           'checkpoint_sha256': sha256(a.checkpoint), 'images': len(names),
           'embedding_dtype': 'float16', 'embedding_shape': [1, 256, 64, 64],
           'encoder_seconds_median': float(np.median(t_enc)) if t_enc else None,
           'total_seconds': round(time.perf_counter() - t0, 1),
           'files': {p.relative_to(a.out).as_posix(): sha256(p) for p in files}}
    (a.out / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1),
                                         encoding='utf-8')
    print(json.dumps(check, indent=1))


if __name__ == '__main__':
    main()
