# -*- coding: utf-8 -*-
"""Контрольные и регрессионные тесты сопровождающего комплекта.

Запуск из корня репозитория:   python -m unittest discover -s tests -v
Тесты не требуют наборов данных и весов моделей. Тесты evaluator требуют pycocotools
(requirements-lr6.txt) и пропускаются, если пакет не установлен.

Регрессионные тесты (класс Regression) воспроизводят дефекты, найденные в предыдущей
версии комплекта, и проверяют их исправление."""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / 'scripts'
CONFIG = SCRIPTS / 'coco_eval_config.json'
sys.path.insert(0, str(SCRIPTS))


def has(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def records(items):
    import numpy as np
    return [{'image_id': r['image_id'], 'boxes': np.array(r['boxes'], np.float32).reshape(-1, 4),
             'scores': np.array(r['scores'], np.float32),
             'labels': np.array(r['labels'], np.int64)} for r in items]


def run_eval(gt, recs):
    import to_coco
    with tempfile.TemporaryDirectory() as d:
        out = to_coco.dump_results(recs, Path(d) / 'res.json')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            m = to_coco.evaluate(gt, out, config_path=CONFIG, verbose=False)
    return m, [str(x.message) for x in w]


class Checksums(unittest.TestCase):
    def test_sha256sums(self):
        sums = ROOT / 'SHA256SUMS'
        if not sums.exists():
            self.skipTest('файл SHA256SUMS не найден')
        bad = []
        for line in sums.read_text(encoding='utf-8').splitlines():
            h, name = line.split(None, 1)
            p = ROOT / name.lstrip('*')
            if hashlib.sha256(p.read_bytes()).hexdigest() != h:
                bad.append(name)
        self.assertEqual(bad, [])


class Trimap(unittest.TestCase):
    def test_conversion_and_check(self):
        import numpy as np
        from PIL import Image
        from trimap_to_classids import check_classid_mask, trimap_to_classids
        t = np.full((6, 6), 2, np.uint8)
        t[1:5, 1:5] = 3
        t[2:4, 2:4] = 1
        r = check_classid_mask(trimap_to_classids(Image.fromarray(t)))
        self.assertEqual([int(v) for v in r['values']], [0, 1, 255])
        self.assertEqual((r['foreground_share'], r['ignore_share']), (0.1111, 0.3333))
        with self.assertRaises(ValueError):                     # повторная перекодировка
            trimap_to_classids(Image.fromarray(np.asarray(trimap_to_classids(
                Image.fromarray(t)))))


@unittest.skipUnless(has('pycocotools'), 'pycocotools не установлен')
class Evaluator(unittest.TestCase):
    def test_toy_example(self):
        import to_coco
        ex = ROOT / 'examples'
        recs = records(to_coco.load_json(ex / 'toy_results_records.json'))
        to_coco.check_records(recs)
        m, _ = run_eval(ex / 'toy_gt.json', recs)
        self.assertEqual(m['AP'], 1.0)
        self.assertEqual(m['AP50'], 1.0)
        self.assertIsNone(m['AP_small'])            # малых объектов нет: не определено
        self.assertEqual(m['maxDets'], [1, 10, 100])
        self.assertEqual(m['n_predictions'], 3)


@unittest.skipUnless(has('pycocotools'), 'pycocotools не установлен')
class Regression(unittest.TestCase):
    """Дефекты предыдущей версии комплекта."""

    def _fake_results(self, xyxy, conf, cls):
        import numpy as np

        class T:
            def __init__(self, a):
                self.a = np.asarray(a)

            def cpu(self):
                return self

            def numpy(self):
                return self.a

        class Boxes:
            def __init__(self):
                self.xyxy, self.conf, self.cls = T(xyxy), T(conf), T(cls)

            def __len__(self):
                return len(self.conf.a)

        class Res:
            boxes = Boxes()
        return [Res()]

    def test_zero_area_boxes_dropped_and_counted(self):
        # ранее: boxes нулевой высоты проходили адаптер, check_records падал AssertionError
        import to_coco
        res = self._fake_results([[10, 0, 50, 0], [10, 10, 50, 40], [5, 5, 5, 9]],
                                 [0.01, 0.9, 0.02], [0, 0, 1])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            recs = to_coco.records_from_ultralytics(res, [7], {0: 1, 1: 2})
        self.assertEqual(recs[0]['boxes'].shape, (1, 4))
        self.assertEqual(recs[0]['n_dropped'], 2)
        self.assertEqual(list(recs[0]['labels']), [1])
        self.assertTrue(any('отброшено 2' in str(x.message) for x in w))
        self.assertTrue(to_coco.check_records(recs))

    def test_check_records_raises_value_error(self):
        import to_coco
        with self.assertRaises(ValueError):
            to_coco.check_records(records([{'image_id': 1, 'boxes': [[10, 10, 10, 20]],
                                            'scores': [0.5], 'labels': [1]}]))

    def test_empty_predictions(self):
        # ранее: IndexError в pycocotools.loadRes на пустом списке
        m, w = run_eval(ROOT / 'examples' / 'toy_gt.json',
                        records([{'image_id': 1, 'boxes': [], 'scores': [], 'labels': []}]))
        self.assertEqual(m['AP'], 0.0)              # объекты есть, верных предсказаний нет
        self.assertEqual(m['AR@100'], 0.0)
        self.assertIsNone(m['AP_small'])            # группа пуста: не определено, не ноль
        self.assertEqual(m['n_predictions'], 0)
        self.assertTrue(any('пустой список' in x for x in w))

    def test_recall_keys(self):
        # ранее: evaluate не возвращал AR, хотя ЛР7 требует recall
        import to_coco
        recs = records(to_coco.load_json(ROOT / 'examples' / 'toy_results_records.json'))
        m, _ = run_eval(ROOT / 'examples' / 'toy_gt.json', recs)
        for k in ('AR@1', 'AR@10', 'AR@100'):
            self.assertIn(k, m)
        self.assertEqual(m['AR@100'], 1.0)
        self.assertEqual(m['AR@1'], 0.5)            # два объекта на одном кадре

    def test_non_ascii_json_in_system_encoding(self):
        # ранее: pycocotools открывал JSON в кодировке системы (cp1251 на Windows)
        import to_coco
        gt = to_coco.load_json(ROOT / 'examples' / 'toy_gt.json')
        gt['categories'][0]['name'] = 'объект'
        recs = records(to_coco.load_json(ROOT / 'examples' / 'toy_results_records.json'))
        with tempfile.TemporaryDirectory() as d:
            g = Path(d) / 'gt.json'
            g.write_text(json.dumps(gt, ensure_ascii=False), encoding='utf-8')
            r = to_coco.dump_results(recs, Path(d) / 'res.json')
            code = ('import sys, json; sys.path.insert(0, %r); import to_coco;'
                    'm = to_coco.evaluate(%r, %r, %r, verbose=False); print("AP", m["AP"])'
                    % (str(SCRIPTS), str(g), str(r), str(CONFIG)))
            env = dict(os.environ)
            env.pop('PYTHONUTF8', None)
            p = subprocess.run([sys.executable, '-X', 'utf8=0', '-c', code],
                               capture_output=True, text=True, encoding='utf-8',
                               errors='replace', env=env)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('AP 1.0', p.stdout)


@unittest.skipUnless(has('matplotlib'), 'matplotlib не установлен')
class SamSessionModeB(unittest.TestCase):
    def test_manual_input(self):
        code = ("import sys, numpy as np; sys.path.insert(0, %r);"
                "import matplotlib; matplotlib.use('Agg');"
                "from sam_session import collect_points;"
                "p, l, t = collect_points(np.zeros((50, 60, 3), np.uint8), mode='B');"
                "print('RESULT', p, l)") % str(SCRIPTS)
        r = subprocess.run([sys.executable, '-c', code], input='12 30 1\nbad\n40 20 0\n\n',
                           capture_output=True, text=True, encoding='utf-8',
                           env=dict(os.environ, PYTHONIOENCODING='utf-8'))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('RESULT [(12.0, 30.0), (40.0, 20.0)] [1, 0]', r.stdout)

    def test_student_stubs_raise(self):
        import sam_session
        for f, args in ((sam_session.choose_candidate, ([], [])),
                        (sam_session.should_stop, (0, [], None)),
                        (sam_session.route_result, (0.9, {}))):
            with self.assertRaises(NotImplementedError):
                f(*args)


class Formats(unittest.TestCase):
    def test_requirement_files_declare_encoding(self):
        # pip 23.2 на Windows читает файлы требований в кодировке системы (cp1251), если
        # в первых двух строках нет объявления кодировки: русские комментарии в UTF-8
        # тогда приводят к UnicodeDecodeError
        for p in sorted(ROOT.glob('*.txt')):
            data = p.read_bytes()
            head = data.split(b'\n')[:2]
            ascii_only = all(b < 128 for b in data)
            declared = any(h.startswith(b'#') and b'coding' in h and b'utf-8' in h for h in head)
            self.assertTrue(ascii_only or declared, p.name)

    def test_examples_are_valid_json(self):
        for p in (ROOT / 'examples').glob('*.json'):
            json.loads(p.read_text(encoding='utf-8'))

    def test_pets_split(self):
        s = json.loads((ROOT / 'data' / 'pets_split.json').read_text(encoding='utf-8'))
        classes = (ROOT / 'data' / 'pets_classes.txt').read_text(encoding='utf-8').split('\n')
        self.assertEqual(s['breeds'], [c for c in classes if c])
        tr = {n for n, _ in s['train']}
        va = {n for n, _ in s['val']}
        te = {n for n, _ in s['test_all']}
        self.assertFalse(tr & va or tr & te or va & te)

    def test_lr7_pool_has_no_labels(self):
        # gold-разметка unlabeled pool не выдаётся студентам
        m = json.loads((ROOT / 'data' / 'lr7_course_edition' / 'manifest.json')
                       .read_text(encoding='utf-8'))
        self.assertEqual(m['unlabeled_parts'], ['pool'])
        names = [p.name for p in ROOT.rglob('*')]
        self.assertNotIn('annotations_pool.json', names)


if __name__ == '__main__':
    unittest.main(verbosity=2)
