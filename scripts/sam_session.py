# -*- coding: utf-8 -*-
"""ЛР5. Каркас интерактивной сессии разметки.

Каркас закрывает только инженерную часть: показ кадра, получение координат,
единый протокол измерения активного времени и журнал взаимодействия.

Содержательная часть ЛР5 в каркас НЕ входит и реализуется студентом:
правило выбора кандидата, stop rule, политика accept/correct/redraw/reject.
Соответствующие функции ниже намеренно оставлены заглушками.

Каноническое окружение: Python 3.12.7, matplotlib 3.9.2.

Два режима, выбор определяется средой:
  Режим A (интерактивный) — plt.ginput, требует GUI-backend (TkAgg или QtAgg).
                            В Jupyter/Colab backend inline НЕ интерактивен:
                            ginput вернёт пустой список, не дожидаясь кликов.
  Режим B (ручной ввод)   — координаты вводятся с клавиатуры, backend не важен.
                            Штатный режим, допустим для всей работы целиком.
"""

import time

import matplotlib
import matplotlib.pyplot as plt

__all__ = [
    'backend_is_interactive', 'probe_interactive', 'collect_points',
    'timed', 'SessionLog', 'choose_candidate', 'should_stop', 'route_result',
]

INTERACTIVE_BACKENDS = ('tkagg', 'qtagg', 'qt5agg', 'macosx', 'gtk3agg', 'gtk4agg', 'wxagg')


# --------------------------------------------------------------------------
# Определение режима
# --------------------------------------------------------------------------
def backend_is_interactive():
    """True, если текущий backend matplotlib поддерживает ginput."""
    return matplotlib.get_backend().lower() in INTERACTIVE_BACKENDS


def probe_interactive(verbose=True):
    """Проверка режима ДО начала сессии. Выполните её один раз в начале работы.

    Для режима A переключите backend до первого построения графика:
        import matplotlib; matplotlib.use('TkAgg')
    """
    backend = matplotlib.get_backend()
    ok = backend_is_interactive()
    if verbose:
        print('backend:', backend)
        print('режим:', 'A (интерактивный, ginput)' if ok else 'B (ручной ввод координат)')
        if not ok:
            print('  ginput в этом backend не дождётся кликов; используется режим B.')
            print("  Для режима A выполните до первого графика: matplotlib.use('TkAgg')")
    return ok


# --------------------------------------------------------------------------
# Сбор точек: единый интерфейс для обоих режимов
# --------------------------------------------------------------------------
def _show(image, overlay, grid, title):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image)
    if overlay is not None:
        ax.imshow(overlay, alpha=0.45)
    ax.set_title(title)
    if grid:
        h, w = image.shape[:2]
        step = max(w, h) // 10 or 1
        ax.set_xticks(range(0, w, step))
        ax.set_yticks(range(0, h, step))
        ax.grid(color='w', linewidth=0.4, alpha=0.6)
    else:
        ax.set_axis_off()
    return fig, ax


def _collect_mode_a(image, overlay, n_max, title):
    """Режим A: клики мышью. ЛКМ — добавить, ПКМ — отменить, СКМ — закончить."""
    fig, _ = _show(image, overlay, grid=False, title=title)
    fig.canvas.draw()
    start = time.perf_counter()
    clicks = plt.ginput(n=n_max, timeout=0, mouse_add=1, mouse_pop=3, mouse_stop=2)
    active_s = time.perf_counter() - start
    plt.close(fig)
    points = [(float(x), float(y)) for x, y in clicks]
    labels = [1] * len(points)          # разметку positive/negative уточняет студент
    return points, labels, active_s


def _collect_mode_b(image, overlay, n_max, title):
    """Режим B: ручной ввод. Формат строки: 'x y label', пустая строка — конец."""
    fig, _ = _show(image, overlay, grid=True, title=title + ' | режим B: введите x y label')
    plt.show(block=False)
    plt.pause(0.1)
    print('Вводите точки по одной: "x y label", где label 1 = positive, 0 = negative.')
    print('Пустая строка — завершить ввод.')
    start = time.perf_counter()
    points, labels = [], []
    while len(points) < n_max:
        raw = input(f'  точка {len(points) + 1}/{n_max}: ').strip()
        if not raw:
            break
        try:
            x, y, lab = raw.split()
            points.append((float(x), float(y)))
            labels.append(int(lab))
        except ValueError:
            print('  формат: x y label (например: 143 208 1)')
    active_s = time.perf_counter() - start
    plt.close(fig)
    return points, labels, active_s


def collect_points(image, overlay=None, n_max=3, title='Укажите точки', mode=None):
    """Показать кадр и получить точки. Возвращает (points_xy, labels, active_s).

    points_xy — список (x, y) в координатах ИСХОДНОГО изображения.
    active_s  — активное время разметчика: от показа кадра до последнего ввода.
                Время вычислений модели (latency) сюда НЕ входит, см. timed().
    """
    if mode is None:
        mode = 'A' if backend_is_interactive() else 'B'
    if mode == 'A':
        return _collect_mode_a(image, overlay, n_max, title)
    return _collect_mode_b(image, overlay, n_max, title)


def collect_box(image, mode=None):
    """Получить box в формате XYXY по двум точкам: левый верхний и правый нижний."""
    pts, _, active_s = collect_points(
        image, n_max=2, title='Два угла box: левый верхний, затем правый нижний', mode=mode)
    if len(pts) != 2:
        return None, active_s
    (x1, y1), (x2, y2) = pts
    box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return box, active_s


# --------------------------------------------------------------------------
# Раздельный учёт latency модели
# --------------------------------------------------------------------------
def timed(fn, *args, **kwargs):
    """Вызвать fn и вернуть (результат, latency_s). Замер отделён от активного времени."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


# --------------------------------------------------------------------------
# Журнал взаимодействия
# --------------------------------------------------------------------------
class SessionLog:
    """Журнал сессии. IoU и прочие метрики присоединяются ТОЛЬКО после сессии."""

    def __init__(self, mode):
        self.mode = mode
        self.rows = []
        self.history = []          # кандидаты по шагам, для возврата к лучшему

    def add(self, image_id, step, prompt_type, selected, active_s, latency_s, decision):
        self.rows.append({
            'image_id': image_id, 'step': step, 'prompt': prompt_type,
            'selected': selected, 'active_s': round(active_s, 3),
            'latency_s': round(latency_s, 3), 'decision': decision,
            'interface_mode': self.mode,
        })

    def push(self, masks, scores, low_res, selected):
        self.history.append({'masks': masks, 'scores': scores,
                             'low_res': low_res, 'selected': selected})

    def revert(self, step_index):
        """Вернуться к состоянию предыдущего шага."""
        return self.history[step_index]

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(self.rows)


# --------------------------------------------------------------------------
# ЗАГЛУШКИ. Содержательная часть ЛР5 — реализуется студентом.
# --------------------------------------------------------------------------
def choose_candidate(masks, quality_scores, observed_features=None):
    """Шаг 4. Правило выбора кандидата БЕЗ доступа к gold.

    Реализуйте выбранное правило: максимум predicted quality score,
    визуальный выбор разметчика либо правило на основе дополнительных признаков.
    Возвращает индекс выбранного кандидата.
    """
    raise NotImplementedError('Шаг 4 ЛР5: правило выбора кандидата реализует студент')


def should_stop(step, history, checklist_result, max_clicks=5):
    """Шаг 5. Stop rule, заданный ДО начала сессии.

    Возвращает True, если коррекцию следует прекратить.
    Запрещено использовать скрытый IoU: правило опирается только на то,
    что доступно разметчику во время сессии.
    """
    raise NotImplementedError('Шаг 5 ЛР5: stop rule задаёт студент')


def route_result(quality_score, observed_features):
    """Шаг 6. Политика маршрутизации.

    Возвращает одно из: 'accept', 'correct', 'redraw', 'reject'.
    Пороги выбираются на gold validation по допустимому риску.
    """
    raise NotImplementedError('Шаг 6 ЛР5: политику маршрутизации задаёт студент')


if __name__ == '__main__':
    probe_interactive()
