# Установка программного окружения

## Что потребуется

- Python 3.12 (каноническая версия курса — 3.12.7).
- Доступ к сети на время установки пакетов и первой загрузки весов моделей либо локальные
  копии пакетов и весов.
- Для ЛР5 — git: пакет segment-anything устанавливается из репозитория по закреплённому коммиту.
- Для интерактивного режима A в ЛР5 — графическая среда и библиотека Tk либо Qt; режиму B
  дополнительные условия не нужны.

## Базовое окружение (ЛР1–ЛР4)

Команды выполняются в корне репозитория. Windows, командная строка:

```
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
pip install -c constraints.txt -r requirements.txt
```

Linux и macOS:

```
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
pip install -c constraints.txt -r requirements.txt
```

`torch` и `torchvision` устанавливаются отдельной командой с индексом CPU-сборок: в общем
индексе PyPI для части платформ по умолчанию предлагается сборка с поддержкой CUDA, которая
существенно больше и для базовых маршрутов не нужна. Файл `constraints.txt` закрепляет версии
всех зависимостей, которые пакеты курса устанавливают сами.

## Пакеты работ ЛР5–ЛР7

```
pip install --default-timeout=120 -c constraints.txt -r requirements-lr5.txt
pip install -c constraints.txt -r requirements-lr6.txt
pip install -c constraints.txt -r requirements-lr7.txt
pip check
```

ЛР7 использует пакеты ЛР6. Пакет segment-anything собирается из исходного кода; перед сборкой
pip загружает сборочные зависимости с PyPI. Ключ `--default-timeout=120` увеличивает допустимое
время ожидания; при повторной ошибке `ReadTimeoutError` команду запускают ещё раз. При отсутствии
надёжного доступа к PyPI используют локальное зеркало пакетов (ключ `--index-url`).

## Проверка

```
python -m unittest discover -s tests -v
```

Версии пакетов в notebook фиксируются служебной ячейкой, описанной в главе 2 пособия.

## Типичные проблемы

| Признак | Причина | Что делать |
|---|---|---|
| Очень большой объём загрузки при установке `torch` | Устанавливается сборка с CUDA | Установить `torch` и `torchvision` командой с индексом CPU-сборок |
| `ReadTimeoutError` при установке пакета ЛР5 | Не успели загрузиться сборочные зависимости | Повторить с `--default-timeout=120` или использовать локальное зеркало |
| Сообщение, что git не найден | segment-anything ставится из репозитория | Установить git и повторить |
| `torch.load` завершается `UnpicklingError` с упоминанием `numpy._core.multiarray.scalar` | В checkpoint записаны числа типов NumPy | Сохранять числа как `float` и `int`; безопасный режим загрузки не отключать |
| `UnicodeDecodeError` при чтении JSON | Файл открыт без указания кодировки | Открывать файлы с `encoding='utf-8'`; функции `to_coco.py` делают это сами |
| `ginput` сразу возвращает пустой список | Неинтерактивный backend Jupyter или Colab | Использовать режим B каркаса `sam_session.py` |
| Неизвестный аргумент постобработки Grounding DINO | Версия transformers отличается от закреплённой | Установить transformers 5.16.1 |

## Google Colab

В Colab пакеты предустановлены, и их версии могут отличаться от закреплённых. Либо установите
закреплённые версии теми же командами (с восклицательным знаком перед `pip`), либо работайте на
предустановленных версиях и укажите их в отчёте. Графической среды в Colab нет: в ЛР5
используется режим B.
