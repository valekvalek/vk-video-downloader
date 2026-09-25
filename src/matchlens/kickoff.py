"""Поиск начала игры в записи трансляции: до свистка — студия, реклама, разминка.

Эвристика: игра началась там, где в кадре стабильно много игроков. Разминка может дать
ложное срабатывание, поэтому результат сопровождается листом кадров для проверки глазами.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .ingest.prepare import format_ts
from .report.png import contact_sheet, resize_nearest, write_png
from .vision.detect import Detector


def pick_kickoff(counts: list[tuple[float, int]], min_players: int = 12,
                 sustain_s: float = 90.0, ratio: float = 0.8) -> float | None:
    """Первое окно длиной sustain_s, где не менее ratio замеров имеют >= min_players игроков.

    counts: [(время_сек, число_игроков)] по возрастанию времени. Возвращает время начала
    игры (первый «полный» замер окна) или None.
    """
    if len(counts) < 2:
        return None
    steps = sorted(b[0] - a[0] for a, b in zip(counts, counts[1:], strict=False))
    step = steps[len(steps) // 2] or 1.0
    n = max(2, math.ceil(sustain_s / step))
    for i in range(len(counts) - n + 1):
        window = counts[i: i + n]
        good = [t for t, c in window if c >= min_players]
        if len(good) / n >= ratio:
            return good[0]
    return None


def scan_counts(frames, detector: Detector, step_s: float, offset_s: float = 0.0,
                min_conf: float = 0.3, thumb: tuple[int, int] = (160, 90)):
    """Прогон редких кадров: число игроков + миниатюры для листа проверки."""
    counts: list[tuple[float, int]] = []
    thumbs: list[np.ndarray] = []
    for idx, frame in frames:
        dets = detector.detect(frame[:, :, ::-1], idx)
        n = sum(1 for d in dets if d.cls in ("player", "goalkeeper") and d.conf >= min_conf)
        counts.append((offset_s + idx * step_s, n))
        thumbs.append(resize_nearest(frame, *thumb))
    return counts, thumbs


def write_kickoff_report(out_dir: str | Path, kickoff: float | None,
                         counts: list[tuple[float, int]], thumbs: list[np.ndarray],
                         params: dict) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Поиск начала игры", ""]
    if kickoff is None:
        lines.append("Начало игры автоматически не найдено (нет стабильного окна с игроками). "
                     "Задайте время вручную: параметр `start`.")
    else:
        lines.append(f"Игра, вероятно, начинается в **{format_ts(kickoff)[:8]}** "
                     f"(порог {params['min_players']}+ игроков в кадре, окно "
                     f"{params['sustain_s']:.0f} с).")
        lines.append("Проверьте по листу `kickoff_sheet.png`: кадры на -60, -30, 0, +30, +60 с "
                     "вокруг найденного момента. Если там разминка — задайте `start` вручную.")
    lines += ["", "## Число игроков в кадре по времени", "", "| время | игроков |", "| --- | --- |"]
    stride = max(1, len(counts) // 40)
    lines += [f"| {format_ts(t)[:8]} | {c} |" for t, c in counts[::stride]]
    text = "\n".join(lines)
    (out / "kickoff.md").write_text(text, encoding="utf-8")

    if kickoff is not None and thumbs:
        picks = []
        for shift in (-60, -30, 0, 30, 60):
            j = min(range(len(counts)), key=lambda i: abs(counts[i][0] - (kickoff + shift)))
            picks.append(thumbs[j])
        write_png(out / "kickoff_sheet.png", contact_sheet(picks, cell=(160, 90), cols=5))
    return text
