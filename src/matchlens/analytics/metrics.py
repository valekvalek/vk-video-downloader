"""Этап 5: метрики по координатам в метрах. Все значения — оценки по видимой части поля."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schemas import PITCH_LENGTH_M as L
from ..schemas import PITCH_WIDTH_M as W


@dataclass
class TeamShape:
    defense_line: float  # средняя глубина 4 самых глубоких полевых (м от своих ворот)
    midfield_line: float
    attack_line: float
    length: float  # расстояние между самым глубоким и самым высоким полевым
    width: float
    hull_area: float


def _depth(x: np.ndarray, attack_dir: int) -> np.ndarray:
    """Глубина «от своих ворот»: растёт в сторону атаки команды."""
    return x if attack_dir > 0 else L - x


def convex_hull_area(points: np.ndarray) -> float:
    pts = np.unique(np.asarray(points, float), axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def team_shape(outfield_xy: np.ndarray, attack_dir: int = 1) -> TeamShape | None:
    """Форма команды по полевым игрокам одного кадра (нужно >= 6 видимых)."""
    pts = np.asarray(outfield_xy, float)
    if len(pts) < 6:
        return None
    order = np.argsort(_depth(pts[:, 0], attack_dir))
    depth = _depth(pts[:, 0], attack_dir)[order]
    n = len(depth)
    k = max(2, n // 3)
    return TeamShape(
        defense_line=float(depth[:k].mean()),
        midfield_line=float(depth[k: n - k].mean()) if n - 2 * k > 0 else float(depth.mean()),
        attack_line=float(depth[n - k:].mean()),
        length=float(depth[-1] - depth[0]),
        width=float(pts[:, 1].max() - pts[:, 1].min()),
        hull_area=convex_hull_area(pts),
    )


def heatmap(points_xy: np.ndarray, bins: tuple[int, int] = (21, 14)) -> np.ndarray:
    """Матрица посещений поля: строки — вдоль ширины, столбцы — вдоль длины."""
    p = np.asarray(points_xy, float).reshape(-1, 2)
    h, _, _ = np.histogram2d(p[:, 1], p[:, 0], bins=(bins[1], bins[0]),
                             range=[[0, W], [0, L]])
    return h


def distance_and_top_speed(points: list[tuple[int, float, float]], fps: float,
                           max_speed_ms: float = 12.0) -> tuple[float, float]:
    """Пройденная дистанция (м) и максимальная скорость (м/с) по видимым отрезкам трека.

    Отрезки с разрывом > 1 кадра и нереальной скоростью (скачок трекера) отбрасываются.
    """
    dist, top = 0.0, 0.0
    for (f0, x0, y0), (f1, x1, y1) in zip(points, points[1:], strict=False):
        if f1 - f0 != 1:
            continue
        step = float(np.hypot(x1 - x0, y1 - y0))
        speed = step * fps
        if speed > max_speed_ms:
            continue
        dist += step
        top = max(top, speed)
    return dist, top
