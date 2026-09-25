"""Этап 4: пиксели кадра -> метры на поле через гомографию (DLT, только numpy).

Камера движется, поэтому матрица оценивается на каждом кадре по найденным ключевым точкам
поля (углы, штрафная, центр). Поиск точек нейросетью — следующий шаг; здесь математика.
"""
from __future__ import annotations

import numpy as np

from ..schemas import PITCH_LENGTH_M as L
from ..schemas import PITCH_WIDTH_M as W

# Опорные точки стандартного поля 105x68 (x — вдоль длинной стороны, y — вдоль короткой).
KEYPOINTS_M: dict[str, tuple[float, float]] = {
    "corner_tl": (0.0, 0.0),
    "corner_tr": (L, 0.0),
    "corner_bl": (0.0, W),
    "corner_br": (L, W),
    "center": (L / 2, W / 2),
    "halfway_top": (L / 2, 0.0),
    "halfway_bottom": (L / 2, W),
    "penalty_spot_left": (11.0, W / 2),
    "penalty_spot_right": (L - 11.0, W / 2),
    "box_left_top": (0.0, W / 2 - 20.16),
    "box_left_bottom": (0.0, W / 2 + 20.16),
    "box_right_top": (L, W / 2 - 20.16),
    "box_right_bottom": (L, W / 2 + 20.16),
}


def estimate_homography(src_px: np.ndarray, dst_m: np.ndarray) -> np.ndarray:
    """Матрица 3x3 (DLT) по >= 4 соответствиям пиксель -> метры."""
    src, dst = np.asarray(src_px, float), np.asarray(dst_m, float)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("Нужно минимум 4 пары точек одинаковой длины")
    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.array(rows))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def apply_homography(h: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_px, float).reshape(-1, 2)
    homog = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
    return homog[:, :2] / homog[:, 2:3]


def inside_pitch(points_m: np.ndarray, margin: float = 3.0) -> np.ndarray:
    """Маска точек, попадающих на поле (с запасом): отсекает зрителей и запасных."""
    p = np.asarray(points_m, float)
    return (
        (p[:, 0] >= -margin) & (p[:, 0] <= L + margin)
        & (p[:, 1] >= -margin) & (p[:, 1] <= W + margin)
    )
