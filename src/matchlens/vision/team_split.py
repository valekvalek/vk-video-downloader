"""Этап 2: разделение игроков на команды по цвету формы (без нейросети, только numpy)."""
from __future__ import annotations

import numpy as np

from ..schemas import OPP, OURS, UNKNOWN, BBox


def torso_crop(frame_rgb: np.ndarray, bbox: BBox) -> np.ndarray:
    """Центральная часть корпуса: меньше травы, рук и ног."""
    h, w = frame_rgb.shape[:2]
    bw, bh = bbox.x2 - bbox.x1, bbox.y2 - bbox.y1
    x1 = int(max(0, bbox.x1 + 0.25 * bw))
    x2 = int(min(w, bbox.x2 - 0.25 * bw))
    y1 = int(max(0, bbox.y1 + 0.15 * bh))
    y2 = int(min(h, bbox.y1 + 0.55 * bh))
    return frame_rgb[y1:y2, x1:x2]


def dominant_color(crop: np.ndarray, min_pixels: int = 20) -> np.ndarray | None:
    """Медианный цвет без «травяных» пикселей. None, если полезных пикселей мало."""
    if crop.size == 0:
        return None
    px = crop.reshape(-1, 3).astype(np.float32)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    grass = (g > r * 1.15) & (g > b * 1.15)
    px = px[~grass]
    if len(px) < min_pixels:
        return None
    return np.median(px, axis=0)


def kmeans(points: np.ndarray, k: int = 2, iters: int = 30, seed: int = 0):
    """Минимальный KMeans (k-means++ инициализация). Возвращает (labels, centers)."""
    rng = np.random.default_rng(seed)
    pts = points.astype(np.float64)
    centers = [pts[rng.integers(len(pts))]]
    for _ in range(1, k):
        d2 = np.min([((pts - c) ** 2).sum(1) for c in centers], axis=0)
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(len(pts), 1 / len(pts))
        centers.append(pts[rng.choice(len(pts), p=probs)])
    centers = np.array(centers)
    labels = np.zeros(len(pts), dtype=int)
    for _ in range(iters):
        dist = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        new_labels = dist.argmin(1)
        for j in range(k):
            if (new_labels == j).any():
                centers[j] = pts[new_labels == j].mean(0)
        if (new_labels == labels).all():
            break
        labels = new_labels
    return labels, centers


def assign_teams(colors: list[np.ndarray | None], our_color: tuple[int, int, int]) -> list[str]:
    """Кластеризует цвета на 2 группы; группа, ближайшая к нашему цвету, = 'ours'."""
    idx = [i for i, c in enumerate(colors) if c is not None]
    result = [UNKNOWN] * len(colors)
    if len(idx) < 2:
        return result
    labels, centers = kmeans(np.array([colors[i] for i in idx]), k=2)
    ours_cluster = int(np.linalg.norm(centers - np.array(our_color), axis=1).argmin())
    for i, lab in zip(idx, labels, strict=True):
        result[i] = OURS if lab == ours_cluster else OPP
    return result
