"""Наблюдения одного куска записи: кто и где в кадре, цвет формы, мяч. Тяжёлая часть (нейросеть).

Результат — небольшой JSON: его можно пересчитывать анализом много раз без повторной детекции.
Видео не сохраняется.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from .vision.detect import Detector
from .vision.team_split import dominant_color, torso_crop


def observe_frames(frames, detector: Detector, fps: float, t0: float,
                   min_conf: float = 0.3, ball_min_conf: float = 0.25) -> dict:
    """frames: (idx, RGB-кадр). t — секунды от начала записи (t0 + idx / fps)."""
    out = []
    size = None
    for idx, frame in frames:
        if size is None:
            size = [int(frame.shape[1]), int(frame.shape[0])]
        players, ball = [], None
        for d in detector.detect(frame[:, :, ::-1], idx):
            if d.cls in ("player", "goalkeeper") and d.conf >= min_conf:
                c = dominant_color(torso_crop(frame, d.bbox))
                col = [int(v) for v in c] if c is not None else [-1, -1, -1]
                b = d.bbox
                players.append([round(b.x1), round(b.y1), round(b.x2), round(b.y2), *col])
            elif d.cls == "ball" and d.conf >= ball_min_conf:
                if ball is None or d.conf > ball[2]:
                    b = d.bbox
                    ball = [round((b.x1 + b.x2) / 2), round((b.y1 + b.y2) / 2), round(d.conf, 2)]
        out.append({"t": round(t0 + idx / fps, 2), "p": players, "b": ball})
    return {"t0": t0, "fps": fps, "size": size or [0, 0], "frames": out}


def save_observations(obs: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obs, f, ensure_ascii=False, separators=(",", ":"))


def load_observations(path: str | Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)
