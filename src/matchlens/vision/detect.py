"""Этап 2: детекция людей и мяча. Реализация подключается лениво (нужен GPU-стек)."""
from __future__ import annotations

from typing import Protocol

from ..schemas import BBox, Detection


class Detector(Protocol):
    def detect(self, frame_bgr, frame_idx: int) -> list[Detection]: ...


class YoloDetector:
    """Обёртка над Ultralytics YOLO (COCO: 0 = person, 32 = sports ball).

    Экспериментально: проверяется на реальном видео на этапе 2 (ROADMAP).
    Лицензия Ultralytics — AGPL; для публичного продукта заменить (RF-DETR и т.п.).
    """

    PERSON, BALL = 0, 32

    def __init__(self, weights: str = "yolo11m.pt", conf: float = 0.25,
                 ball_conf: float = 0.15, device: str | None = None):
        from ultralytics import YOLO  # ленивый импорт

        self.model = YOLO(weights)
        self.conf, self.ball_conf, self.device = conf, ball_conf, device

    def detect(self, frame_bgr, frame_idx: int) -> list[Detection]:
        res = self.model.predict(
            frame_bgr, conf=min(self.conf, self.ball_conf), classes=[self.PERSON, self.BALL],
            device=self.device, verbose=False,
        )[0]
        out: list[Detection] = []
        for box in res.boxes:
            cls_id, conf = int(box.cls[0]), float(box.conf[0])
            is_ball = cls_id == self.BALL
            if conf < (self.ball_conf if is_ball else self.conf):
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            out.append(Detection(frame_idx, "ball" if is_ball else "player",
                                 BBox(x1, y1, x2, y2), conf))
        return out
