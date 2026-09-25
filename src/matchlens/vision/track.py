"""Этап 2: простой трекер по IoU (базовый уровень; позже заменяется на ByteTrack)."""
from __future__ import annotations

from dataclasses import dataclass

from ..schemas import BBox


@dataclass
class _Track:
    track_id: int
    bbox: BBox
    lost: int = 0


class IouTracker:
    """Жадное сопоставление рамок соседних кадров по IoU.

    update() возвращает список (track_id, bbox) для текущего кадра.
    Трек живёт `max_lost` кадров без совпадений, чтобы пережить перекрытия игроков.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 25):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, boxes: list[BBox]) -> list[tuple[int, BBox]]:
        pairs = sorted(
            ((t.bbox.iou(b), ti, bi)
             for ti, t in enumerate(self._tracks) for bi, b in enumerate(boxes)),
            reverse=True,
        )
        used_t: set[int] = set()
        used_b: set[int] = set()
        assigned: dict[int, int] = {}
        for score, ti, bi in pairs:
            if score < self.iou_threshold:
                break
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            assigned[bi] = ti

        for ti, t in enumerate(self._tracks):
            if ti not in used_t:
                t.lost += 1
        result: list[tuple[int, BBox]] = []
        for bi, box in enumerate(boxes):
            if bi in assigned:
                t = self._tracks[assigned[bi]]
                t.bbox, t.lost = box, 0
            else:
                t = _Track(self._next_id, box)
                self._next_id += 1
                self._tracks.append(t)
            result.append((t.track_id, box))
        self._tracks = [t for t in self._tracks if t.lost <= self.max_lost]
        return result
