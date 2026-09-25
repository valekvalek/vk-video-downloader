"""Общие модели данных пайплайна. Все стадии обмениваются только ими (JSON на диске)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

OURS, OPP, REF, UNKNOWN = "ours", "opp", "ref", "unknown"
TEAMS = (OURS, OPP)

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def foot(self) -> tuple[float, float]:
        """Точка опоры игрока в пикселях (середина нижней грани рамки)."""
        return ((self.x1 + self.x2) / 2, self.y2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def iou(self, other: BBox) -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Detection:
    frame: int
    cls: str  # player | goalkeeper | referee | ball
    bbox: BBox
    conf: float = 1.0


@dataclass
class PlayerTrack:
    track_id: int
    team: str = UNKNOWN
    number: int | None = None
    number_conf: float = 0.0
    # (frame, x_m, y_m) в координатах поля; x вдоль длинной стороны 0..105
    points: list[tuple[int, float, float]] = field(default_factory=list)


@dataclass
class Event:
    t: float  # секунды от начала записи
    kind: str  # turnover_loss | turnover_win | shot | pass ...
    team: str
    number: int | None = None
    x: float | None = None
    y: float | None = None
    note: str = ""


@dataclass
class Finding:
    """Найденная проблема: всегда привязана к моменту видео (для клипа)."""

    kind: str
    severity: int  # 1 (мелочь) .. 3 (критично)
    t: float
    team: str
    numbers: list[int] = field(default_factory=list)
    text: str = ""
    clip: tuple[float, float] = (0.0, 0.0)


@dataclass
class MatchData:
    fps: float
    tracks: list[PlayerTrack] = field(default_factory=list)
    ball: list[tuple[int, float, float]] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    attack_dir: dict[str, int] = field(default_factory=lambda: {OURS: 1, OPP: -1})

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> MatchData:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            fps=raw["fps"],
            tracks=[
                PlayerTrack(**{**t, "points": [tuple(p) for p in t["points"]]})
                for t in raw["tracks"]
            ],
            ball=[tuple(b) for b in raw["ball"]],
            events=[Event(**e) for e in raw["events"]],
            attack_dir=raw.get("attack_dir", {OURS: 1, OPP: -1}),
        )
