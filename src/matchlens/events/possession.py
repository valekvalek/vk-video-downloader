"""Этап 5: владение мячом и смена владения (потери и отборы) по позициям в метрах."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schemas import OPP, OURS, Event


@dataclass
class Owner:
    team: str
    track_id: int
    number: int | None = None


def ball_owner(
    ball_xy: tuple[float, float] | None,
    players: list[tuple[int, str, int | None, float, float]],
    max_dist: float = 2.0,
) -> Owner | None:
    """Ближайший к мячу игрок в радиусе max_dist. players: (track_id, team, number, x, y)."""
    if ball_xy is None or not players:
        return None
    bx, by = ball_xy
    best, best_d = None, max_dist
    for track_id, team, number, x, y in players:
        if team not in (OURS, OPP):
            continue
        d = float(np.hypot(x - bx, y - by))
        if d <= best_d:
            best, best_d = Owner(team, track_id, number), d
    return best


def possession_share(owners: list[Owner | None]) -> dict[str, float]:
    """Доля кадров с владением у каждой команды (среди кадров, где владелец известен)."""
    known = [o.team for o in owners if o is not None]
    if not known:
        return {OURS: 0.0, OPP: 0.0}
    return {t: known.count(t) / len(known) for t in (OURS, OPP)}


def detect_turnovers(
    owners: list[Owner | None],
    ball_xy: list[tuple[float, float] | None],
    fps: float,
    min_hold_s: float = 0.5,
) -> list[Event]:
    """События потери/отбора: команда владеет мячом минимум min_hold_s, затем владение у другой.

    Короткие переключения (борьба, рикошет) игнорируются — это подавляет шум трекинга.
    """
    min_hold = max(1, int(round(min_hold_s * fps)))
    events: list[Event] = []
    current: Owner | None = None
    cand: Owner | None = None
    cand_run = 0
    last_owner_frame = 0
    for i, o in enumerate(owners):
        if o is None:
            continue
        if current is None:
            current, cand, cand_run = o, None, 0
            last_owner_frame = i
            continue
        if o.team == current.team:
            current, cand, cand_run = o, None, 0
            last_owner_frame = i
            continue
        if cand is None or cand.team != o.team:
            cand, cand_run = o, 1
        else:
            cand_run += 1
        if cand_run >= min_hold:
            xy = ball_xy[last_owner_frame] or (None, None)
            t = last_owner_frame / fps
            events.append(Event(t, "turnover_loss", current.team, current.number, *xy))
            events.append(Event(t, "turnover_win", cand.team, cand.number, *xy))
            current, cand, cand_run = cand, None, 0
            last_owner_frame = i
    return events
