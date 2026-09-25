"""Этап 6: правила поиска ошибок. Каждая находка привязана к моменту (для клипа) и номерам."""
from __future__ import annotations

from ..schemas import OURS, Event, Finding
from ..schemas import PITCH_LENGTH_M as L
from .metrics import TeamShape


def _clip(t: float, before: float, after: float) -> tuple[float, float]:
    return (max(0.0, t - before), t + after)


def losses_in_own_third(
    events: list[Event],
    attack_dir: int = 1,
    fraction: float = 0.333,
    before: float = 6.0,
    after: float = 4.0,
) -> list[Finding]:
    """Потеря мяча нашей командой в своей трети поля — самая дорогая ошибка."""
    limit = L * fraction
    out: list[Finding] = []
    for e in events:
        if e.kind != "turnover_loss" or e.team != OURS or e.x is None:
            continue
        depth = e.x if attack_dir > 0 else L - e.x
        if depth <= limit:
            who = f"№{e.number}" if e.number is not None else "игрок (номер не определён)"
            out.append(Finding(
                kind="loss_own_third", severity=3, t=e.t, team=OURS,
                numbers=[e.number] if e.number is not None else [],
                text=f"{who} потерял мяч в своей трети ({depth:.0f} м от своих ворот)",
                clip=_clip(e.t, before, after),
            ))
    return out


def line_gaps(
    shapes: list[tuple[float, TeamShape]],
    team: str = OURS,
    max_gap: float = 25.0,
    before: float = 6.0,
    after: float = 4.0,
    min_gap_s: float = 10.0,
) -> list[Finding]:
    """Слишком большой разрыв между линиями защиты и атаки; повторные срабатывания склеиваются.

    shapes: список (t_сек, TeamShape). Находки не чаще, чем раз в min_gap_s.
    """
    out: list[Finding] = []
    last_t = -1e9
    for t, s in shapes:
        gap = s.attack_line - s.defense_line
        if gap > max_gap and t - last_t >= min_gap_s:
            out.append(Finding(
                kind="line_gap", severity=2, t=t, team=team,
                text=f"Команда растянута: {gap:.0f} м между линиями защиты и атаки",
                clip=_clip(t, before, after),
            ))
            last_t = t
    return out
