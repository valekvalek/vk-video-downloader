"""Сквозной пример на синтетических данных — показывает, как стадии соединяются."""
from __future__ import annotations

import numpy as np

from .analytics.errors import line_gaps, losses_in_own_third
from .analytics.metrics import team_shape
from .events.possession import Owner, detect_turnovers
from .report.plan import build_plan, render_markdown
from .schemas import OPP, OURS

FPS = 25.0


def synthetic_owners() -> tuple[list[Owner | None], list[tuple[float, float] | None]]:
    """Наш №6 держит мяч у своих ворот, затем соперник №9 отбирает."""
    owners: list[Owner | None] = []
    ball: list[tuple[float, float] | None] = []
    for i in range(100):
        owners.append(Owner(OURS, 6, 6))
        ball.append((18.0 + i * 0.02, 30.0))
    for _ in range(50):
        owners.append(Owner(OPP, 21, 9))
        ball.append((20.0, 30.0))
    return owners, ball


def run_demo() -> str:
    owners, ball = synthetic_owners()
    events = detect_turnovers(owners, ball, FPS)
    findings = losses_in_own_third(events, attack_dir=1)

    rng = np.random.default_rng(1)
    stretched = np.column_stack([np.linspace(8, 70, 10), rng.uniform(5, 63, 10)])
    shape = team_shape(stretched, attack_dir=1)
    if shape:
        findings += line_gaps([(300.0, shape)], max_gap=25.0)

    return render_markdown(build_plan(findings))
