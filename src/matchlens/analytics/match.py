"""Анализ матча по наблюдениям (без поля в метрах и без номеров): владение, потери, давление,
растянутость, провалы по окнам времени. Камера ведёт игру, поэтому всё считается в пикселях,
нормированных на рост игрока, и остаётся оценкой по видимой части поля."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..ingest.prepare import format_ts
from ..schemas import OPP, OURS, REF, Finding

DEFAULT_COLORS = {OURS: (25, 75, 170), OPP: (235, 235, 235), REF: (50, 140, 230)}


def _clip(t: float, before: float = 6.0, after: float = 4.0) -> tuple[float, float]:
    return (max(0.0, t - before), t + after)


def cluster_teams(frames: list[dict], colors: dict[str, tuple[int, int, int]] | None = None,
                  iters: int = 20) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """K-means по цветам формы, центры стартуют с цветов из состава (синие, белые, судья)."""
    anchors = dict(DEFAULT_COLORS)
    anchors.update(colors or {})
    roles = [OURS, OPP, REF]
    centers = np.array([anchors[r] for r in roles], dtype=np.float64)
    pts = np.array([p[4:7] for f in frames for p in f["p"] if p[4] >= 0], dtype=np.float64)
    counts = {r: 0 for r in roles}
    if len(pts) < 30:
        return dict(zip(roles, centers, strict=True)), counts
    labels = np.zeros(len(pts), dtype=int)
    for _ in range(iters):
        new = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(2).argmin(1)
        for j in range(3):
            if (new == j).any():
                centers[j] = pts[new == j].mean(0)
        if (new == labels).all():
            break
        labels = new
    counts = {r: int((labels == j).sum()) for j, r in enumerate(roles)}
    return dict(zip(roles, centers, strict=True)), counts


def label_frames(frames: list[dict], centers: dict[str, np.ndarray]) -> list[dict]:
    """Каждому игроку — роль ours/opp/ref по ближайшему центру цвета."""
    roles = list(centers)
    cm = np.array([centers[r] for r in roles])
    out = []
    for f in frames:
        teams: dict[str, list[list[int]]] = {OURS: [], OPP: [], REF: []}
        for p in f["p"]:
            if p[4] < 0:
                continue
            role = roles[int(((cm - np.array(p[4:7])) ** 2).sum(1).argmin())]
            teams[role].append(p[:4])
        out.append({"t": f["t"], "teams": teams, "ball": f["b"]})
    return out


def _foot(b) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2, b[3])


def _h(b) -> float:
    return max(1.0, b[3] - b[1])


def owner_of(frame: dict, max_rel: float = 0.6):
    """Владелец мяча: ближайший игрок, расстояние до мяча не более max_rel его роста.
    Возвращает (команда, bbox) или None."""
    ball = frame["ball"]
    if ball is None:
        return None
    best, best_d = None, max_rel
    for team in (OURS, OPP):
        for b in frame["teams"][team]:
            fx, fy = _foot(b)
            d = math.hypot(ball[0] - fx, ball[1] - (fy - 0.1 * _h(b))) / _h(b)
            if d <= best_d:
                best, best_d = (team, b), d
    return best


def pressure(frame: dict, team: str, b) -> float:
    """Расстояние (в ростах) от владельца до ближайшего соперника; inf, если соперников нет."""
    other = OPP if team == OURS else OURS
    fx, fy = _foot(b)
    ds = [math.hypot(fx - _foot(o)[0], fy - _foot(o)[1]) / _h(b) for o in frame["teams"][other]]
    return min(ds) if ds else math.inf


def spread(frame: dict, team: str) -> float | None:
    """Растянутость команды: разброс по x и y в ростах; None, если видно меньше 3 игроков."""
    bs = frame["teams"][team]
    if len(bs) < 3:
        return None
    hs = float(np.mean([_h(b) for b in bs]))
    xs = [_foot(b)[0] for b in bs]
    ys = [_foot(b)[1] for b in bs]
    return float(math.hypot(np.std(xs), np.std(ys)) / hs)


@dataclass
class Turnover:
    t: float
    loser: str
    pressed_dist: float  # расстояние до ближайшего соперника в момент потери, в ростах


def find_turnovers(frames: list[dict], min_hold: int = 2, max_gap_s: float = 4.0) -> list[Turnover]:
    """Смена владения: новая команда держит мяч не меньше min_hold замеров подряд."""
    out: list[Turnover] = []
    cur = None  # (team, t, pressure_dist)
    cand_team, cand_n = None, 0
    prev_t = None
    for f in frames:
        o = owner_of(f)
        if o is None:
            continue
        team, b = o
        if prev_t is not None and f["t"] - prev_t > max_gap_s:
            cur, cand_team, cand_n = None, None, 0  # разрыв (склейка кусков, повтор, реклама)
        prev_t = f["t"]
        if cur is None or team == cur[0]:
            cur, cand_team, cand_n = (team, f["t"], pressure(f, team, b)), None, 0
            continue
        if cand_team != team:
            cand_team, cand_n = team, 1
        else:
            cand_n += 1
        if cand_n >= min_hold:
            out.append(Turnover(cur[1], cur[0], cur[2]))
            cur, cand_team, cand_n = (team, f["t"], pressure(f, team, b)), None, 0
    return out


@dataclass
class Window:
    start: float
    end: float
    own_frames: int = 0
    ours_frames: int = 0
    pressed_ours: int = 0
    losses_ours: int = 0
    losses_opp: int = 0
    spread_ours: list[float] = field(default_factory=list)
    spread_opp: list[float] = field(default_factory=list)

    @property
    def share(self) -> float | None:
        return self.ours_frames / self.own_frames if self.own_frames else None


def analyse(frames: list[dict], window_s: float = 120.0, press_dist: float = 1.2,
            free_dist: float = 2.0) -> dict:
    """Метрики по всему матчу и по окнам времени + список находок."""
    if not frames:
        return {"frames": 0, "windows": [], "findings": [], "turnovers": []}
    t_start = frames[0]["t"]
    wins: dict[int, Window] = {}

    def win(t: float) -> Window:
        k = int((t - t_start) // window_s)
        if k not in wins:
            wins[k] = Window(t_start + k * window_s, t_start + (k + 1) * window_s)
        return wins[k]

    total_own = ours_own = pressed_ours_frames = 0
    ball_frames = 0
    for f in frames:
        w = win(f["t"])
        ball_frames += f["ball"] is not None
        o = owner_of(f)
        if o is not None:
            team, b = o
            w.own_frames += 1
            total_own += 1
            if team == OURS:
                w.ours_frames += 1
                ours_own += 1
                if pressure(f, OURS, b) <= press_dist:
                    w.pressed_ours += 1
                    pressed_ours_frames += 1
        for team, store in ((OURS, w.spread_ours), (OPP, w.spread_opp)):
            s = spread(f, team)
            if s is not None:
                store.append(s)
    tos = find_turnovers(frames)
    for t in tos:
        w = win(t.t)
        if t.loser == OURS:
            w.losses_ours += 1
        else:
            w.losses_opp += 1
    span = frames[-1]["t"] - t_start
    lost = [t for t in tos if t.loser == OURS]
    won = [t for t in tos if t.loser == OPP]
    pressed_losses = [t for t in lost if t.pressed_dist <= press_dist]
    free_losses = [t for t in lost if t.pressed_dist >= free_dist]
    summary = {
        "frames": len(frames),
        "span_s": round(span, 1),
        "ball_frame_share": round(ball_frames / len(frames), 3),
        "owner_known_share": round(total_own / len(frames), 3),
        "possession_ours": round(ours_own / total_own, 3) if total_own else None,
        "pressed_share_ours": round(pressed_ours_frames / ours_own, 3) if ours_own else None,
        "losses_ours": len(lost), "losses_opp": len(won),
        "losses_ours_pressed": len(pressed_losses), "losses_ours_free": len(free_losses),
        "spread_ours": _mean([s for w in wins.values() for s in w.spread_ours]),
        "spread_opp": _mean([s for w in wins.values() for s in w.spread_opp]),
    }
    ws = [w for _, w in sorted(wins.items())]
    findings = _findings(ws, lost, press_dist, free_dist, summary)
    return {"summary": summary, "turnovers": tos, "windows": ws, "findings": findings}


def _mean(v: list[float]) -> float | None:
    return round(float(np.mean(v)), 2) if v else None


def _findings(ws: list[Window], lost: list[Turnover], press_dist: float, free_dist: float,
              summary: dict) -> list[Finding]:
    out: list[Finding] = []
    for t in lost:
        if t.pressed_dist >= free_dist:
            out.append(Finding("loss_free", 2, t.t, OURS,
                               text="Потеря мяча без давления соперника", clip=_clip(t.t)))
        elif t.pressed_dist <= press_dist:
            out.append(Finding("loss_pressed", 2, t.t, OURS,
                               text="Потеря мяча под давлением соперника", clip=_clip(t.t)))
    for w in ws:
        lbl = f"{format_ts(w.start)[:8]}–{format_ts(w.end)[:8]}"
        if w.share is not None and w.own_frames >= 20 and w.share < 0.35:
            out.append(Finding("fade_window", 2, w.start, OURS,
                               text=f"Провал владения: {w.share * 100:.0f}% мяча в отрезке {lbl}",
                               clip=(w.start, w.end)))
        if w.ours_frames >= 20 and w.pressed_ours / w.ours_frames > 0.6:
            out.append(Finding("pressed_window", 2, w.start, OURS,
                               text=f"Сильное давление на нашего владельца в отрезке {lbl}",
                               clip=(w.start, w.end)))
        if w.spread_ours and w.spread_opp:
            so, sp = float(np.mean(w.spread_ours)), float(np.mean(w.spread_opp))
            if len(w.spread_ours) >= 20 and so > 1.3 * sp:
                out.append(Finding("spread_window", 1, w.start, OURS,
                                   text=f"Мы растянуты сильнее соперника ({so:.1f} против "
                                        f"{sp:.1f}) в отрезке {lbl}", clip=(w.start, w.end)))
    return out
