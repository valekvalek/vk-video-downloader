import numpy as np

from matchlens.analytics.errors import line_gaps, losses_in_own_third
from matchlens.analytics.metrics import (
    convex_hull_area,
    distance_and_top_speed,
    heatmap,
    team_shape,
)
from matchlens.demo import run_demo
from matchlens.events.possession import Owner, ball_owner, detect_turnovers, possession_share
from matchlens.report.plan import build_llm_prompt, build_plan, episodes, render_markdown
from matchlens.schemas import OPP, OURS, Event, Finding, MatchData, PlayerTrack


def test_ball_owner_picks_nearest_within_radius():
    players = [(1, OURS, 6, 10.0, 10.0), (2, OPP, 9, 11.0, 10.0), (3, "ref", None, 10.2, 10.0)]
    o = ball_owner((10.9, 10.0), players, max_dist=2.0)
    assert o is not None and o.track_id == 2  # судья игнорируется
    assert ball_owner((50.0, 50.0), players) is None


def test_possession_share():
    owners = [Owner(OURS, 1)] * 3 + [Owner(OPP, 2)] + [None]
    s = possession_share(owners)
    assert s[OURS] == 0.75 and s[OPP] == 0.25


def test_turnover_needs_hold_time():
    fps = 25.0
    ours, opp = Owner(OURS, 1, 6), Owner(OPP, 2, 9)
    ball = [(20.0, 30.0)] * 200
    flicker = [ours] * 50 + [opp] * 3 + [ours] * 50  # 3 кадра — шум
    assert detect_turnovers(flicker, ball[:103], fps, min_hold_s=0.5) == []
    real = [ours] * 50 + [opp] * 30
    ev = detect_turnovers(real, ball[:80], fps, min_hold_s=0.5)
    kinds = {e.kind: e for e in ev}
    assert kinds["turnover_loss"].team == OURS and kinds["turnover_loss"].number == 6
    assert kinds["turnover_win"].team == OPP


def test_loss_in_own_third_only_for_ours_and_deep():
    events = [
        Event(10, "turnover_loss", OURS, 6, 20.0, 30.0),  # своя треть
        Event(20, "turnover_loss", OURS, 8, 60.0, 30.0),  # середина
        Event(30, "turnover_loss", OPP, 5, 10.0, 30.0),  # соперник
    ]
    f = losses_in_own_third(events, attack_dir=1)
    assert len(f) == 1 and f[0].numbers == [6] and f[0].clip == (4.0, 14.0)
    flipped = losses_in_own_third(events, attack_dir=-1)  # атакуем влево: глубоко = большой x
    assert flipped == []


def test_convex_hull_area_square():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]])
    assert abs(convex_hull_area(sq) - 100.0) < 1e-9


def test_team_shape_and_line_gaps():
    pts = np.column_stack([np.linspace(10, 70, 9), np.linspace(5, 60, 9)])
    s = team_shape(pts, attack_dir=1)
    assert s.defense_line < s.midfield_line < s.attack_line
    assert team_shape(pts[:4]) is None
    found = line_gaps([(100.0, s), (105.0, s), (130.0, s)], max_gap=20.0, min_gap_s=10.0)
    assert [f.t for f in found] == [100.0, 130.0]  # 105 склеено с 100


def test_heatmap_counts_points():
    h = heatmap(np.array([[10.0, 10.0], [10.5, 10.2], [90.0, 60.0]]))
    assert h.sum() == 3


def test_distance_and_speed_skip_gaps_and_jumps():
    pts = [(0, 0.0, 0.0), (1, 0.2, 0.0), (2, 0.4, 0.0), (5, 9.0, 0.0), (6, 50.0, 0.0)]
    dist, top = distance_and_top_speed(pts, fps=25.0)
    assert abs(dist - 0.4) < 1e-9 and abs(top - 5.0) < 1e-9


def test_plan_requires_evidence_and_orders_by_severity():
    fs = [
        Finding("line_gap", 2, 100.0, OURS, text="x", clip=(94, 104)),
        Finding("loss_own_third", 3, 200.0, OURS, numbers=[6], text="y", clip=(194, 204)),
        Finding("loss_own_third", 3, 400.0, OURS, numbers=[4], text="z", clip=(394, 404)),
    ]
    plan = build_plan(fs)
    assert plan.recommendations[0].numbers == [4, 6]
    assert all(r.evidence for r in plan.recommendations)
    md = render_markdown(plan)
    assert "№4, №6" in md and "F002@00:03:20" in md
    prompt = build_llm_prompt(fs, {"possession": {OURS: 0.5}})
    assert "F001@" in prompt and "не выдумывай" in prompt


def test_demo_runs_end_to_end():
    text = run_demo()
    assert "потерял мяч" not in text  # текст находок — в JSON/промпте, не в плане
    assert "Безопасный выход" in text and "Компактность" in text


def test_matchdata_json_roundtrip():
    import tempfile
    from pathlib import Path

    md = MatchData(fps=25.0, tracks=[PlayerTrack(1, OURS, 6, 0.9, [(0, 1.0, 2.0)])],
                   ball=[(0, 5.0, 6.0)], events=[Event(1.0, "turnover_loss", OURS, 6, 1.0, 2.0)])
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.json"
        md.to_json(p)
        back = MatchData.from_json(p)
    assert back.tracks[0].points == [(0, 1.0, 2.0)] and back.events[0].number == 6


def test_episodes_plural():
    assert [episodes(n) for n in (1, 2, 5, 11, 21, 22)] == [
        "1 эпизод", "2 эпизода", "5 эпизодов", "11 эпизодов", "21 эпизод", "22 эпизода"]
