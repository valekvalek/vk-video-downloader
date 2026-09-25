import tempfile

from matchlens.analytics.match import analyse, cluster_teams, find_turnovers, label_frames
from matchlens.cli import main
from matchlens.observe import load_observations, save_observations
from matchlens.report.match_report import render_match_report

BLUE, WHITE, REF = (42, 49, 81), (195, 199, 222), (36, 35, 42)  # как в реальной палитре


def _player(x, y, col, h=100):
    return [x, y, x + h // 2, y + h, *col]


def _frame(t, owner, pressed):
    """Синие слева, белые справа, судья с краю; мяч у ног владельца (ours/opp)."""
    ours = [_player(100 + 60 * i, 200, BLUE) for i in range(5)]
    opp = [_player(800 + 60 * i, 200, WHITE) for i in range(5)]
    if owner == "ours":
        b = ours[0]
        if pressed:
            opp[0] = _player(b[0] + 40, 200, WHITE)
    else:
        b = opp[0]
    ball = [(b[0] + b[2]) // 2, b[3] - 10, 0.9]
    return {"t": t, "p": ours + opp + [_player(500, 250, REF)], "b": ball}


def _match():
    frames, t = [], 0.0
    # 0-60 с: наше владение, потеря без давления; 60-120: у соперника; 120-180: наше с давлением
    for _ in range(30):
        frames.append(_frame(t, "ours", False))
        t += 2
    for _ in range(30):
        frames.append(_frame(t, "opp", False))
        t += 2
    for _ in range(30):
        frames.append(_frame(t, "ours", True))
        t += 2
    return frames


def test_teams_and_turnovers():
    fr = _match()
    centers, counts = cluster_teams(fr)
    assert counts["ours"] == 5 * 90 and counts["opp"] == 5 * 90 and counts["ref"] == 90
    tos = find_turnovers(label_frames(fr, centers))
    assert [t.loser for t in tos] == ["ours", "opp"]
    assert tos[0].pressed_dist > 2.0  # потеря без давления


def test_analyse_and_report():
    fr = _match()
    centers, counts = cluster_teams(fr)
    res = analyse(label_frames(fr, centers), window_s=60)
    s = res["summary"]
    assert s["losses_ours"] == 1 and s["losses_opp"] == 1
    assert 0.6 < s["possession_ours"] < 0.7
    assert s["pressed_share_ours"] is not None and s["pressed_share_ours"] > 0.4
    assert any(f.kind == "loss_free" for f in res["findings"])
    text = render_match_report(res, {r: (counts[r], centers[r][0]) for r in centers},
                               "https://vkvideo.ru/video-1_2")
    assert "План улучшений" in text and "?t=" in text and "Потери мяча" in text


def test_observation_roundtrip_and_merge_cli(capsys=None):
    fr = _match()
    with tempfile.TemporaryDirectory() as d:
        save_observations({"t0": 0, "fps": 0.5, "size": [1280, 720], "frames": fr[:45]},
                          d + "/c00.json.gz")
        save_observations({"t0": 90, "fps": 0.5, "size": [1280, 720], "frames": fr[45:]},
                          d + "/c01.json.gz")
        assert len(load_observations(d + "/c00.json.gz")["frames"]) == 45
        code = main(["merge", d, "--out", d + "/rep", "--window", "1",
                     "--video-url", "https://vkvideo.ru/live-1_2"])
        assert code == 0
        text = open(d + "/rep/match_report.md", encoding="utf-8").read()
        assert "video-1_2?t=" in text  # ссылка нормализована
