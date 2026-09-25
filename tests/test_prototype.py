import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from matchlens.ingest.stream import iter_frames
from matchlens.pipeline_prototype import run_prototype, verdict
from matchlens.report.png import contact_sheet, resize_nearest, write_png
from matchlens.schemas import BBox, Detection


class FakeDetector:
    """Два «игрока» разного роста и «мяч» на каждом кадре."""

    def __init__(self, tall: float):
        self.tall = tall

    def detect(self, frame_bgr, frame_idx):
        return [
            Detection(frame_idx, "player", BBox(20, 10, 20 + self.tall / 3, 10 + self.tall), 0.9),
            Detection(frame_idx, "player", BBox(120, 10, 150, 60), 0.9),
            Detection(frame_idx, "ball", BBox(90, 90, 96, 96), 0.5),
        ]


def _frames(n=6, size=(180, 320)):
    for i in range(n):
        f = np.full((size[0], size[1], 3), (30, 160, 40), np.uint8)
        f[10:110, 20:60] = (220, 30, 30)
        f[10:60, 120:150] = (30, 40, 200)
        yield i, f


def test_png_and_contact_sheet_roundtrip():
    crops = [np.full((40, 30, 3), 200, np.uint8)] * 5
    sheet = contact_sheet(crops, cell=(24, 32), cols=3, pad=2)
    assert sheet.shape == (2 * 34 + 2, 3 * 26 + 2, 3)
    assert resize_nearest(crops[0], 10, 12).shape == (12, 10, 3)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.png"
        write_png(p, sheet)
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_prototype_summary_and_files():
    with tempfile.TemporaryDirectory() as d:
        s = run_prototype(_frames(), FakeDetector(tall=100), d, fps=2.0, max_crops=6)
        assert s["frames"] == 6 and s["players_per_frame_median"] == 2
        assert s["legible_share"] == 0.5  # один из двух игроков >= 100 px
        assert s["ball_frame_share"] == 1.0
        assert (Path(d) / "crops_sheet.png").exists()
        assert "Вывод" in (Path(d) / "report.md").read_text(encoding="utf-8")
        assert json.loads((Path(d) / "summary.json").read_text())["frames"] == 6


def test_verdict_levels():
    base = {"frames": 10, "legible_share": 0.5}
    assert "имеет смысл" in verdict(base)
    assert "эпизодически" in verdict({**base, "legible_share": 0.15})
    assert "слишком мелкие" in verdict({**base, "legible_share": 0.0})
    assert "Кадров не получено" in verdict({**base, "frames": 0})


def test_iter_frames_streams_rgb_frames():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "t.mp4"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25",
             "-t", "3", "-pix_fmt", "yuv420p", str(src)], check=True)
        frames = list(iter_frames(src, fps=2))
    assert 5 <= len(frames) <= 7
    assert frames[0][1].shape == (180, 320, 3) and frames[0][1].dtype == np.uint8


def test_pick_kickoff_ignores_studio_and_short_bursts():
    from matchlens.kickoff import pick_kickoff

    # 0-300 с студия (0 игроков), 300-360 вспышка 15 игроков, пауза, с 600 с — игра
    counts = [(t, 0) for t in range(0, 300, 10)]
    counts += [(t, 15) for t in range(300, 360, 10)] + [(t, 2) for t in range(360, 600, 10)]
    counts += [(t, 16) for t in range(600, 900, 10)]
    assert pick_kickoff(counts, min_players=9, sustain_s=90) == 600
    assert pick_kickoff([(t, 3) for t in range(0, 600, 10)]) is None
    assert pick_kickoff([]) is None


def test_scan_and_kickoff_report_files():
    import tempfile

    from matchlens.kickoff import scan_counts, write_kickoff_report

    class Det:
        def detect(self, frame_bgr, i):
            n = 14 if i >= 3 else 0
            return [Detection(i, "player", BBox(0, 0, 10, 20), 0.9) for _ in range(n)]

    counts, thumbs = scan_counts(_frames(8), Det(), step_s=10.0, offset_s=100.0)
    assert counts[0] == (100.0, 0) and counts[3] == (130.0, 14) and len(thumbs) == 8
    with tempfile.TemporaryDirectory() as d:
        text = write_kickoff_report(d, 130.0, counts, thumbs, {"min_players": 12, "sustain_s": 90})
        assert "00:02:10" in text and (Path(d) / "kickoff_sheet.png").exists()
        none = write_kickoff_report(d + "/x", None, counts, thumbs, {"min_players": 12,
                                                                     "sustain_s": 90})
        assert "не найдено" in none


def test_iter_frames_start_and_duration():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "t.mp4"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=25",
             "-t", "6", "-pix_fmt", "yuv420p", str(src)], check=True)
        all_frames = list(iter_frames(src, fps=1))
        window = list(iter_frames(src, fps=1, start=2, duration=2))
    assert len(all_frames) == 6 and 1 <= len(window) <= 3


def test_referee_cluster_excluded_from_legibility():
    """Судья (малочисленная цветовая группа, крупный) не должен завышать читаемость номеров."""
    import tempfile

    import numpy as np

    from matchlens.pipeline_prototype import run_prototype
    from matchlens.schemas import BBox, Detection

    def scene(n):
        f = np.zeros((300, 400, 3), np.uint8)
        f[:, :] = (20, 120, 20)
        dets = []
        # 6 маленьких красных, 6 маленьких синих, 1 крупный жёлтый (судья) — рост 200 px
        for k, (x, col) in enumerate([(10 + 25 * i, (200, 30, 30)) for i in range(6)]
                                      + [(10 + 25 * i, (30, 30, 200)) for i in range(6, 12)]):
            y = 100 if k < 6 else 150
            f[y:y + 40, x:x + 20] = col
            dets.append(Detection(n, "player", BBox(x, y, x + 20, y + 40), 0.9))
        f[20:220, 300:380] = (230, 220, 30)
        dets.append(Detection(n, "player", BBox(300, 20, 380, 220), 0.9))
        return f, dets

    class Det:
        def __init__(self):
            self.cache = {}

        def detect(self, frame_bgr, idx):
            return self.cache[idx]

    det = Det()
    frames = []
    for i in range(200):  # 2600 наблюдений > размера выборки вырезок: статистика — по всем
        f, d = scene(i)
        det.cache[i] = d
        frames.append((i, f))
    with tempfile.TemporaryDirectory() as d:
        s = run_prototype(frames, det, d, fps=2.0, legible_px=100)
    assert s["referee_cluster_observations"] == 200
    assert sum(c["n"] for c in s["clusters"]) == 2600
    assert s["legible_share"] == 0.0
