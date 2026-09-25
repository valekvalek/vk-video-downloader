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
