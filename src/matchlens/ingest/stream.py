"""Анализ без скачивания на свой компьютер: берём только нужный фрагмент на машине-анализаторе,
отдаём кадры в память и удаляем временный файл. Полное видео никуда не сохраняется."""
from __future__ import annotations

import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .prepare import probe


def normalize_url(url: str) -> str:
    """live-123_456 / clip123_456 -> https://vkvideo.ru/video-123_456 (так URL понимает yt-dlp)."""
    url = url.strip()
    m = re.search(r"vk(?:video)?\.(?:ru|com)/.*?(?:live|video|clip)(-?\d+_\d+)", url)
    return f"https://vkvideo.ru/video{m.group(1)}" if m else url


def fetch_section(url: str, start: float, end: float, dest_dir: str | Path,
                  height: int = 720) -> Path:
    """Скачать через yt-dlp только [start, end] секунд, без звука (для CV он не нужен)."""
    import yt_dlp
    from yt_dlp.utils import download_range_func

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": f"bv*[height<={height}]/bv*/b[height<={height}]/b",
        "outtmpl": str(dest / "section.%(ext)s"),
        "download_ranges": download_range_func(None, [(start, end)]),
        "concurrent_fragment_downloads": 8,
        "retries": 10,
        "fragment_retries": 10,
        "noplaylist": True,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([normalize_url(url)])
    files = sorted(dest.glob("section.*"))
    if not files:
        raise RuntimeError("yt-dlp не создал файл фрагмента")
    return files[0]


def iter_frames(path: str | Path, fps: float = 2.0, start: float = 0.0,
                duration: float | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Кадры RGB (H, W, 3) с частотой fps; читаются потоком из ffmpeg, файл целиком не грузится.

    start/duration — окно внутри файла (секунды); индексы кадров идут с 0 от начала окна.
    """
    info = probe(path)
    w, h = info.width, info.height
    cmd = ["ffmpeg", "-v", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h * 3
    try:
        idx = 0
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield idx, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            idx += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
