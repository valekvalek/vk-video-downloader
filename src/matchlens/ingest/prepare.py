"""Этап 1: подготовка видео. Легковесная копия для анализа + нарезка на куски.

Команды ffmpeg собираются чистыми функциями (их легко тестировать), запуск — отдельно.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoInfo:
    duration: float
    width: int
    height: int
    fps: float


def parse_ts(value: str | float | int) -> float:
    """'1:02:03.5' | '02:03' | '95' -> секунды."""
    if isinstance(value, (int, float)):
        return float(value)
    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3 or not all(re.fullmatch(r"\d+(\.\d+)?", p) for p in parts):
        raise ValueError(f"Некорректное время: {value!r}")
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + float(p)
    return seconds


def format_ts(seconds: float) -> str:
    """Секунды -> 'HH:MM:SS.mmm' (формат для ffmpeg)."""
    if seconds < 0:
        raise ValueError("Время не может быть отрицательным")
    ms = round(seconds * 1000)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def parse_fps(rate: str) -> float:
    """'25/1' | '30000/1001' -> float."""
    num, _, den = rate.partition("/")
    return float(num) / float(den or 1) if float(den or 1) else 0.0


def probe(path: str | Path) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate:format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    st = data["streams"][0]
    return VideoInfo(
        duration=float(data["format"]["duration"]),
        width=int(st["width"]),
        height=int(st["height"]),
        fps=parse_fps(st["avg_frame_rate"]),
    )


def build_proxy_cmd(
    src: str | Path,
    dst: str | Path,
    height: int = 720,
    fps: int = 25,
    crf: int = 23,
    start: float | None = None,
    end: float | None = None,
) -> list[str]:
    """Прокси для анализа: фиксированные fps и высота, без звука (для CV звук не нужен)."""
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    if start is not None:
        cmd += ["-ss", format_ts(start)]
    if end is not None:
        cmd += ["-to", format_ts(end)]
    cmd += [
        "-i", str(src),
        "-vf", f"scale=-2:{height},fps={fps}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
        str(dst),
    ]
    return cmd


def build_cut_cmd(src: str | Path, dst: str | Path, start: float, end: float,
                  reencode: bool = False) -> list[str]:
    """Клип [start, end] для отчёта. По умолчанию без перекодирования (быстро)."""
    if end <= start:
        raise ValueError("end должен быть больше start")
    cmd = ["ffmpeg", "-hide_banner", "-y", "-ss", format_ts(start), "-to", format_ts(end),
           "-i", str(src)]
    cmd += ["-c:v", "libx264", "-crf", "20", "-c:a", "aac"] if reencode else ["-c", "copy"]
    return cmd + [str(dst)]


def chunk_ranges(duration: float, chunk: float, overlap: float = 0.0) -> list[tuple[float, float]]:
    """Разбивка записи на куски для пакетной обработки (с перехлёстом для склейки треков)."""
    if chunk <= 0 or overlap < 0 or overlap >= chunk:
        raise ValueError("Нужно chunk > 0 и 0 <= overlap < chunk")
    ranges, start = [], 0.0
    while start < duration:
        end = min(start + chunk, duration)
        ranges.append((start, end))
        if end >= duration:
            break
        start = end - overlap
    return ranges


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def make_proxy(src: str | Path, dst: str | Path, **kw) -> Path:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    run(build_proxy_cmd(src, dst, **kw))
    return Path(dst)
