"""Запись PNG и «контактный лист» из вырезок без внешних библиотек (только numpy + zlib)."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def write_png(path: str | Path, rgb: np.ndarray) -> None:
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def resize_nearest(img: np.ndarray, width: int, height: int) -> np.ndarray:
    ys = (np.arange(height) * img.shape[0] / height).astype(int)
    xs = (np.arange(width) * img.shape[1] / width).astype(int)
    return img[ys][:, xs]


def contact_sheet(crops: list[np.ndarray], cell: tuple[int, int] = (96, 128),
                  cols: int = 8, pad: int = 4) -> np.ndarray:
    """Сетка из вырезок одинакового размера ячейки (ширина, высота)."""
    cw, ch = cell
    rows = max(1, -(-len(crops) // cols))
    sheet = np.full((rows * (ch + pad) + pad, cols * (cw + pad) + pad, 3), 40, np.uint8)
    for i, crop in enumerate(crops):
        r, c = divmod(i, cols)
        y, x = pad + r * (ch + pad), pad + c * (cw + pad)
        sheet[y:y + ch, x:x + cw] = resize_nearest(crop, cw, ch)
    return sheet
