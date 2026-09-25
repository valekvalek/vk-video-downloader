"""Проверочный прототип: хватает ли качества видео, чтобы читать номера и различать команды.

Ничего не обучает и не «додумывает»: меряет размер игроков в кадре, долю мяча, разделение
команд по цвету и складывает самые крупные вырезки в один лист для быстрого просмотра глазами.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .report.png import contact_sheet, write_png
from .schemas import BBox
from .vision.detect import Detector
from .vision.team_split import dominant_color, kmeans, torso_crop

# Номер на спине ~13% роста игрока: при росте 100 px цифра ~13 px — граница для OCR.
DEFAULT_LEGIBLE_PX = 100


def _upper_body(frame: np.ndarray, b: BBox) -> np.ndarray:
    """Голова-плечи-спина: где находится номер."""
    h, w = frame.shape[:2]
    y2 = int(min(h, b.y1 + 0.6 * (b.y2 - b.y1)))
    return frame[int(max(0, b.y1)):y2, int(max(0, b.x1)):int(min(w, b.x2))]


def run_prototype(frames, detector: Detector, out_dir: str | Path, fps: float,
                  max_crops: int = 48, legible_px: int = DEFAULT_LEGIBLE_PX,
                  min_conf: float = 0.3) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    heights: list[float] = []
    per_frame: list[int] = []
    colors: list[np.ndarray] = []
    balls = 0
    n_frames = 0
    pool: list[tuple[float, int, np.ndarray]] = []  # (высота, кадр, вырезка)

    for idx, frame in frames:
        n_frames += 1
        dets = detector.detect(frame[:, :, ::-1], idx)  # детектор ждёт BGR
        players = [d for d in dets if d.cls in ("player", "goalkeeper") and d.conf >= min_conf]
        balls += any(d.cls == "ball" for d in dets)
        per_frame.append(len(players))
        for d in players:
            hgt = d.bbox.y2 - d.bbox.y1
            heights.append(hgt)
            c = dominant_color(torso_crop(frame, d.bbox))
            if c is not None:
                colors.append(c)
            crop = _upper_body(frame, d.bbox)
            if crop.size:
                pool.append((hgt, idx, crop.copy()))
        if len(pool) > 6 * max_crops:
            pool = sorted(pool, key=lambda p: -p[0])[: 3 * max_crops]

    h = np.array(heights) if heights else np.zeros(1)
    legible_share = float((h >= legible_px).mean()) if heights else 0.0

    team_counts = {"A": 0, "B": 0}
    if len(colors) >= 10:
        labels, _ = kmeans(np.array(colors), k=2)
        team_counts = {"A": int((labels == 0).sum()), "B": int((labels == 1).sum())}

    best = sorted(pool, key=lambda p: -p[0])[:max_crops]
    best.sort(key=lambda p: p[1])
    if best:
        write_png(out / "crops_sheet.png", contact_sheet([c for _, _, c in best]))

    summary = {
        "frames": n_frames,
        "analysed_seconds": round(n_frames / fps, 1),
        "players_per_frame_median": float(np.median(per_frame)) if per_frame else 0.0,
        "player_height_px": {
            "p50": float(np.percentile(h, 50)),
            "p90": float(np.percentile(h, 90)),
        },
        "legible_threshold_px": legible_px,
        "legible_share": round(legible_share, 3),
        "ball_frame_share": round(balls / n_frames, 3) if n_frames else 0.0,
        "team_color_clusters": team_counts,
        "crops_saved": len(best),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    (out / "report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def verdict(s: dict) -> str:
    share = s["legible_share"]
    if s["frames"] == 0:
        return "Кадров не получено — проверьте ссылку и время начала."
    if share >= 0.3:
        return ("Крупных игроков достаточно: автоматическое чтение номеров имеет смысл пробовать "
                "(этап 3), для остальных — ручной ввод состава.")
    if share >= 0.1:
        return ("Крупных планов мало: номера читаются только эпизодически. Разумный путь — "
                "состав вручную + распознавание по цвету и позиции, номера авто — как бонус.")
    return ("Игроки в кадре слишком мелкие для чтения номеров. Идти по схеме «цвет формы + "
            "позиция + ручной состав»; проверить другое видео или камеру ближе.")


def render_report(s: dict) -> str:
    return "\n".join([
        "# Отчёт проверочного прототипа",
        "",
        f"Проанализировано: {s['analysed_seconds']} с ({s['frames']} кадров).",
        f"Игроков в кадре (медиана): {s['players_per_frame_median']:.0f}.",
        f"Рост игрока в кадре: медиана {s['player_height_px']['p50']:.0f} px, "
        f"крупные (90-й процентиль) {s['player_height_px']['p90']:.0f} px.",
        f"Доля игроков с ростом не менее {s['legible_threshold_px']} px "
        f"(грубый порог читаемости номера): {s['legible_share'] * 100:.0f}%.",
        f"Мяч найден на {s['ball_frame_share'] * 100:.0f}% кадров.",
        f"Разделение по цвету формы: группа A — {s['team_color_clusters']['A']}, "
        f"группа B — {s['team_color_clusters']['B']} наблюдений.",
        "",
        "## Вывод",
        "",
        verdict(s),
        "",
        "Порог в пикселях — эвристика: окончательно читаемость покажет модель распознавания. "
        "Лист `crops_sheet.png` — самые крупные игроки; посмотрите его глазами: видны ли номера.",
    ])
