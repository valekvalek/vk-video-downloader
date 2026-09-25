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
    rec_h: list[float] = []              # рост каждого наблюдения с известным цветом
    rec_c: list[np.ndarray] = []         # цвет формы этого наблюдения
    balls = 0
    n_frames = 0
    reservoir: list[tuple[float, int, np.ndarray, np.ndarray | None]] = []  # выборка вырезок
    cap = 1500
    seen = 0
    rng = np.random.default_rng(0)

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
                rec_h.append(hgt)
                rec_c.append(c)
            crop = _upper_body(frame, d.bbox)
            if crop.size:  # резервуарная выборка: статистика по всем, вырезки — по выборке
                seen += 1
                item = (hgt, idx, crop.copy(), c)
                if len(reservoir) < cap:
                    reservoir.append(item)
                else:
                    j = int(rng.integers(seen))
                    if j < cap:
                        reservoir[j] = item

    # Три группы по цвету формы: команда 1, команда 2 и судья (в кадре обычно один и ближе всех
    # к камере — его крупные планы иначе завышают оценку читаемости номеров игроков).
    clusters: list[dict] = []
    ref_cluster = None
    centers = None
    labels = np.zeros(0, dtype=int)
    if len(rec_c) >= 30:
        labels, centers = kmeans(np.array(rec_c), k=3)
        hs_all = np.array(rec_h)
        for j in range(3):
            mask = labels == j
            if not mask.any():
                continue
            hs = hs_all[mask]
            clusters.append({"id": j, "n": int(mask.sum()), "p50": float(np.percentile(hs, 50)),
                             "p90": float(np.percentile(hs, 90)),
                             "legible_share": round(float((hs >= legible_px).mean()), 3)})
        if len(clusters) == 3:
            ref_cluster = min(clusters, key=lambda c: c["n"])["id"]  # самая малочисленная
    if len(rec_c) >= 30:
        keep = labels != ref_cluster if ref_cluster is not None else np.ones(len(rec_h), bool)
        h = np.array(rec_h)[keep]
    else:
        h = np.array(heights) if heights else np.zeros(1)
    legible_share = float((h >= legible_px).mean()) if len(h) else 0.0

    team_counts = {"A": 0, "B": 0}
    others = [c for c in clusters if c["id"] != ref_cluster]
    for key, c in zip(("A", "B"), others, strict=False):
        team_counts[key] = c["n"]

    def _cluster_of(color):
        return int(np.linalg.norm(centers - color, axis=1).argmin())

    by_cluster: dict[int, list] = {c["id"]: [] for c in clusters}
    pool = []
    for it in reservoir:
        if centers is not None and it[3] is not None:
            cid = _cluster_of(it[3])
            by_cluster.setdefault(cid, []).append(it)
            if cid != ref_cluster:
                pool.append((it[0], it[1], it[2]))
        elif centers is None:
            pool.append((it[0], it[1], it[2]))
    for cid, members in by_cluster.items():
        top = sorted(members, key=lambda m: -m[0])[:16]
        if top:
            tag = "_referee" if cid == ref_cluster else ""
            write_png(out / f"crops_cluster{cid}{tag}.png", contact_sheet([m[2] for m in top]))

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
        "referee_cluster_observations": next(
            (c["n"] for c in clusters if c["id"] == ref_cluster), 0),
        "clusters": [{k: v for k, v in c.items() if k != "members"} for c in clusters],
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
        f"группа B — {s['team_color_clusters']['B']} наблюдений; "
        f"предположительно судья (отброшен из оценки) — {s['referee_cluster_observations']}.",
        "",
        "## Вывод",
        "",
        verdict(s),
        "",
        "Порог в пикселях — эвристика: окончательно читаемость покажет модель распознавания. "
        "Листы `crops_cluster*.png` — самые крупные игроки каждой цветовой группы (файл с "
        "`_referee` — предполагаемый судья); посмотрите глазами: видны ли номера.",
    ])
