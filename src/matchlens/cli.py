"""Командная строка: matchlens <команда>."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .ingest import prepare as prep
from .roster import RosterError, load_roster


def cmd_probe(a) -> int:
    info = prep.probe(a.video)
    print(f"{info.width}x{info.height} @ {info.fps:.2f} fps, {prep.format_ts(info.duration)}")
    return 0


def cmd_prepare(a) -> int:
    start = prep.parse_ts(a.start) if a.start else None
    end = prep.parse_ts(a.end) if a.end else None
    out = prep.make_proxy(a.video, a.output, height=a.height, fps=a.fps, start=start, end=end)
    print(f"Готово: {out}")
    return 0


def cmd_roster_check(a) -> int:
    try:
        r = load_roster(a.roster)
    except RosterError as e:
        print(f"Ошибка состава: {e}", file=sys.stderr)
        return 1
    print(f"OK: {r.ours.name} ({len(r.ours.players)} игроков), "
          f"{r.opp.name} ({len(r.opp.players)} игроков)")
    return 0


def cmd_prototype(a) -> int:
    """Проверка на реальном видео: фрагмент -> кадры -> детекция -> отчёт о читаемости номеров.

    --start auto: сначала сканирует начало записи и находит начало игры (kickoff.md).
    """
    import tempfile

    from .ingest.stream import fetch_section, iter_frames
    from .kickoff import pick_kickoff, scan_counts, write_kickoff_report
    from .pipeline_prototype import run_prototype
    from .vision.detect import YoloDetector

    local = Path(a.source).exists()
    duration = a.minutes * 60
    with tempfile.TemporaryDirectory() as tmp:  # видео живёт только здесь и удаляется
        if a.start == "auto":
            scan_from = 0.0 if local else prep.parse_ts(a.scan_from)
            scan_len = a.scan_minutes * 60
            print(f"Ищу начало игры: {a.scan_minutes:g} мин с {prep.format_ts(scan_from)[:8]} ...")
            if local:
                scan_video, scan_start = Path(a.source), 0.0
            else:
                scan_video = fetch_section(a.source, scan_from, scan_from + scan_len, tmp + "/scan",
                                           height=480)
                scan_start = 0.0
            step = a.scan_step
            scanner = YoloDetector("yolo11n.pt", imgsz=960)
            counts, thumbs = scan_counts(
                iter_frames(scan_video, fps=1 / step, start=scan_start, duration=scan_len),
                scanner, step_s=step, offset_s=scan_from)
            kickoff = pick_kickoff(counts, min_players=a.min_players, sustain_s=90.0)
            params = {"min_players": a.min_players, "sustain_s": 90.0}
            print(write_kickoff_report(a.out, kickoff, counts, thumbs, params))
            if kickoff is None:
                print("Начало игры не найдено — анализ остановлен. Укажите --start вручную.")
                return 2
            start = kickoff
        else:
            start = prep.parse_ts(a.start)

        if local:
            video, offset = Path(a.source), start
        else:
            print(f"Беру фрагмент {prep.format_ts(start)[:8]} + {a.minutes:g} мин ...")
            video = fetch_section(a.source, start, start + duration, tmp + "/main", height=a.height)
            offset = 0.0
        detector = YoloDetector(a.weights, imgsz=a.imgsz)
        summary = run_prototype(
            iter_frames(video, fps=a.fps, start=offset, duration=duration),
            detector, a.out, fps=a.fps)
    print((Path(a.out) / "report.md").read_text(encoding="utf-8"))
    print(f"Файлы отчёта: {a.out}/ (видео не сохранено). Кадров: {summary['frames']}")
    return 0


def cmd_chunks(a) -> int:
    """Список кусков для параллельного запуска (JSON для matrix в GitHub Actions)."""
    import json

    start = prep.parse_ts(a.start)
    total, step = a.total * 60, a.chunk * 60
    items, pos, i = [], 0.0, 0
    while pos < total - 1:
        length = min(step, total - pos)
        items.append({"name": f"c{i:02d}", "start": prep.format_ts(start + pos)[:8],
                      "minutes": round(length / 60, 2)})
        pos += length
        i += 1
    print(json.dumps(items, ensure_ascii=False))
    return 0


def cmd_observe(a) -> int:
    """Один кусок записи -> файл наблюдений (детекция игроков, мяча, цвета формы)."""
    import tempfile

    from .ingest.stream import fetch_section, iter_frames
    from .observe import observe_frames, save_observations
    from .vision.detect import YoloDetector

    start, duration = prep.parse_ts(a.start), a.minutes * 60
    with tempfile.TemporaryDirectory() as tmp:  # видео живёт только здесь и удаляется
        print(f"Беру фрагмент {prep.format_ts(start)[:8]} + {a.minutes:g} мин ...")
        video = fetch_section(a.source, start, start + duration, tmp, height=a.height)
        detector = YoloDetector(a.weights, imgsz=a.imgsz)
        obs = observe_frames(iter_frames(video, fps=a.fps, duration=duration), detector,
                             fps=a.fps, t0=start)
    save_observations(obs, a.out)
    print(f"Готово: {len(obs['frames'])} кадров -> {a.out}")
    return 0


def cmd_merge(a) -> int:
    """Собрать наблюдения всех кусков в отчёт по матчу и план улучшений."""
    from .analytics.match import analyse, cluster_teams, label_frames
    from .ingest.stream import normalize_url
    from .observe import load_observations
    from .report.match_report import render_match_report

    files = sorted(Path(a.inputs).rglob("*.json.gz")) if Path(a.inputs).is_dir() \
        else [Path(a.inputs)]
    if not files:
        print("Нет файлов наблюдений (*.json.gz)")
        return 2
    frames = []
    for f in files:
        frames += load_observations(f)["frames"]
    frames.sort(key=lambda x: x["t"])
    colors = {}
    if a.ours_color:
        colors["ours"] = tuple(int(v) for v in a.ours_color.split(","))
    if a.opp_color:
        colors["opp"] = tuple(int(v) for v in a.opp_color.split(","))
    if a.ref_color:
        colors["ref"] = tuple(int(v) for v in a.ref_color.split(","))
    centers, counts = cluster_teams(frames, colors)
    res = analyse(label_frames(frames, centers), window_s=a.window * 60)
    clusters = {r: (counts[r], centers[r]) for r in centers}
    url = normalize_url(a.video_url) if a.video_url else ""
    text = render_match_report(res, clusters, url)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "match_report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def cmd_demo(a) -> int:
    """Сквозной пример на синтетических данных: события -> находки -> план."""
    from .demo import run_demo

    text = run_demo()
    if a.output:
        Path(a.output).write_text(text, encoding="utf-8")
        print(f"План записан в {a.output}")
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="matchlens", description="Анализ футбольного матча по видео")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("probe", help="параметры видеофайла")
    s.add_argument("video")
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("prepare", help="сделать легковесную копию для анализа")
    s.add_argument("video")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--height", type=int, default=720)
    s.add_argument("--fps", type=int, default=25)
    s.add_argument("--start", help="начало, напр. 00:05:00")
    s.add_argument("--end", help="конец, напр. 00:10:00")
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("roster-check", help="проверить файл состава")
    s.add_argument("roster")
    s.set_defaults(func=cmd_roster_check)

    s = sub.add_parser("prototype", help="проверка на реальном видео (нужны ultralytics и ffmpeg)")
    s.add_argument("source", help="ссылка на видео или путь к файлу")
    s.add_argument("--start", default="auto",
                   help="начало фрагмента чч:мм:сс или auto (найти начало игры)")
    s.add_argument("--minutes", type=float, default=10.0)
    s.add_argument("--scan-from", default="00:00:00", help="с какого места искать начало игры")
    s.add_argument("--scan-minutes", type=float, default=40.0)
    s.add_argument("--scan-step", type=float, default=10.0, help="шаг сканирования, секунд")
    s.add_argument("--min-players", type=int, default=1)
    s.add_argument("--fps", type=float, default=2.0)
    s.add_argument("--height", type=int, default=720, help="макс. высота потока")
    s.add_argument("--weights", default="yolo11s.pt")
    s.add_argument("--imgsz", type=int, default=1280)
    s.add_argument("--out", default="out")
    s.set_defaults(func=cmd_prototype)

    s = sub.add_parser("chunks", help="нарезать матч на куски для параллельного запуска")
    s.add_argument("--start", default="00:00:00")
    s.add_argument("--total", type=float, required=True, help="сколько минут анализировать")
    s.add_argument("--chunk", type=float, default=10.0, help="минут в куске")
    s.set_defaults(func=cmd_chunks)

    s = sub.add_parser("observe", help="один кусок записи -> файл наблюдений")
    s.add_argument("source")
    s.add_argument("--start", required=True)
    s.add_argument("--minutes", type=float, default=10.0)
    s.add_argument("--fps", type=float, default=2.0)
    s.add_argument("--height", type=int, default=720)
    s.add_argument("--weights", default="yolo11s.pt")
    s.add_argument("--imgsz", type=int, default=1280)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_observe)

    s = sub.add_parser("merge", help="наблюдения кусков -> отчёт по матчу и план")
    s.add_argument("inputs", help="папка с *.json.gz или один файл")
    s.add_argument("--video-url", default="")
    s.add_argument("--ours-color", help="R,G,B формы нашей команды")
    s.add_argument("--opp-color", help="R,G,B формы соперника")
    s.add_argument("--ref-color", help="R,G,B формы судьи")
    s.add_argument("--window", type=float, default=2.0, help="длина отрезка в минутах")
    s.add_argument("--out", default="out")
    s.set_defaults(func=cmd_merge)

    s = sub.add_parser("demo", help="пример плана на синтетических данных")
    s.add_argument("-o", "--output")
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

