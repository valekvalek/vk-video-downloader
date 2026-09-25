"""Отчёт по матчу: цифры, провалы по окнам времени, план улучшений со ссылками на моменты."""
from __future__ import annotations

from ..ingest.prepare import format_ts
from ..schemas import Finding
from .plan import build_plan, episodes, finding_id


def moment_link(video_url: str, t: float) -> str:
    return f"{video_url}?t={int(t)}"


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def render_match_report(res: dict, clusters: dict, video_url: str = "") -> str:
    s = res["summary"]
    lines = ["# Анализ матча по видео", "",
             f"Проанализировано: {s['span_s'] / 60:.1f} мин, кадров {s['frames']}. "
             f"Мяч найден на {_pct(s['ball_frame_share'])} кадров, владелец мяча определён на "
             f"{_pct(s['owner_known_share'])}.", "",
             "## Итоги", "",
             f"- Владение мячом: наша команда {_pct(s['possession_ours'])}.",
             f"- Потери мяча: мы {s['losses_ours']}, соперник {s['losses_opp']} "
             f"(из наших потерь под давлением {s['losses_ours_pressed']}, "
             f"без давления {s['losses_ours_free']}).",
             f"- Наш владелец под давлением: {_pct(s['pressed_share_ours'])} времени владения.",
             f"- Растянутость (в ростах игрока): мы {s['spread_ours']}, "
             f"соперник {s['spread_opp']}.",
             "", "## Отрезки по 2 минуты", "",
             "| отрезок | наше владение | потерь у нас | потерь у соперника |",
             "| --- | --- | --- | --- |"]
    for w in res["windows"]:
        lines.append(f"| {format_ts(w.start)[:8]}–{format_ts(w.end)[:8]} | {_pct(w.share)} | "
                     f"{w.losses_ours} | {w.losses_opp} |")
    lines += ["", "## Цвета формы (проверка деления на команды)", ""]
    for role, (n, c) in clusters.items():
        lines.append(f"- {role}: {n} наблюдений, цвет RGB ({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f})")
    findings: list[Finding] = res["findings"]
    plan = build_plan(findings)
    lines += ["", "## План улучшений на следующую игру (без дополнительных тренировок)", ""]
    if not plan.recommendations:
        lines.append("Значимых проблем не найдено.")
    for i, r in enumerate(plan.recommendations, 1):
        lines += [f"### {i}. {r.title}", "", r.action, ""]
        shown = r.evidence[:8]
        more = f" и ещё {episodes(len(r.evidence) - 8)}" if len(r.evidence) > 8 else ""
        lines.append("Эпизоды: " + ", ".join(shown) + more)
        lines.append("")
    if video_url:
        lines += ["## Моменты для просмотра", ""]
        by_id = {finding_id(f, i): f for i, f in enumerate(findings, 1)}
        for fid, f in list(by_id.items())[:40]:
            lines.append(f"- {fid}: {f.text} — {moment_link(video_url, max(0, f.t - 6))}")
    lines += ["", "---", "Метрики — оценки по видимой части поля (камера ведёт игру); "
              "анализ по номерам игроков в этом отчёте не делается."]
    return "\n".join(lines)
