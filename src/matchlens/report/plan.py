"""Этап 7: план улучшений «без тренировок». Сначала детерминированный каркас, потом LLM.

Принцип: рекомендация без ссылки на эпизод не допускается. LLM получает только JSON находок
и обязан ссылаться на их id (см. build_llm_prompt).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from ..ingest.prepare import format_ts
from ..schemas import Finding

# Что можно сделать без тренировки: установка, расстановка, ролевая задача, сигнал.
ADVICE: dict[str, tuple[str, str]] = {
    "loss_own_third": (
        "Безопасный выход из обороны",
        "Установка перед игрой: в своей трети — первый пас вперёд-в-сторону или длинный, "
        "без дриблинга. Назначить страхующего рядом с каждым, кто получает мяч спиной к чужим "
        "воротам. Ответственный по сигналу «свои» — центральный защитник.",
    ),
    "line_gap": (
        "Компактность линий",
        "Установка: при потере мяча ближайшие трое сжимаются к мячу, линия защиты поднимается "
        "синхронно с полузащитой. Ориентир для капитана: между линиями не более ~20 м.",
    ),
}
DEFAULT_ADVICE = ("Разбор эпизода", "Разобрать эпизод на видеопросмотре перед игрой.")


@dataclass
class Recommendation:
    title: str
    action: str
    evidence: list[str] = field(default_factory=list)  # ссылки на находки (id)
    numbers: list[int] = field(default_factory=list)
    priority: int = 1


@dataclass
class ImprovementPlan:
    recommendations: list[Recommendation]
    caveat: str = (
        "Метрики — оценки по видимой части поля; номера, не распознанные уверенно, "
        "не приписываются игрокам."
    )


def episodes(n: int) -> str:
    """1 эпизод, 2 эпизода, 5 эпизодов."""
    if 10 <= n % 100 <= 20 or n % 10 in (0, 5, 6, 7, 8, 9):
        word = "эпизодов"
    elif n % 10 == 1:
        word = "эпизод"
    else:
        word = "эпизода"
    return f"{n} {word}"


def finding_id(f: Finding, idx: int) -> str:
    return f"F{idx:03d}@{format_ts(f.t)[:8]}"


def build_plan(findings: list[Finding]) -> ImprovementPlan:
    groups: dict[str, list[tuple[str, Finding]]] = defaultdict(list)
    for i, f in enumerate(findings, 1):
        groups[f.kind].append((finding_id(f, i), f))
    recs: list[Recommendation] = []
    for kind, items in groups.items():
        title, action = ADVICE.get(kind, DEFAULT_ADVICE)
        numbers = sorted({n for _, f in items for n in f.numbers})
        recs.append(Recommendation(
            title=f"{title} ({episodes(len(items))})",
            action=action,
            evidence=[fid for fid, _ in items],
            numbers=numbers,
            priority=max(f.severity for _, f in items),
        ))
    recs.sort(key=lambda r: (-r.priority, -len(r.evidence)))
    return ImprovementPlan(recs)


def render_markdown(plan: ImprovementPlan) -> str:
    lines = ["# План улучшений на следующую игру (без дополнительных тренировок)", ""]
    if not plan.recommendations:
        lines.append("Значимых проблем не найдено.")
    for i, r in enumerate(plan.recommendations, 1):
        lines += [f"## {i}. {r.title}", "", r.action, ""]
        if r.numbers:
            lines.append("Игроки: " + ", ".join(f"№{n}" for n in r.numbers))
        lines.append("Эпизоды: " + ", ".join(r.evidence))
        lines.append("")
    lines += ["---", plan.caveat]
    return "\n".join(lines)


def build_llm_prompt(findings: list[Finding], stats: dict) -> str:
    """Запрос к языковой модели: только факты из JSON, ссылки на эпизоды обязательны."""
    payload = {
        "stats": stats,
        "findings": [
            {"id": finding_id(f, i), **asdict(f)} for i, f in enumerate(findings, 1)
        ],
    }
    return (
        "Ты помощник тренера. По данным ниже составь план улучшений на следующий матч, "
        "который выполняется БЕЗ дополнительных тренировок: установки, расстановка, "
        "индивидуальные задачи по номерам, сигналы, замены.\n"
        "Правила: используй только данные из JSON; к каждой рекомендации приложи id эпизодов; "
        "если данных мало — так и скажи; не выдумывай игроков и события.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
