"""Этап 3: номера игроков. Одиночное чтение шумное, поэтому номер трека = взвешенное голосование.

Модель распознавания (OCR / классификатор) подключается через протокол NumberReader;
здесь — то, что превращает шум покадровых чтений в устойчивый ответ.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol


class NumberReader(Protocol):
    def read(self, crop_rgb) -> tuple[int, float] | None:
        """Вернуть (номер, уверенность 0..1) или None, если номера не видно."""
        ...


@dataclass
class VoteResult:
    number: int | None
    confidence: float  # доля веса победителя среди всех голосов
    votes: int


class NumberVoter:
    def __init__(self, min_votes: int = 3, min_margin: float = 1.5, min_conf: float = 0.3):
        self.min_votes, self.min_margin, self.min_conf = min_votes, min_margin, min_conf
        self._w: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._n: dict[int, int] = defaultdict(int)

    def add(self, track_id: int, number: int, conf: float) -> None:
        if conf < self.min_conf:
            return
        self._w[track_id][number] += conf
        self._n[track_id] += 1

    def result(self, track_id: int) -> VoteResult:
        weights = self._w.get(track_id)
        if not weights or self._n[track_id] < self.min_votes:
            return VoteResult(None, 0.0, self._n.get(track_id, 0))
        ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
        best_num, best_w = ranked[0]
        second_w = ranked[1][1] if len(ranked) > 1 else 0.0
        if second_w > 0 and best_w / second_w < self.min_margin:
            return VoteResult(None, best_w / sum(weights.values()), self._n[track_id])
        return VoteResult(best_num, best_w / sum(weights.values()), self._n[track_id])
