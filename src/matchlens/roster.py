"""Составы команд: номер -> игрок. Файл roster.yaml заполняет тренер."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

POSITIONS = {"GK", "DF", "MF", "FW", "UT"}  # UT — универсал


class RosterError(ValueError):
    pass


@dataclass
class Player:
    number: int
    name: str = ""
    position: str = ""


@dataclass
class TeamRoster:
    name: str
    color: tuple[int, int, int]  # RGB основной цвет формы
    players: dict[int, Player] = field(default_factory=dict)

    def get(self, number: int | None) -> Player | None:
        return self.players.get(number) if number is not None else None


@dataclass
class Roster:
    ours: TeamRoster
    opp: TeamRoster

    def team(self, key: str) -> TeamRoster:
        return {"ours": self.ours, "opp": self.opp}[key]


def _parse_team(key: str, raw: dict) -> TeamRoster:
    if not isinstance(raw, dict):
        raise RosterError(f"{key}: ожидается словарь")
    color = raw.get("color")
    if not (isinstance(color, list) and len(color) == 3 and all(0 <= c <= 255 for c in color)):
        raise RosterError(f"{key}.color: нужен список [R, G, B] 0..255")
    team = TeamRoster(name=str(raw.get("name", key)), color=tuple(color))
    for item in raw.get("players", []) or []:
        num = item.get("number")
        if not isinstance(num, int) or not 0 <= num <= 99:
            raise RosterError(f"{key}: некорректный номер {num!r}")
        if num in team.players:
            raise RosterError(f"{key}: номер {num} повторяется")
        pos = item.get("position", "")
        if pos and pos not in POSITIONS:
            raise RosterError(f"{key} #{num}: позиция {pos!r} не из {sorted(POSITIONS)}")
        team.players[num] = Player(num, str(item.get("name", "")), pos)
    return team


def load_roster(path: str | Path) -> Roster:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for key in ("ours", "opp"):
        if key not in raw:
            raise RosterError(f"В файле нет секции {key!r}")
    return Roster(ours=_parse_team("ours", raw["ours"]), opp=_parse_team("opp", raw["opp"]))
