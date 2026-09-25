import tempfile
from pathlib import Path

from matchlens.roster import RosterError, load_roster

GOOD = """
ours: {name: A, color: [200, 30, 30], players: [{number: 1, position: GK}, {number: 9}]}
opp: {name: B, color: [30, 60, 200], players: [{number: 10, position: FW}]}
"""


def _load(text: str):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "roster.yaml"
        p.write_text(text, encoding="utf-8")
        return load_roster(p)


def _fails(text: str) -> bool:
    try:
        _load(text)
    except RosterError:
        return True
    return False


def test_loads_valid_roster():
    r = _load(GOOD)
    assert r.ours.get(1).position == "GK"
    assert r.opp.get(99) is None and r.opp.get(None) is None


def test_duplicate_number_rejected():
    assert _fails(GOOD.replace("{number: 9}", "{number: 1}"))


def test_bad_position_and_color_rejected():
    assert _fails(GOOD.replace("GK", "XX"))
    assert _fails(GOOD.replace("[200, 30, 30]", "[300, 0, 0]"))


def test_missing_section_rejected():
    assert _fails("ours: {name: A, color: [1, 2, 3]}")


def test_example_file_is_valid():
    root = Path(__file__).resolve().parents[1]
    r = load_roster(root / "data" / "roster.example.yaml")
    assert r.ours.players and r.opp.players
