from matchlens.ingest.prepare import (
    build_cut_cmd,
    build_proxy_cmd,
    chunk_ranges,
    format_ts,
    parse_fps,
    parse_ts,
)


def test_parse_and_format_ts_roundtrip():
    assert parse_ts("1:02:03.5") == 3723.5
    assert parse_ts("02:03") == 123.0
    assert parse_ts("95") == 95.0
    assert format_ts(3723.5) == "01:02:03.500"
    assert parse_ts(format_ts(59.999)) == 59.999


def test_parse_ts_rejects_garbage():
    for bad in ("", "a:b", "1:2:3:4", "-5"):
        try:
            parse_ts(bad)
        except ValueError:
            continue
        raise AssertionError(f"должно было упасть: {bad!r}")


def test_parse_fps():
    assert parse_fps("25/1") == 25.0
    assert abs(parse_fps("30000/1001") - 29.97) < 0.01
    assert parse_fps("0/0") == 0.0


def test_proxy_cmd_has_scale_fps_and_no_audio():
    cmd = build_proxy_cmd("in.mp4", "out.mp4", height=720, fps=25, start=10, end=70)
    assert cmd[cmd.index("-ss") + 1] == "00:00:10.000"
    assert cmd[cmd.index("-to") + 1] == "00:01:10.000"
    assert cmd[cmd.index("-vf") + 1] == "scale=-2:720,fps=25"
    assert "-an" in cmd and cmd[-1] == "out.mp4"


def test_cut_cmd_copy_vs_reencode():
    assert "copy" in build_cut_cmd("a.mp4", "b.mp4", 5, 15)
    assert "libx264" in build_cut_cmd("a.mp4", "b.mp4", 5, 15, reencode=True)
    try:
        build_cut_cmd("a.mp4", "b.mp4", 15, 5)
    except ValueError:
        return
    raise AssertionError("end <= start должен отклоняться")


def test_chunk_ranges_cover_duration_with_overlap():
    r = chunk_ranges(700, 300, overlap=10)
    assert r[0] == (0.0, 300.0)
    assert r[1][0] == 290.0
    assert r[-1][1] == 700
    assert all(b > a for a, b in r)
