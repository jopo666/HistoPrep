from histoprep.helpers._time import format_seconds, get_ETC


def test_format_seconds():
    for dst, src in {
        "59s": 59,
        "1m:0s": 60,
        "1h:0m:1s": 60 * 60 + 1,
        "1d 0h:0m:1s": 60 * 60 * 24 + 1,
        "1y 0d 0h:0m:1s": 60 * 60 * 24 * 365.25 + 1,
    }.items():
        assert dst == format_seconds(src)


def test_ETC():
    times = [1, 2]
    assert "ETC: 3s" == get_ETC(times, 2)
    assert "ETC: 4.5s" == get_ETC(times, 3)
    assert "" == get_ETC([], 2)
