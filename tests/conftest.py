"""Shared test helper for image comparisons, and a summary of failures by source location.

assert_images_equal says which pixels differ and where. The hooks below add a section
"failures by location" after a failing run: for each failure, the line of the test that failed and the
deepest line inside src/projection/ that was running, which is usually where the fault is.
"""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src" / "projection"
TESTS = ROOT / "tests"
SHOWN = 5


def assert_images_equal(actual, expected, what: str = "pixels") -> None:
    """Fail with the number of differing pixels, their bounding box, and the first few with both values.

    Pseudo-code:
        1. different shapes: fail with both shapes
        2. places = every index where the arrays differ; none: pass
        3. fail with: count of places, the box from the lowest to the highest index on each axis,
           and the first SHOWN places as "(row, col): got X, expected Y"
    """
    __tracebackhide__ = True
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(f"{what}: shape {actual.shape}, expected {expected.shape}")
    places = np.argwhere(actual != expected)
    if not len(places):
        return
    box = ", ".join(f"{low}..{high}" for low, high in zip(places.min(axis=0), places.max(axis=0), strict=True))
    shown = "; ".join(
        f"{tuple(int(i) for i in place)}: got {actual[tuple(place)]}, expected {expected[tuple(place)]}"
        for place in places[:SHOWN]
    )
    more = f"; and {len(places) - SHOWN} more" if len(places) > SHOWN else ""
    raise AssertionError(f"{len(places)} of {actual.size} {what} differ, within [{box}]: {shown}{more}")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if report.failed and call.excinfo is not None:
        report.location_summary = _locate(call.excinfo)
    return report


def pytest_terminal_summary(terminalreporter):
    reports = [r for key in ("failed", "error") for r in terminalreporter.stats.get(key, [])
               if hasattr(r, "location_summary")]
    if not reports:
        return
    terminalreporter.section("failures by location")
    for report in reports:
        test_line, source_line, message = report.location_summary
        where = f"{test_line} -> {source_line}" if source_line else test_line
        terminalreporter.line(f"{where}  {report.head_line}")
        terminalreporter.line(f"    {message}")


def _locate(excinfo) -> tuple[str, str | None, str]:
    """(last test line, deepest src/projection line or None, first line of the message)."""
    test_line, source_line = "", None
    for entry in excinfo.traceback:
        path = Path(str(entry.path)).resolve()
        if entry.ishidden(excinfo):
            continue
        where = f"{path.relative_to(ROOT).as_posix()}:{entry.lineno + 1}" if path.is_relative_to(ROOT) else None
        if path.is_relative_to(TESTS):
            test_line = where or test_line
        elif path.is_relative_to(SOURCE):
            source_line = where
    lines = [line for line in str(excinfo.value).splitlines() if line.strip()]
    message = lines[0] if lines else excinfo.typename
    if not excinfo.errisinstance(AssertionError):
        message = f"{excinfo.typename}: {message}"
    return test_line, source_line, message
