from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_coverage(path: Path) -> tuple[float, str]:
    """Return coverage percentage (0-100) and default color."""
    try:
        root = ET.parse(path).getroot()
        rate = float(root.attrib.get("line-rate", 0.0))
    except (FileNotFoundError, ET.ParseError, ValueError) as exc:
        raise RuntimeError(f"Impossible de lire coverage.xml ({path}): {exc}") from exc

    percent = rate * 100
    if percent >= 90:
        color = "22c55e"  # green
    elif percent >= 75:
        color = "facc15"  # amber
    else:
        color = "ef4444"  # red
    return percent, color


def parse_pytest_report(path: Path) -> tuple[int, int, int, int, str]:
    """
    Return total tests, passed, skipped, failed+errors and badge color.

    JUnit report root may be <testsuite> or <testsuites>.
    """
    try:
        tree = ET.parse(path)
    except (FileNotFoundError, ET.ParseError) as exc:
        raise RuntimeError(f"Impossible de lire le rapport Pytest ({path}): {exc}") from exc

    root = tree.getroot()
    suites = []
    if root.tag == "testsuite":
        suites = [root]
    else:
        suites = root.findall("testsuite")

    total = passed = skipped = failed_errors = 0
    for suite in suites:
        suite_tests = int(suite.attrib.get("tests", 0))
        suite_failures = int(suite.attrib.get("failures", 0))
        suite_errors = int(suite.attrib.get("errors", 0))
        suite_skipped = int(suite.attrib.get("skipped", 0))

        total += suite_tests
        skipped += suite_skipped
        failed_errors += suite_failures + suite_errors
        passed += suite_tests - suite_failures - suite_errors - suite_skipped

    color = "22c55e" if failed_errors == 0 else "ef4444"
    return total, passed, skipped, failed_errors, color


def fmt_percent(value: float) -> str:
    rounded = round(value, 1)
    if math.isclose(rounded, round(value)):
        return f"{round(value)}%"
    return f"{rounded}%"


def make_badge(label: str, message: str, color: str) -> dict[str, str | int]:
    return {
        "schemaVersion": 1,
        "label": label,
        "message": message,
        "color": color,
    }


def write_badge(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def generate_badges(coverage_path: Path, junit_path: Path, output_dir: Path) -> None:
    coverage_pct, coverage_color = parse_coverage(coverage_path)
    total, passed, skipped, failed_errors, tests_color = parse_pytest_report(junit_path)

    coverage_badge = make_badge(
        label="couverture",
        message=fmt_percent(coverage_pct),
        color=coverage_color,
    )
    skipped_suffix = f" ({skipped} skipped)" if skipped else ""
    if failed_errors == 0:
        tests_message = f"{passed} passed{skipped_suffix}"
    else:
        tests_message = f"{failed_errors} failing / {total} tests"

    tests_badge = make_badge(
        label="tests",
        message=tests_message,
        color=tests_color,
    )

    write_badge(output_dir / "coverage.json", coverage_badge)
    write_badge(output_dir / "tests.json", tests_badge)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Génère les badges JSON pour Shields.io.")
    parser.add_argument(
        "--coverage",
        default="coverage.xml",
        type=Path,
        help="Chemin vers coverage.xml (défaut: coverage.xml)",
    )
    parser.add_argument(
        "--junit",
        default="pytest-report.xml",
        type=Path,
        help="Chemin vers le rapport JUnit généré par Pytest.",
    )
    parser.add_argument(
        "--output",
        default=Path("badges"),
        type=Path,
        help="Dossier de sortie pour les fichiers JSON.",
    )
    args = parser.parse_args(argv)

    generate_badges(args.coverage, args.junit, Path(args.output))


if __name__ == "__main__":
    main(sys.argv[1:])
