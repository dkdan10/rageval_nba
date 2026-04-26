from importlib import resources

from click.testing import CliRunner

import rageval
from rageval.cli import main


def test_package_importable() -> None:
    assert rageval.__version__ == "0.1.0"


def test_report_template_packaged() -> None:
    template = resources.files("rageval").joinpath("templates/report.html.j2")

    assert template.is_file()


def test_demo_suite_packaged() -> None:
    suite = resources.files("rageval").joinpath("examples/nba_test_suite.yaml")

    assert suite.is_file()


def test_demo_help_exits_zero() -> None:
    result = CliRunner().invoke(main, ["demo", "--help"])

    assert result.exit_code == 0
    assert "demo-report.html" in result.output
