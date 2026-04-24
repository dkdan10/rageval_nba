"""Tests for the minimal CLI stub (src/rageval/cli.py)."""

from click.testing import CliRunner

from rageval.cli import main


def test_cli_help_exits_zero() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0


def test_cli_help_mentions_rageval() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert "rageval" in result.output.lower()


def test_cli_version_exits_zero() -> None:
    result = CliRunner().invoke(main, ["version"])
    assert result.exit_code == 0


def test_cli_version_prints_version_string() -> None:
    result = CliRunner().invoke(main, ["version"])
    assert "0.1.0" in result.output


def test_cli_version_help() -> None:
    result = CliRunner().invoke(main, ["version", "--help"])
    assert result.exit_code == 0
