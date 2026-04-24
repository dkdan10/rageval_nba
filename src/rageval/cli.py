"""Minimal CLI entry point for rageval (stub — full implementation in Milestone 8)."""

import click

import rageval


@click.group()
def main() -> None:
    """rageval — evaluation harness for hybrid RAG systems."""


@main.command()
def version() -> None:
    """Print the installed package version."""
    click.echo(f"rageval {rageval.__version__}")
