"""Unit tests for the CLI index command."""

import pytest
from typer.testing import CliRunner

from specagent.cli import app

runner = CliRunner()


@pytest.mark.unit
def test_index_command_has_docs_dir_option():
    """specagent index --help shows --docs-dir option (not --download)."""
    result = runner.invoke(app, ["index", "--help"])
    assert result.exit_code == 0
    assert "--docs-dir" in result.output
    assert "--download" not in result.output


@pytest.mark.unit
def test_index_command_has_library_option():
    """specagent index --help shows --library option."""
    result = runner.invoke(app, ["index", "--help"])
    assert result.exit_code == 0
    assert "--library" in result.output


@pytest.mark.unit
def test_index_command_has_force_option():
    """specagent index --help shows --force option."""
    result = runner.invoke(app, ["index", "--help"])
    assert result.exit_code == 0
    assert "--force" in result.output
