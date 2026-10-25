"""Tests for token display formatting (including @0 on timed places)."""

from cpnpy.cpn.cpn_imp import Token
from cpnpy.visualization.visualizer import format_token


def test_format_token_legacy_omits_zero_timestamp():
    assert format_token(Token(1, 0)) == "1"
    assert format_token(Token(1, 5)) == "1@5"


def test_format_token_timed_includes_zero_timestamp():
    assert format_token(Token(1, 0), include_timestamp=True) == "1@0"
    assert format_token(Token(1, 42), include_timestamp=True) == "1@42"
    # Dict values are HTML-escaped for Graphviz labels.
    assert format_token(Token({"a": 1}, 0), include_timestamp=True).endswith("@0")
    assert "a" in format_token(Token({"a": 1}, 0), include_timestamp=True)


def test_format_token_untimed_never_includes_timestamp():
    assert format_token(Token(1, 0), include_timestamp=False) == "1"
    assert format_token(Token(1, 42), include_timestamp=False) == "1"
