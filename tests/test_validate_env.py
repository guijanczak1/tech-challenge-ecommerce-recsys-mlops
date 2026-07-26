"""Testes de scripts/validate_env.py."""

from __future__ import annotations

import validate_env


def test_version_at_least_compares_major_minor() -> None:
    assert validate_env.version_at_least("2.13.0", "2.2") is True
    assert validate_env.version_at_least("2.2.5", "2.2") is True
    assert validate_env.version_at_least("1.0", "2.0") is False


def test_check_python_reports_current() -> None:
    result = validate_env.check_python()
    assert result.name == "python 3.12.x"
    assert result.detail  # traz a versão detectada


def test_check_package_missing_is_not_ok() -> None:
    result = validate_env.check_package("pacote_inexistente_xyz", "1.0", "fake")
    assert result.ok is False
    assert "não instalado" in result.detail


def test_run_checks_all_ok_in_configured_env() -> None:
    results = validate_env.run_checks()
    failed = [c.name for c in results if not c.ok]
    assert failed == [], f"checagens falharam: {failed}"
