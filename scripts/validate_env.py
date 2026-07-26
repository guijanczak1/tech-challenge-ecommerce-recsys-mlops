"""Valida o ambiente de desenvolvimento/execução do projeto.

Checa a versão do Python, a presença e versão mínima das dependências e se o
pacote ``recsys`` e a configuração carregam. Uso:

    python scripts/validate_env.py

Sai com código != 0 se algum item obrigatório falhar.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass

# import-name -> (versão mínima, nome amigável/distribuição)
REQUIRED_PACKAGES: dict[str, tuple[str, str]] = {
    "torch": ("2.2", "torch"),
    "sklearn": ("1.5", "scikit-learn"),
    "mlflow": ("2.14", "mlflow"),
    "dvc": ("3.50", "dvc"),
    "pandas": ("2.2", "pandas"),
    "numpy": ("1.26", "numpy"),
    "pydantic": ("2.7", "pydantic"),
    "pydantic_settings": ("2.3", "pydantic-settings"),
    "yaml": ("6.0", "pyyaml"),
}
MIN_PY = (3, 12)
MAX_PY_EXCLUSIVE = (3, 13)


@dataclass
class Check:
    """Resultado de uma checagem de ambiente."""

    name: str
    ok: bool
    detail: str = ""


def _to_tuple(version: str) -> tuple[int, ...]:
    parts = version.split(".")[:2]
    return tuple(int(p) for p in parts if p.isdigit())


def version_at_least(actual: str, minimum: str) -> bool:
    """Compara major.minor: ``actual >= minimum``."""
    return _to_tuple(actual) >= _to_tuple(minimum)


def check_python() -> Check:
    """Confere se a versão do Python está em [3.12, 3.13)."""
    current = sys.version_info[:3]
    ok = MIN_PY <= current[:2] < MAX_PY_EXCLUSIVE
    return Check("python 3.12.x", ok, ".".join(map(str, current)))


def check_package(import_name: str, minimum: str, dist: str) -> Check:
    """Confere se um pacote está instalado e atende à versão mínima."""
    try:
        module = importlib.import_module(import_name)
    except ImportError:
        return Check(dist, False, "não instalado")
    actual = getattr(module, "__version__", "0")
    ok = version_at_least(actual, minimum)
    return Check(dist, ok, f"{actual} (min {minimum})")


def check_project_import() -> Check:
    """Confere se o pacote ``recsys`` importa."""
    try:
        importlib.import_module("recsys")
    except ImportError as exc:
        return Check("import recsys", False, str(exc))
    return Check("import recsys", True)


def check_config_loads() -> Check:
    """Confere se a configuração (env + params.yaml) carrega."""
    try:
        from recsys.config import load_config

        cfg = load_config()
    except Exception as exc:  # noqa: BLE001 — diagnóstico, não propaga
        return Check("load_config()", False, str(exc)[:120])
    return Check("load_config()", True, f"seed={cfg.seed}, model={cfg.model.name}")


def run_checks() -> list[Check]:
    """Roda todas as checagens e devolve os resultados."""
    results = [check_python()]
    for import_name, (minimum, dist) in REQUIRED_PACKAGES.items():
        results.append(check_package(import_name, minimum, dist))
    results.append(check_project_import())
    results.append(check_config_loads())
    return results


def _print(results: list[Check]) -> None:
    for check in results:
        mark = "+" if check.ok else "x"
        line = f" {mark} {check.name}"
        print(line + (f" - {check.detail}" if check.detail else ""))


def main() -> int:
    """Roda a validação, imprime o relatório e retorna o código de saída."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    results = run_checks()
    print("=== Validação de Ambiente ===")
    _print(results)
    failed = [c.name for c in results if not c.ok]
    print(f"\n{'FALHOU' if failed else 'OK'}" + (f" — pendências: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
