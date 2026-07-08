"""Prop. 5 validation -- B2 variant: r_b + C > 0 on F_cap (reward_shift=150).

Thin wrapper around experiments.prop5_validation.run that points to the B2
config by default. All training, analysis and figure logic is in the parent
module; this file exists to (a) give run_all_flat.py a distinct module path and
(b) provide a convenient default config for standalone VM invocations.

Usage (VM):
    PYTHONUTF8=1 uv run python -m experiments.prop5_validation_b2.run \\
        --parallel 8 [--skip-existing] [--analyze-only]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Re-export hooks consumed by run_all_flat (build_jobs, _worker, AGENTS).
from experiments.prop5_validation.run import (  # noqa: F401
    build_jobs,
    _worker,
    AGENTS,
    run_cell,
    collect_results,
    make_analysis,
)
import experiments.prop5_validation.run as _parent_mod

_DEFAULT_CONFIG = _HERE / "config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prop. 5 B2 validation (reward_shift=150, r_b+C > 0 on F_cap)"
    )
    parser.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help="Config file (default: prop5_validation_b2/config.yaml)",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--analyze-only", action="store_true")
    cli = parser.parse_args()

    # Delegate to parent's main() by injecting the resolved argv.
    new_argv = [sys.argv[0], "--config", str(cli.config)]
    if cli.skip_existing:
        new_argv.append("--skip-existing")
    if cli.parallel > 1:
        new_argv.extend(["--parallel", str(cli.parallel)])
    if cli.analyze_only:
        new_argv.append("--analyze-only")

    saved_argv = sys.argv
    sys.argv = new_argv
    try:
        _parent_mod.main()
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    main()
