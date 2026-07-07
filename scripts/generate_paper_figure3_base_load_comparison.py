#!/usr/bin/env python3
"""Generate manuscript Figure 3: base-load accuracy and runtime comparison."""

from __future__ import annotations

from manuscript_figure_generation import (
    generate_figure3_base_load_comparison,
    run_single_figure_cli,
)


if __name__ == "__main__":
    raise SystemExit(
        run_single_figure_cli(
            figure_number=3,
            description=__doc__ or "",
            generator=generate_figure3_base_load_comparison,
        )
    )
