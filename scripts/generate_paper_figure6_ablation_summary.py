#!/usr/bin/env python3
"""Generate figure file figure6 (manuscript Figure 8): ablation summary."""

from __future__ import annotations

from manuscript_figure_generation import (
    generate_figure6_ablation_summary,
    run_single_figure_cli,
)


if __name__ == "__main__":
    raise SystemExit(
        run_single_figure_cli(
            figure_number=6,
            description=__doc__ or "",
            generator=generate_figure6_ablation_summary,
        )
    )
