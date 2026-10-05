#!/usr/bin/env python3
"""Generate figure file figure8 (manuscript Figure 6): accuracy vs EV-session count under per-session response heterogeneity."""

from __future__ import annotations

from manuscript_figure_generation import (
    generate_figure8_heterogeneous_response,
    run_single_figure_cli,
)


if __name__ == "__main__":
    raise SystemExit(
        run_single_figure_cli(
            figure_number=8,
            description=__doc__ or "",
            generator=generate_figure8_heterogeneous_response,
        )
    )
