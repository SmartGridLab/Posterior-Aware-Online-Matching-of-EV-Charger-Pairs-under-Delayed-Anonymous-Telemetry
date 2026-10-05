#!/usr/bin/env python3
"""Generate figure file figure7 (manuscript Figure 5): impairment sweeps at the representative point."""

from __future__ import annotations

from manuscript_figure_generation import (
    generate_figure7_impairment_sweeps,
    run_single_figure_cli,
)


if __name__ == "__main__":
    raise SystemExit(
        run_single_figure_cli(
            figure_number=7,
            description=__doc__ or "",
            generator=generate_figure7_impairment_sweeps,
        )
    )
