#!/usr/bin/env python3
"""Generate manuscript Figure 4: EV sampling sensitivity."""

from __future__ import annotations

from manuscript_figure_generation import (
    generate_figure4_sampling_sensitivity,
    run_single_figure_cli,
)


if __name__ == "__main__":
    raise SystemExit(
        run_single_figure_cli(
            figure_number=4,
            description=__doc__ or "",
            generator=generate_figure4_sampling_sensitivity,
        )
    )
