#!/usr/bin/env python3
"""Generate manuscript Figure 5: posterior diagnostic."""

from __future__ import annotations

from manuscript_figure_generation import (
    generate_figure5_posterior_diagnostic,
    run_single_figure_cli,
)


if __name__ == "__main__":
    raise SystemExit(
        run_single_figure_cli(
            figure_number=5,
            description=__doc__ or "",
            generator=generate_figure5_posterior_diagnostic,
        )
    )
