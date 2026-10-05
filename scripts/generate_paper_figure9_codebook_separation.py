#!/usr/bin/env python3
"""Generate figure file figure9 (manuscript Figure 7): accuracy vs realised command-codebook separation (set-point band sweep)."""

from __future__ import annotations

from manuscript_figure_generation import (
    generate_figure9_codebook_separation,
    run_single_figure_cli,
)


if __name__ == "__main__":
    raise SystemExit(
        run_single_figure_cli(
            figure_number=9,
            description=__doc__ or "",
            generator=generate_figure9_codebook_separation,
        )
    )
