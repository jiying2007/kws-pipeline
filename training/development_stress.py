#!/usr/bin/env python3
from __future__ import annotations

from development_stress_impl import (
    apply_development_stress_scene,
    build_development_stress_plan as _build_development_stress_plan,
)


def build_development_stress_plan(
    config: dict,
    axes: dict,
    *,
    split: str,
    positive_scene_count: int,
) -> dict:
    plan = _build_development_stress_plan(
        config,
        axes,
        split=split,
        positive_scene_count=positive_scene_count,
    )
    # Keep the renderer summary field compact while the implementation also
    # exposes baseline/resulting support and pairwise before/after for tests.
    plan["planned_support"] = dict(plan["resulting_support"])
    return plan


__all__ = ["apply_development_stress_scene", "build_development_stress_plan"]
