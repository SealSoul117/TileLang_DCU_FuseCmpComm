"""Generalized physical-layout fusion support.

This package entry point exports only the pure-Python geometry contract, so it
remains importable on hosts without a working HIP runtime.  DSL generators,
GPU preparation/crop kernels, and the executable runner are imported directly
from their respective modules when needed.
"""

from .contract import FusionPlan, ceildiv, make_plan

__all__ = ["FusionPlan", "ceildiv", "make_plan"]
