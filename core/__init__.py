"""Numerical core; backend-dependent public types are loaded on demand."""

__all__ = ["PairwiseFusionGraph"]


def __getattr__(name: str):
    if name != "PairwiseFusionGraph":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .fusion.types import PairwiseFusionGraph

    globals()[name] = PairwiseFusionGraph
    return PairwiseFusionGraph


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
