"""Evolution-grounded tumor simulation and benchmark-truth generation."""

from .config import (
    CopyNumberEvolutionConfig,
    TumorSimulationConfig,
)


def __getattr__(name: str):
    if name == "simulate_tumor":
        from .generator import simulate_tumor

        globals()[name] = simulate_tumor
        return simulate_tumor
    if name == "validate_generated_tumor_directory":
        from .output import validate_generated_tumor_directory

        globals()[name] = validate_generated_tumor_directory
        return validate_generated_tumor_directory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CopyNumberEvolutionConfig",
    "TumorSimulationConfig",
    "simulate_tumor",
    "validate_generated_tumor_directory",
]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
