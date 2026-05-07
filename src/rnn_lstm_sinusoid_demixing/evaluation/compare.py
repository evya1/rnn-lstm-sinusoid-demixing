"""Model comparison utilities."""

from typing import Any

from numpy.typing import NDArray


def compare_models(
    results: dict[str, dict[str, Any]],
) -> NDArray:
    """Summarise test MSE for each model into a comparison table.

    Args:
        results: Mapping model_name -> {'test_mse': float, ...}.

    Returns:
        Structured array or dict ready for display and saving.
    """
    raise NotImplementedError("Phase 7: compare.compare_models")
