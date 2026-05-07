"""Model comparison utilities."""


def compare_models(results: dict[str, float]) -> dict[str, float]:
    """Sort models by ascending test MSE.

    Args:
        results: Mapping model_name -> test_mse (scalar float).

    Returns:
        Dict sorted by ascending test MSE value.
    """
    return dict(sorted(results.items(), key=lambda item: item[1]))
