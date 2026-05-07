"""Model-specific input tensor preparation.

The dataset builder returns raw numpy arrays (inputs, selectors, targets).
After conversion to tensors, these functions format them for each model family:

    FC:       cat([window, selector], dim=-1)  →  (B, W + C)
    RNN/LSTM: interleave sigma + selector at each timestep  →  (B, W, 1 + C)
"""

import torch


def prepare_fc_input(
    windows: torch.Tensor,
    selectors: torch.Tensor,
) -> torch.Tensor:
    """Concatenate composite windows and one-hot selectors for the FC model.

    Args:
        windows:   (batch, window_size)  — noisy composite windows.
        selectors: (batch, num_components) — one-hot selector vectors.

    Returns:
        Tensor of shape (batch, window_size + num_components).
    """
    return torch.cat([windows, selectors], dim=-1)


def prepare_seq_input(
    windows: torch.Tensor,
    selectors: torch.Tensor,
) -> torch.Tensor:
    """Interleave composite signal and selector at every timestep for RNN/LSTM.

    Each timestep feature vector is [sigma_t, c_1, c_2, c_3, c_4], matching
    the layout documented in CLAUDE.md §6 and PRD_model_comparison.md.

    Args:
        windows:   (batch, seq_len)      — noisy composite windows.
        selectors: (batch, num_components) — one-hot selector vectors.

    Returns:
        Tensor of shape (batch, seq_len, 1 + num_components).
    """
    seq_len = windows.shape[1]
    sigma = windows.unsqueeze(-1)                              # (B, W, 1)
    selector_exp = selectors.unsqueeze(1).expand(-1, seq_len, -1)  # (B, W, C)
    return torch.cat([sigma, selector_exp], dim=-1)            # (B, W, 1+C)
