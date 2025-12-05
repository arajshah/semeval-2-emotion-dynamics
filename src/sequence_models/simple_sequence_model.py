from __future__ import annotations

import torch
import torch.nn as nn


class SimpleSequenceRegressor(nn.Module):
    """
    LSTM-based sequence regressor for predicting ΔV/ΔA from embedding sequences.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 2)  # output ΔV, ΔA

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        inputs:  (B, L, D)
        lengths: (B,) actual sequence lengths (for packing), or None
        Returns:
            preds: (B, 2)
        """
        if lengths is not None:
            lengths_cpu = lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                inputs,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(inputs)

        last_hidden = h_n[-1]  # (B, hidden_dim)
        preds = self.fc(last_hidden)
        return preds

