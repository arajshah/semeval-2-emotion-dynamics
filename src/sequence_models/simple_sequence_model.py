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


class TransformerSequenceRegressor(nn.Module):
    """
    Transformer-based sequence regressor for predicting ΔV/ΔA from embedding sequences.

    Inputs:
        - inputs:  (B, L, D) float tensor of embeddings (padded on the left)
        - lengths: (B,) long tensor of actual sequence lengths (<= L)

    Output:
        - preds: (B, 2) predictions for [ΔV, ΔA]
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        if hidden_dim != embedding_dim:
            self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.fc = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            inputs:  (B, L, D)
            lengths: (B,) with actual sequence lengths, or None

        Returns:
            preds: (B, 2)
        """
        B, L, _ = inputs.shape

        x = self.input_proj(inputs)

        if lengths is not None:
            mask = torch.ones((B, L), dtype=torch.bool, device=inputs.device)
            for i in range(B):
                valid_len = int(lengths[i].item())
                if valid_len <= 0:
                    continue
                mask[i, L - valid_len :] = False
        else:
            mask = None

        encoded = self.encoder(x, src_key_padding_mask=mask)

        if lengths is not None:
            pooled = torch.zeros((B, self.hidden_dim), device=encoded.device)
            for i in range(B):
                valid_len = int(lengths[i].item())
                if valid_len <= 0:
                    continue
                start = L - valid_len
                pooled[i] = encoded[i, start:].mean(dim=0)
        else:
            pooled = encoded.mean(dim=1)

        preds = self.fc(pooled)
        return preds


class SequenceStateRegressor(nn.Module):
    """
    GRU-based sequence regressor with numeric state/history features.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_features: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim + num_features)
        self.fc1 = nn.Linear(hidden_dim + num_features, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        x_seq: torch.Tensor,
        lengths: torch.Tensor,
        x_num: torch.Tensor,
    ) -> torch.Tensor:
        lengths_cpu = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x_seq, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        seq_repr = h_n[-1]
        x = torch.cat([seq_repr, x_num], dim=1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)

