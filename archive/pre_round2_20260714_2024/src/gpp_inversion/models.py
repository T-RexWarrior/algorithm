"""TCN/Transformer GPP model and independently switchable ablation variants."""

from __future__ import annotations

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class ChannelLayerNorm(nn.Module):
    """Apply LayerNorm to channels while preserving ``[B, C, T]`` layout."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
        *,
        normalized: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.GELU() if normalized else nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.GELU() if normalized else nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.residual_norm = (
            ChannelLayerNorm(n_outputs) if normalized else nn.Identity()
        )
        self.relu = nn.GELU() if normalized else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(self.residual_norm(out + residual))


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        *,
        normalized: bool = False,
    ):
        super().__init__()
        layers = []
        for index, out_channels in enumerate(num_channels):
            dilation = 2**index
            in_channels = num_inputs if index == 0 else num_channels[index - 1]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout,
                    normalized=normalized,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCN_Transformer_CrossAttention(nn.Module):
    def __init__(
        self,
        num_forcing_features: int,
        num_state_features: int,
        seq_len: int,
        num_static: int = 2,
        time_feature_dim: int = 4,
        num_lc_classes: int | None = None,
        lc_embed_dim: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        *,
        tcn_layers: int = 6,
        normalized_tcn: bool = False,
        cross_attention_residual: bool = False,
        lag_encoding: str = "none",
    ):
        super().__init__()
        if lag_encoding not in {"none", "continuous", "embedding"}:
            raise ValueError(f"Unsupported lag encoding: {lag_encoding}")
        self.seq_len = seq_len
        self.cross_attention_residual = cross_attention_residual
        self.lag_encoding = lag_encoding

        self.tcn = TemporalConvNet(
            num_inputs=num_forcing_features,
            num_channels=[d_model] * tcn_layers,
            kernel_size=3,
            dropout=dropout,
            normalized=normalized_tcn,
        )
        self.lc_embedding = (
            nn.Embedding(num_lc_classes, lc_embed_dim)
            if num_lc_classes is not None
            else None
        )
        combined_state_dim = num_state_features + num_static
        if self.lc_embedding is not None:
            combined_state_dim += lc_embed_dim
        self.state_linear = nn.Linear(combined_state_dim, d_model)
        self.time_projector = nn.Linear(time_feature_dim, d_model)

        if lag_encoding == "embedding":
            self.lag_projector = nn.Embedding(seq_len, d_model)
        elif lag_encoding == "continuous":
            self.lag_projector = nn.Sequential(
                nn.Linear(1, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.lag_projector = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        if cross_attention_residual:
            self.cross_dropout = nn.Dropout(dropout)
            self.cross_norm1 = nn.LayerNorm(d_model)
            self.cross_norm2 = nn.LayerNorm(d_model)
            self.cross_ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout),
            )
            self.fusion_gate = nn.Linear(2 * d_model, d_model)

        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )

    def _lag_features(self, length: int, device: torch.device) -> torch.Tensor | int:
        if self.lag_encoding == "none":
            return 0
        if length > self.seq_len:
            raise ValueError(
                f"Input length {length} exceeds configured seq_len {self.seq_len}"
            )
        if self.lag_encoding == "embedding":
            lag_ids = torch.arange(length - 1, -1, -1, device=device)
            return self.lag_projector(lag_ids).unsqueeze(0)
        relative_age = torch.linspace(
            1.0, 0.0, length, device=device
        ).view(1, length, 1)
        return self.lag_projector(relative_age)

    def forward(
        self,
        x_forcing: torch.Tensor,
        x_state: torch.Tensor,
        time_x: torch.Tensor,
        x_static: torch.Tensor,
        x_lc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        f_met_memory = self.tcn(x_forcing.transpose(1, 2)).transpose(1, 2)

        state_parts = [x_state, x_static]
        if self.lc_embedding is not None:
            if x_lc is None:
                raise ValueError(
                    "x_lc is required when land-cover embedding is enabled"
                )
            state_parts.append(self.lc_embedding(x_lc))
        x_s_emb = self.state_linear(torch.cat(state_parts, dim=-1))
        state_input = (
            x_s_emb
            + self.time_projector(time_x)
            + self._lag_features(x_state.size(1), x_state.device)
        )
        f_state_global = self.transformer_encoder(state_input)

        attention_output, _ = self.cross_attention(
            query=f_state_global,
            key=f_met_memory,
            value=f_met_memory,
            need_weights=False,
        )
        if self.cross_attention_residual:
            attention_residual = self.cross_norm1(
                f_state_global + self.cross_dropout(attention_output)
            )
            attention_residual = self.cross_norm2(
                attention_residual + self.cross_ffn(attention_residual)
            )
            gate = torch.sigmoid(
                self.fusion_gate(
                    torch.cat([f_state_global, attention_residual], dim=-1)
                )
            )
            fused_features = (
                gate * attention_residual + (1.0 - gate) * f_state_global
            )
        else:
            fused_features = attention_output

        return self.regressor(fused_features[:, -1, :]).squeeze(-1)


TCNTransformerCrossAttention = TCN_Transformer_CrossAttention
