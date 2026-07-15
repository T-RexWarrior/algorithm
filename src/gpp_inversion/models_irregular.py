"""Mamba and discrete Neural CDE models extracted from the latest notebooks."""

from __future__ import annotations

import torch
import torch.nn as nn

try:  # Optional and frequently unavailable on Windows.
    from mamba_ssm import Mamba as _NativeMamba
except Exception:  # Import can fail for missing compiled CUDA extensions.
    _NativeMamba = None


class MambaResidualBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if _NativeMamba is None:
            raise RuntimeError("mamba_ssm is not available")
        self.norm = nn.LayerNorm(d_model)
        self.mamba = _NativeMamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.dropout(self.mamba(self.norm(values)))


class MambaLikeBlock(nn.Module):
    """Portable gated causal-convolution fallback used by the Notebook."""

    def __init__(
        self,
        d_model: int,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.inner_dim = int(d_model * expand)
        self.norm = nn.LayerNorm(d_model)
        self.input_projection = nn.Linear(d_model, self.inner_dim * 2)
        self.depthwise_convolution = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=d_conv,
            groups=self.inner_dim,
            padding=d_conv - 1,
        )
        self.activation = nn.SiLU()
        self.output_projection = nn.Linear(self.inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        sequence_length = values.shape[1]
        projected, gate_values = self.input_projection(self.norm(values)).chunk(2, dim=-1)
        convolved = self.depthwise_convolution(projected.transpose(1, 2))
        convolved = convolved[:, :, :sequence_length].transpose(1, 2)
        gated = self.activation(convolved) * torch.sigmoid(gate_values)
        return residual + self.dropout(self.output_projection(gated))


class TimeAwareMambaEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_native_mamba: bool | None = None,
    ) -> None:
        super().__init__()
        if use_native_mamba is None:
            use_native_mamba = _NativeMamba is not None
        if use_native_mamba and _NativeMamba is None:
            raise RuntimeError("Native Mamba requested but mamba_ssm cannot be imported")
        self.uses_native_mamba = use_native_mamba
        self.input_projection = nn.Linear(input_dim, d_model)
        block_type = MambaResidualBlock if use_native_mamba else MambaLikeBlock
        self.blocks = nn.ModuleList(
            [
                block_type(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                if use_native_mamba
                else block_type(
                    d_model=d_model,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.input_projection(values)
        for block in self.blocks:
            values = block(values)
        return self.final_norm(values)


class TimeAwareMambaTransformerCrossAttention(nn.Module):
    def __init__(
        self,
        num_forcing_features: int,
        num_state_features: int,
        seq_len: int,
        num_static: int = 2,
        num_lc_classes: int = 13,
        lc_embed_dim: int = 8,
        time_feature_dim: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        num_mamba_layers: int = 4,
        dim_feedforward: int = 128,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        dropout: float = 0.1,
        use_native_mamba: bool | None = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.forcing_encoder = TimeAwareMambaEncoder(
            input_dim=num_forcing_features + time_feature_dim,
            d_model=d_model,
            num_layers=num_mamba_layers,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=dropout,
            use_native_mamba=use_native_mamba,
        )
        self.land_cover_embedding = nn.Embedding(num_lc_classes, lc_embed_dim)
        self.state_projection = nn.Linear(
            num_state_features + num_static + lc_embed_dim, d_model
        )
        self.time_projection = nn.Linear(time_feature_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.state_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.fusion_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, x_forcing, x_state, time_x, x_static, x_lc):
        forcing_memory = self.forcing_encoder(torch.cat([x_forcing, time_x], dim=-1))
        land_cover = self.land_cover_embedding(x_lc)
        state_values = self.state_projection(
            torch.cat([x_state, x_static, land_cover], dim=-1)
        ) + self.time_projection(time_x)
        state_memory = self.state_encoder(state_values)
        cross_values, _ = self.cross_attention(
            query=state_memory, key=forcing_memory, value=forcing_memory
        )
        gate = self.fusion_gate(torch.cat([cross_values, state_memory], dim=-1))
        fused = gate * cross_values + (1.0 - gate) * state_memory
        return self.regressor(fused[:, -1, :]).squeeze(-1)


class CDEVectorField(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        input_dim: int,
        mlp_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim * input_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.network(hidden).view(-1, self.hidden_dim, self.input_dim)


class DiscreteNeuralCDEEncoder(nn.Module):
    """Euler approximation used in the Neural CDE Notebook."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        context_dim: int = 0,
        vector_field_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        increment_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if increment_scale <= 0:
            raise ValueError("increment_scale must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.increment_scale = increment_scale
        self.initial_network = nn.Sequential(
            nn.Linear(input_dim + context_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.vector_fields = nn.ModuleList(
            [
                CDEVectorField(hidden_dim, input_dim, vector_field_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, path: torch.Tensor, context: torch.Tensor | None = None):
        initial = path[:, 0, :]
        if context is not None:
            initial = torch.cat([initial, context], dim=-1)
        hidden = self.initial_network(initial)
        memory = [hidden]
        increments = path[:, 1:, :] - path[:, :-1, :]
        for step in range(increments.shape[1]):
            increment = increments[:, step, :]
            for vector_field, layer_norm in zip(self.vector_fields, self.layer_norms):
                field = vector_field(hidden)
                delta = torch.einsum("bhi,bi->bh", field, increment)
                delta = torch.tanh(delta / self.increment_scale)
                hidden = layer_norm(hidden + self.dropout(delta))
            memory.append(hidden)
        return torch.stack(memory, dim=1)


class NeuralCDECrossAttentionGPP(nn.Module):
    def __init__(
        self,
        num_forcing_features: int,
        num_state_features: int,
        seq_len: int,
        num_static: int = 2,
        num_lc_classes: int = 13,
        lc_embed_dim: int = 8,
        time_feature_dim: int = 7,
        d_model: int = 64,
        nhead: int = 4,
        cde_layers: int = 1,
        cde_vector_field_dim: int = 128,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        increment_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.land_cover_embedding = nn.Embedding(num_lc_classes, lc_embed_dim)
        context_dim = num_static + lc_embed_dim
        self.forcing_cde = DiscreteNeuralCDEEncoder(
            num_forcing_features + time_feature_dim,
            d_model,
            context_dim,
            cde_vector_field_dim,
            cde_layers,
            dropout,
            increment_scale,
        )
        self.state_cde = DiscreteNeuralCDEEncoder(
            num_state_features + time_feature_dim,
            d_model,
            context_dim,
            cde_vector_field_dim,
            cde_layers,
            dropout,
            increment_scale,
        )
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.fusion_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.context_projection = nn.Sequential(
            nn.Linear(context_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.post_fusion = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, x_forcing, x_state, time_x, x_static, x_lc):
        context = torch.cat(
            [x_static[:, -1, :], self.land_cover_embedding(x_lc[:, -1])], dim=-1
        )
        forcing_memory = self.forcing_cde(
            torch.cat([x_forcing, time_x], dim=-1), context=context
        )
        state_memory = self.state_cde(
            torch.cat([x_state, time_x], dim=-1), context=context
        )
        cross_values, _ = self.cross_attention(
            query=state_memory, key=forcing_memory, value=forcing_memory
        )
        gate = self.fusion_gate(torch.cat([cross_values, state_memory], dim=-1))
        fused = gate * cross_values + (1.0 - gate) * state_memory
        last = fused[:, -1, :] + self.context_projection(context)
        last = last + self.post_fusion(last)
        return self.regressor(last).squeeze(-1)


# Historical class names remain importable while new code uses PEP 8 names.
TimeAwareMamba_Transformer_CrossAttention = TimeAwareMambaTransformerCrossAttention
NeuralCDE_CrossAttention_GPP = NeuralCDECrossAttentionGPP
