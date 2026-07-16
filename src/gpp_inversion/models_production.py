"""Production-oriented TCN variants for sparse observations and physiology."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import TemporalConvNet


def _causal_observation_metadata(mask: torch.Tensor, no_observation_age: float):
    valid = mask > 0
    steps = mask.size(1)
    positions = torch.arange(steps, device=mask.device).view(1, -1)
    observed_positions = torch.where(valid, positions, torch.full_like(positions, -1))
    last = torch.cummax(observed_positions, dim=1).values
    age = positions - last
    age = torch.where(last >= 0, age, torch.full_like(age, int(no_observation_age)))
    counts = valid.sum(dim=1, keepdim=True)
    return valid, age.to(mask.dtype), counts.to(mask.dtype)


class ObservationAwareTCNGPP(nn.Module):
    """TCN forcing encoder with masked sparse-EPIC state attention."""

    def __init__(
        self,
        num_forcing_features: int,
        num_state_features: int,
        seq_len: int = 96,
        num_static: int = 2,
        time_feature_dim: int = 4,
        num_lc_classes: int | None = 13,
        lc_embed_dim: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        tcn_layers: int = 6,
        satellite_mask_index: int = 0,
        no_observation_age_hours: float = 240.0,
        use_endpoint_observation_age: bool = True,
        use_observation_count: bool = True,
        use_token_recency: bool = False,
        nonnegative_output: bool = False,
    ) -> None:
        super().__init__()
        if satellite_mask_index >= num_state_features:
            raise ValueError("satellite_mask_index exceeds state feature dimension")
        self.satellite_mask_index = satellite_mask_index
        self.seq_len = int(seq_len)
        self.no_observation_age_hours = float(no_observation_age_hours)
        self.use_endpoint_observation_age = bool(use_endpoint_observation_age)
        self.use_observation_count = bool(use_observation_count)
        self.use_token_recency = bool(use_token_recency)
        self.nonnegative_output = nonnegative_output
        self.tcn = TemporalConvNet(
            num_forcing_features, [d_model] * tcn_layers, kernel_size=3, dropout=dropout
        )
        self.lc_embedding = (
            nn.Embedding(num_lc_classes, lc_embed_dim)
            if num_lc_classes is not None else None
        )
        static_dim = num_static + (lc_embed_dim if self.lc_embedding is not None else 0)
        state_input_features = num_state_features + time_feature_dim + int(self.use_token_recency)
        self.state_projector = nn.Linear(state_input_features, d_model)
        self.static_projector = nn.Sequential(
            nn.Linear(static_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        summary_features = int(self.use_endpoint_observation_age) + int(self.use_observation_count)
        self.age_projector = nn.Linear(summary_features, d_model) if summary_features else None
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.state_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.no_observation_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.no_observation_token, std=0.02)
        self.state_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.forcing_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.fusion = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _static_context(self, x_static, x_lc):
        parts = [x_static[:, -1, :]]
        if self.lc_embedding is not None:
            if x_lc is None:
                raise ValueError("x_lc is required")
            parts.append(self.lc_embedding(x_lc[:, -1]))
        return self.static_projector(torch.cat(parts, dim=-1))

    def project_state_inputs(self, x_state, time_x):
        """Project state tokens with the same causal metadata used in inference."""
        state_inputs = [x_state, time_x]
        if self.use_token_recency:
            steps = x_state.size(1)
            recency = torch.arange(
                steps - 1, -1, -1, device=x_state.device, dtype=x_state.dtype
            ).view(1, steps, 1)
            recency = torch.log1p(recency) / math.log1p(max(1, steps - 1))
            state_inputs.append(recency.expand(x_state.size(0), -1, -1))
        return self.state_projector(torch.cat(state_inputs, dim=-1))

    def encode(self, x_forcing, x_state, time_x, x_static, x_lc):
        forcing_memory = self.tcn(x_forcing.transpose(1, 2)).transpose(1, 2)
        valid, age, counts = _causal_observation_metadata(
            x_state[:, :, self.satellite_mask_index], self.no_observation_age_hours
        )
        state_memory = self.project_state_inputs(x_state, time_x)
        all_missing = ~valid.any(dim=1)
        if all_missing.any():
            state_memory = state_memory.clone()
            valid = valid.clone()
            state_memory[all_missing, -1, :] = self.no_observation_token[
                0, 0
            ].to(dtype=state_memory.dtype)
            valid[all_missing, -1] = True
        state_memory = self.state_encoder(
            state_memory, src_key_padding_mask=~valid
        )
        static_context = self._static_context(x_static, x_lc)
        query = static_context
        summary_values = []
        if self.use_endpoint_observation_age:
            endpoint_age = age[:, -1:].clamp_max(self.no_observation_age_hours)
            summary_values.append(
                torch.log1p(endpoint_age) / math.log1p(self.no_observation_age_hours)
            )
        if self.use_observation_count:
            summary_values.append(counts / x_state.size(1))
        if self.age_projector is not None:
            query = query + self.age_projector(torch.cat(summary_values, dim=1))
        query_token = query[:, None, :]
        state_summary, _ = self.state_attention(
            query_token,
            state_memory,
            state_memory,
            key_padding_mask=~valid,
            need_weights=False,
        )
        forcing_summary, _ = self.forcing_attention(
            query_token, forcing_memory, forcing_memory, need_weights=False
        )
        return forcing_summary[:, 0], state_summary[:, 0], query

    def forward(self, x_forcing, x_state, time_x, x_static, x_lc):
        forcing, state, query = self.encode(x_forcing, x_state, time_x, x_static, x_lc)
        prediction = self.fusion(torch.cat([forcing, state, query], dim=-1)).squeeze(-1)
        return F.softplus(prediction) if self.nonnegative_output else prediction


class MultiscaleTCNGPP(ObservationAwareTCNGPP):
    """Observation-aware hourly model with an optional 30-day daily memory."""

    def __init__(self, *args, daily_context_features: int = 5, daily_context_hidden: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        d_model = self.static_projector[-1].out_features
        self.daily_encoder = nn.GRU(
            daily_context_features, daily_context_hidden, batch_first=True
        )
        self.daily_film = nn.Linear(daily_context_hidden, 2 * d_model)

    def forward(self, x_forcing, x_state, time_x, x_static, x_lc, daily_context):
        forcing, state, query = self.encode(x_forcing, x_state, time_x, x_static, x_lc)
        if daily_context is not None:
            _, hidden = self.daily_encoder(daily_context)
            gamma, beta = self.daily_film(hidden[-1]).chunk(2, dim=-1)
            forcing = forcing * (1.0 + torch.tanh(gamma)) + beta
        prediction = self.fusion(torch.cat([forcing, state, query], dim=-1)).squeeze(-1)
        return F.softplus(prediction) if self.nonnegative_output else prediction


class HybridLUETCNGPP(ObservationAwareTCNGPP):
    """Non-negative light-use-efficiency baseline plus a neural residual."""

    def __init__(
        self,
        *args,
        sw_in_index: int = 0,
        vpd_index: int = 4,
        ta_index: int = 5,
        swc_index: int = 7,
        **kwargs,
    ) -> None:
        kwargs["nonnegative_output"] = False
        super().__init__(*args, **kwargs)
        self.indices = (sw_in_index, vpd_index, ta_index, swc_index)
        classes = self.lc_embedding.num_embeddings if self.lc_embedding is not None else 1
        self.log_epsilon = nn.Embedding(classes, 1)
        nn.init.constant_(self.log_epsilon.weight, -3.0)
        self.temperature = nn.Parameter(torch.tensor([10.0, 20.0]))
        self.vpd_half = nn.Parameter(torch.tensor(10.0))
        self.swc_half = nn.Parameter(torch.tensor(20.0))
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("forcing_offset", torch.zeros(1, 1, 1), persistent=True)
        self.register_buffer("forcing_scale", torch.ones(1, 1, 1), persistent=True)

    def configure_scaling(self, offset, scale) -> None:
        device = self.forcing_offset.device
        offset = torch.as_tensor(
            offset, dtype=torch.float32, device=device
        ).view(1, 1, -1)
        scale = torch.as_tensor(
            scale, dtype=torch.float32, device=device
        ).view(1, 1, -1)
        self.forcing_offset = offset
        self.forcing_scale = scale

    def forward(self, x_forcing, x_state, time_x, x_static, x_lc):
        forcing, state, query = self.encode(x_forcing, x_state, time_x, x_static, x_lc)
        residual = self.fusion(torch.cat([forcing, state, query], dim=-1)).squeeze(-1)
        raw = x_forcing * self.forcing_scale + self.forcing_offset
        sw = raw[:, -1, self.indices[0]]
        vpd = raw[:, -1, self.indices[1]]
        ta = raw[:, -1, self.indices[2]]
        swc = raw[:, -1, self.indices[3]]
        sorted_temperature = torch.sort(self.temperature).values
        low = sorted_temperature[0]
        high = sorted_temperature[1]
        temp_stress = torch.sigmoid((ta - low) / 3.0) * torch.sigmoid((high - ta) / 3.0)
        vpd_stress = 1.0 / (1.0 + F.relu(vpd) / F.softplus(self.vpd_half))
        swc_stress = torch.sigmoid((swc - F.softplus(self.swc_half)) / 5.0)
        if self.lc_embedding is not None:
            epsilon = F.softplus(self.log_epsilon(x_lc[:, -1]).squeeze(-1))
        else:
            epsilon = F.softplus(self.log_epsilon.weight[0, 0]).expand_as(sw)
        lue = F.relu(sw) * epsilon * temp_stress * vpd_stress * swc_stress
        return F.softplus(lue + self.residual_scale * residual)
