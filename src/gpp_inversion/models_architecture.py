"""Windows-native architecture candidates for the third-round GPP study."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticContext(nn.Module):
    """Encode location and land cover once per window."""

    def __init__(self, num_static, num_lc_classes, lc_embed_dim, d_model):
        super().__init__()
        self.lc_embedding = (
            nn.Embedding(num_lc_classes, lc_embed_dim)
            if num_lc_classes is not None
            else None
        )
        width = num_static + (lc_embed_dim if self.lc_embedding is not None else 0)
        self.projector = nn.Sequential(
            nn.Linear(width, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, x_static, x_lc):
        parts = [x_static[:, -1, :]]
        if self.lc_embedding is not None:
            if x_lc is None:
                raise ValueError("x_lc is required when land-cover embedding is enabled")
            parts.append(self.lc_embedding(x_lc[:, -1]))
        return self.projector(torch.cat(parts, dim=-1))


class PreNormCrossBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.state_norm = nn.LayerNorm(d_model)
        self.state_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.state_ffn_norm = nn.LayerNorm(d_model)
        self.state_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout),
        )
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.query_ffn_norm = nn.LayerNorm(d_model)
        self.query_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout),
        )

    def forward(self, patches, query, exogenous, *, diagnostics=False):
        normalized = self.state_norm(patches)
        attended, _ = self.state_attention(normalized, normalized, normalized, need_weights=False)
        patches = patches + attended
        patches = patches + self.state_ffn(self.state_ffn_norm(patches))
        memory = torch.cat([patches, exogenous], dim=1)
        crossed, weights = self.cross_attention(
            self.query_norm(query), self.memory_norm(memory), self.memory_norm(memory),
            need_weights=diagnostics, average_attn_weights=False,
        )
        query = query + crossed
        query = query + self.query_ffn(self.query_ffn_norm(query))
        return patches, query, weights


class TimeXerGPP(nn.Module):
    """Patch/global-token exogenous forecaster inspired by TimeXer."""

    def __init__(
        self, num_forcing_features, num_state_features, seq_len,
        num_static=2, time_feature_dim=4, num_lc_classes=13, lc_embed_dim=8,
        d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
        dropout=0.1, patch_length=8, patch_stride=4,
    ):
        super().__init__()
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.state_patch = nn.Conv1d(
            num_state_features + time_feature_dim, d_model,
            kernel_size=patch_length, stride=patch_stride,
        )
        self.forcing_tokenizer = nn.Linear(seq_len, d_model)
        self.forcing_variable_embedding = nn.Parameter(
            torch.empty(1, num_forcing_features, d_model)
        )
        nn.init.normal_(self.forcing_variable_embedding, std=0.02)
        self.static_context = StaticContext(
            num_static, num_lc_classes, lc_embed_dim, d_model
        )
        self.gpp_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.gpp_token, std=0.02)
        self.blocks = nn.ModuleList(
            PreNormCrossBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def _forward_impl(self, x_forcing, x_state, time_x, x_static, x_lc, diagnostics):
        dynamic_state = torch.cat([x_state, time_x], dim=-1).transpose(1, 2)
        patches = self.state_patch(dynamic_state).transpose(1, 2)
        exogenous = self.forcing_tokenizer(x_forcing.transpose(1, 2))
        exogenous = exogenous + self.forcing_variable_embedding
        query = self.gpp_token.expand(x_state.size(0), -1, -1)
        query = query + self.static_context(x_static, x_lc).unsqueeze(1)
        entropies = []
        for block in self.blocks:
            patches, query, weights = block(
                patches, query, exogenous, diagnostics=diagnostics
            )
            if diagnostics:
                p = weights.float().clamp_min(1e-8)
                entropies.append(-(p * p.log()).sum(dim=-1).mean(dim=(1, 2)))
        prediction = self.regressor(self.output_norm(query[:, 0])).squeeze(-1)
        values = {
            "cross_attention_entropy_by_layer": torch.stack(entropies, dim=1)
        } if diagnostics else None
        return (prediction, values) if diagnostics else prediction

    def forward(self, x_forcing, x_state, time_x, x_static, x_lc=None):
        return self._forward_impl(x_forcing, x_state, time_x, x_static, x_lc, False)

    def forward_with_diagnostics(self, x_forcing, x_state, time_x, x_static, x_lc=None):
        return self._forward_impl(x_forcing, x_state, time_x, x_static, x_lc, True)


class ModernTCNBlock(nn.Module):
    def __init__(self, d_model, large_kernel, small_kernel, expansion, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.large = nn.Conv1d(
            d_model, d_model, large_kernel, padding=large_kernel // 2,
            groups=d_model,
        )
        self.small = nn.Conv1d(
            d_model, d_model, small_kernel, padding=small_kernel // 2,
            groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, 1)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * expansion), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        normalized = self.norm1(x).transpose(1, 2)
        temporal = self.pointwise(self.large(normalized) + self.small(normalized))
        x = x + temporal.transpose(1, 2)
        return x + self.ffn(self.norm2(x))


class ModernTCNGPP(nn.Module):
    """Large-kernel depthwise temporal model inspired by ModernTCN."""

    def __init__(
        self, num_forcing_features, num_state_features, seq_len,
        num_static=2, time_feature_dim=4, num_lc_classes=13, lc_embed_dim=8,
        d_model=64, dropout=0.1, patch_length=8, patch_stride=4,
        num_blocks=4, large_kernel=13, small_kernel=3, expansion=2,
        **_,
    ):
        super().__init__()
        dynamic_dim = num_forcing_features + num_state_features + time_feature_dim
        self.patch_embed = nn.Conv1d(
            dynamic_dim, d_model, patch_length, stride=patch_stride
        )
        self.blocks = nn.ModuleList(
            ModernTCNBlock(d_model, large_kernel, small_kernel, expansion, dropout)
            for _ in range(num_blocks)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.static_context = StaticContext(
            num_static, num_lc_classes, lc_embed_dim, d_model
        )
        self.regressor = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _forward_impl(self, x_forcing, x_state, time_x, x_static, x_lc, diagnostics):
        dynamic = torch.cat([x_forcing, x_state, time_x], dim=-1).transpose(1, 2)
        patches = self.patch_embed(dynamic).transpose(1, 2)
        activations = []
        for block in self.blocks:
            patches = block(patches)
            if diagnostics:
                activations.append(patches.float().square().mean(dim=(1, 2)).sqrt())
        pooled = self.final_norm(patches).mean(dim=1)
        static = self.static_context(x_static, x_lc)
        prediction = self.regressor(torch.cat([pooled, static], dim=-1)).squeeze(-1)
        values = {"block_rms_activation": torch.stack(activations, dim=1)} if diagnostics else None
        return (prediction, values) if diagnostics else prediction

    def forward(self, x_forcing, x_state, time_x, x_static, x_lc=None):
        return self._forward_impl(x_forcing, x_state, time_x, x_static, x_lc, False)

    def forward_with_diagnostics(self, x_forcing, x_state, time_x, x_static, x_lc=None):
        return self._forward_impl(x_forcing, x_state, time_x, x_static, x_lc, True)


class SpectralMixerBlock(nn.Module):
    def __init__(self, d_model, expansion, dropout, top_k):
        super().__init__()
        self.top_k = top_k
        self.norm1 = nn.LayerNorm(d_model)
        self.period_projection = nn.Linear(d_model, d_model)
        self.trend_projection = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * expansion), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, *, diagnostics=False):
        normalized = self.norm1(x)
        spectrum = torch.fft.rfft(normalized.float(), dim=1)
        energy = spectrum.abs().mean(dim=(0, 2))
        if energy.numel() > 1:
            selectable = energy.clone()
            selectable[0] = -1
            indices = torch.topk(selectable, min(self.top_k, selectable.numel())).indices
        else:
            indices = torch.zeros(1, dtype=torch.long, device=x.device)
        diagnostic_indices = indices
        if indices.numel() < self.top_k:
            diagnostic_indices = F.pad(indices, (0, self.top_k - indices.numel()), value=-1)
        mask = torch.zeros_like(spectrum)
        mask[:, indices, :] = spectrum[:, indices, :]
        periodic = torch.fft.irfft(mask, n=x.size(1), dim=1).to(x.dtype)
        trend = F.avg_pool1d(
            normalized.transpose(1, 2), kernel_size=3, stride=1, padding=1
        ).transpose(1, 2)
        x = x + self.period_projection(periodic) + self.trend_projection(trend)
        x = x + self.channel_mlp(self.norm2(x))
        return x, diagnostic_indices if diagnostics else None


class TimeMixerPlusPlusGPP(nn.Module):
    """Three-scale patch/period/trend mixer inspired by TimeMixer++."""

    def __init__(
        self, num_forcing_features, num_state_features, seq_len,
        num_static=2, time_feature_dim=4, num_lc_classes=13, lc_embed_dim=8,
        d_model=64, dropout=0.1, patch_length=8, patch_stride=4,
        num_blocks=2, top_k=3, expansion=2, **_,
    ):
        super().__init__()
        dynamic_dim = num_forcing_features + num_state_features + time_feature_dim
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.scales = (1, 2, 4)
        self.patch_embeds = nn.ModuleList(
            nn.Conv1d(dynamic_dim, d_model, patch_length, stride=patch_stride)
            for _ in self.scales
        )
        self.scale_embeddings = nn.Parameter(torch.empty(len(self.scales), 1, 1, d_model))
        nn.init.normal_(self.scale_embeddings, std=0.02)
        self.blocks = nn.ModuleList(
            SpectralMixerBlock(d_model, expansion, dropout, top_k)
            for _ in range(num_blocks)
        )
        self.scale_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1)
        )
        self.static_context = StaticContext(
            num_static, num_lc_classes, lc_embed_dim, d_model
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.regressor = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _downsample(self, x, factor):
        if factor == 1:
            return x
        return F.avg_pool1d(x.transpose(1, 2), factor, factor).transpose(1, 2)

    def _forward_impl(self, x_forcing, x_state, time_x, x_static, x_lc, diagnostics):
        dynamic = torch.cat([x_forcing, x_state, time_x], dim=-1)
        summaries = []
        frequency_indices = []
        scale_rms = []
        for scale_index, (factor, patch_embed) in enumerate(zip(self.scales, self.patch_embeds)):
            scaled = self._downsample(dynamic, factor).transpose(1, 2)
            if scaled.size(-1) < self.patch_length:
                scaled = F.pad(scaled, (0, self.patch_length - scaled.size(-1)))
            patches = patch_embed(scaled).transpose(1, 2)
            patches = patches + self.scale_embeddings[scale_index]
            scale_frequencies = []
            for block in self.blocks:
                patches, indices = block(patches, diagnostics=diagnostics)
                if diagnostics:
                    scale_frequencies.append(indices)
            summaries.append(patches.mean(dim=1))
            if diagnostics:
                frequency_indices.append(torch.stack(scale_frequencies))
                scale_rms.append(patches.float().square().mean(dim=(1, 2)).sqrt())
        stacked = torch.stack(summaries, dim=1)
        weights = torch.softmax(self.scale_gate(stacked).squeeze(-1), dim=1)
        pooled = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        static = self.static_context(x_static, x_lc)
        prediction = self.regressor(
            torch.cat([self.output_norm(pooled), static], dim=-1)
        ).squeeze(-1)
        values = None
        if diagnostics:
            values = {
                "scale_weights": weights,
                "scale_rms_activation": torch.stack(scale_rms, dim=1),
                "top_frequency_indices": torch.stack(frequency_indices),
            }
        return (prediction, values) if diagnostics else prediction

    def forward(self, x_forcing, x_state, time_x, x_static, x_lc=None):
        return self._forward_impl(x_forcing, x_state, time_x, x_static, x_lc, False)

    def forward_with_diagnostics(self, x_forcing, x_state, time_x, x_static, x_lc=None):
        return self._forward_impl(x_forcing, x_state, time_x, x_static, x_lc, True)
