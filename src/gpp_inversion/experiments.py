"""Model factory for the unified experiment pipeline."""

from __future__ import annotations

from .config import FeatureColumns, ModelConfig, ModelKind
from .models import LayerNormLSTMGPP, TCNTransformerCrossAttention
from .models_architecture import ModernTCNGPP, TimeMixerPlusPlusGPP, TimeXerGPP
from .models_irregular import (
    NeuralCDECrossAttentionGPP,
    TimeAwareMambaTransformerCrossAttention,
)
from .models_production import (
    HybridLUETCNGPP,
    MultiscaleTCNGPP,
    ObservationAwareTCNGPP,
)


def build_model(
    config: ModelConfig,
    features: FeatureColumns,
    *,
    seq_len: int,
    time_feature_dim: int,
):
    common = {
        "num_forcing_features": len(features.forcing),
        "num_state_features": features.state_dimension,
        "seq_len": seq_len,
        "num_static": len(features.static),
        "time_feature_dim": time_feature_dim,
        "d_model": config.d_model,
        "nhead": config.nhead,
        "dropout": config.dropout,
        "dim_feedforward": config.dim_feedforward,
    }
    if config.kind is ModelKind.TCN:
        return TCNTransformerCrossAttention(
            **common,
            num_lc_classes=(
                config.num_land_cover_classes if features.land_cover else None
            ),
            lc_embed_dim=config.land_cover_embedding_dim,
            num_layers=config.num_layers,
            tcn_layers=config.tcn_layers,
            normalized_tcn=config.normalized_tcn,
            cross_attention_residual=config.cross_attention_residual,
            lag_encoding=config.lag_encoding,
            static_context_mode=config.static_context_mode,
            state_norm_first=config.state_norm_first,
            cross_fusion_mode=config.cross_fusion_mode,
            cross_direction=config.cross_direction,
            temporal_pooling=config.temporal_pooling,
        )
    if config.kind in {
        ModelKind.TCN_OBSERVATION_AWARE,
        ModelKind.TCN_MULTISCALE,
        ModelKind.HYBRID_LUE_TCN,
    }:
        production_common = {
            **common,
            "num_lc_classes": (
                config.num_land_cover_classes if features.land_cover else None
            ),
            "lc_embed_dim": config.land_cover_embedding_dim,
            "num_layers": config.num_layers,
            "tcn_layers": config.tcn_layers,
            "satellite_mask_index": config.satellite_mask_index,
            "no_observation_age_hours": config.no_observation_age_hours,
            "use_endpoint_observation_age": config.use_endpoint_observation_age,
            "use_observation_count": config.use_observation_count,
            "use_token_recency": config.use_token_recency,
            "nonnegative_output": config.nonnegative_output,
        }
        if config.kind is ModelKind.TCN_OBSERVATION_AWARE:
            return ObservationAwareTCNGPP(**production_common)
        if config.kind is ModelKind.TCN_MULTISCALE:
            return MultiscaleTCNGPP(
                **production_common,
                daily_context_features=config.daily_context_features,
                daily_context_hidden=config.daily_context_hidden,
            )
        production_common.pop("nonnegative_output")
        return HybridLUETCNGPP(**production_common)
    if config.kind is ModelKind.LSTM:
        return LayerNormLSTMGPP(
            num_forcing_features=len(features.forcing),
            num_state_features=features.state_dimension,
            time_feature_dim=time_feature_dim,
            num_static=len(features.static),
            num_lc_classes=(
                config.num_land_cover_classes if features.land_cover else None
            ),
            lc_embed_dim=config.land_cover_embedding_dim,
            d_model=config.d_model,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            dropout=config.dropout,
        )
    architecture_common = {
        **common,
        "num_lc_classes": (
            config.num_land_cover_classes if features.land_cover else None
        ),
        "lc_embed_dim": config.land_cover_embedding_dim,
        "patch_length": config.patch_length,
        "patch_stride": config.patch_stride,
    }
    if config.kind is ModelKind.TIMEXER:
        return TimeXerGPP(**architecture_common, num_layers=config.num_layers)
    if config.kind is ModelKind.MODERN_TCN:
        return ModernTCNGPP(
            **architecture_common,
            num_blocks=config.modern_tcn_blocks,
            large_kernel=config.modern_large_kernel,
            small_kernel=config.modern_small_kernel,
        )
    if config.kind is ModelKind.TIME_MIXER_PP:
        return TimeMixerPlusPlusGPP(
            **architecture_common,
            num_blocks=config.mixer_blocks,
            top_k=config.mixer_top_k,
        )
    if features.land_cover is None or config.num_land_cover_classes is None:
        raise ValueError(f"{config.kind.value} requires land-cover IDs")
    if config.kind is ModelKind.MAMBA:
        return TimeAwareMambaTransformerCrossAttention(
            **common,
            num_lc_classes=config.num_land_cover_classes,
            lc_embed_dim=config.land_cover_embedding_dim,
            num_transformer_layers=config.num_layers,
            num_mamba_layers=config.num_mamba_layers,
            mamba_d_state=config.mamba_d_state,
            mamba_d_conv=config.mamba_d_conv,
            mamba_expand=config.mamba_expand,
            use_native_mamba=config.use_native_mamba,
        )
    return NeuralCDECrossAttentionGPP(
        **common,
        num_lc_classes=config.num_land_cover_classes,
        lc_embed_dim=config.land_cover_embedding_dim,
        cde_layers=config.cde_layers,
        cde_vector_field_dim=config.cde_vector_field_dim,
        increment_scale=config.increment_scale,
    )
