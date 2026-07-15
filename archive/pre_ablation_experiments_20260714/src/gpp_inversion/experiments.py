"""Model factory for the unified experiment pipeline."""

from __future__ import annotations

from .config import FeatureColumns, ModelConfig, ModelKind
from .models import TCNTransformerCrossAttention
from .models_irregular import (
    NeuralCDECrossAttentionGPP,
    TimeAwareMambaTransformerCrossAttention,
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
        "num_state_features": len(features.state),
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
