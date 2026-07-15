from __future__ import annotations

from dataclasses import replace
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gpp_inversion.config import (
    EvaluationConfig, ExperimentConfig, FeatureColumns, LossKind, ModelConfig,
    ModelKind, TimeFeatureMode, TrainingConfig, WindowConfig,
)
from gpp_inversion.experiments import build_model
from gpp_inversion.pipeline import run_experiment


FEATURES = FeatureColumns(
    forcing=tuple(f"forcing_{index}" for index in range(9)),
    state=("state_0", "state_1", "state_2"),
    static=("Lat", "Long"),
    target="GPP",
    time="date",
    land_cover="Veg_ID",
)


def inputs(batch=2, length=96):
    generator = torch.Generator().manual_seed(123)
    return (
        torch.randn(batch, length, 9, generator=generator),
        torch.randn(batch, length, 3, generator=generator),
        torch.randn(batch, length, 4, generator=generator),
        torch.randn(batch, length, 2, generator=generator),
        torch.ones(batch, length, dtype=torch.long),
    )


def base_config(**changes):
    config = ModelConfig(
        kind=ModelKind.TCN,
        d_model=16,
        nhead=4,
        dim_feedforward=32,
        num_layers=1,
        tcn_layers=2,
        dropout=0.0,
        num_land_cover_classes=13,
    )
    return replace(config, **changes)


def test_default_tcn_strict_state_dict_compatibility():
    first = build_model(base_config(), FEATURES, seq_len=96, time_feature_dim=4)
    second = build_model(base_config(), FEATURES, seq_len=96, time_feature_dim=4)
    second.load_state_dict(first.state_dict(), strict=True)
    keys = tuple(first.state_dict())
    assert not any("gpp_query" in key or "alpha_attn" in key for key in keys)


def test_zero_initialized_gates_preserve_state_path():
    model = build_model(
        base_config(cross_fusion_mode="zero_init_gated"),
        FEATURES, seq_len=96, time_feature_dim=4,
    ).eval()
    first = list(inputs())
    second = list(first)
    second[0] = first[0] + 100.0
    with torch.no_grad():
        first_prediction, diagnostics = model.forward_with_diagnostics(*first)
        second_prediction = model(*second)
    torch.testing.assert_close(first_prediction, second_prediction)
    assert diagnostics["alpha_attn"].item() == 0.0
    assert diagnostics["alpha_ffn"].item() == 0.0


def test_bidirectional_attention_and_query_pooling_shapes():
    bidirectional = build_model(
        base_config(
            cross_fusion_mode="bidirectional_gated",
            cross_direction="bidirectional",
        ),
        FEATURES, seq_len=96, time_feature_dim=4,
    )
    prediction, diagnostics = bidirectional.forward_with_diagnostics(*inputs())
    assert prediction.shape == (2,)
    assert diagnostics["forcing_to_state_attention_entropy"].shape == (2,)

    pooling = build_model(
        base_config(temporal_pooling="gpp_query"),
        FEATURES, seq_len=96, time_feature_dim=4,
    )
    prediction, diagnostics = pooling.forward_with_diagnostics(*inputs())
    assert prediction.shape == (2,)
    assert diagnostics["pooling_weights"].shape == (2, 4, 96)


def test_patch_boundaries_and_multiscale_lengths():
    timexer = build_model(
        replace(base_config(), kind=ModelKind.TIMEXER, patch_length=8, patch_stride=4),
        FEATURES, seq_len=96, time_feature_dim=4,
    )
    patch_count = timexer.state_patch(torch.randn(2, 7, 96)).shape[-1]
    assert patch_count == 23

    mixer = build_model(
        replace(base_config(), kind=ModelKind.TIME_MIXER_PP),
        FEATURES, seq_len=96, time_feature_dim=4,
    )
    dynamic = torch.randn(2, 16, 96)
    lengths = []
    for factor, embed in zip(mixer.scales, mixer.patch_embeds):
        scaled = mixer._downsample(dynamic.transpose(1, 2), factor).transpose(1, 2)
        lengths.append(embed(scaled).shape[-1])
    assert lengths == [23, 11, 5]


def test_new_architectures_forward_diagnostics_and_parameter_cap():
    cap = int(226_537 * 1.5)
    for kind in (ModelKind.TIMEXER, ModelKind.MODERN_TCN, ModelKind.TIME_MIXER_PP):
        config = ModelConfig(kind=kind, num_land_cover_classes=13)
        model = build_model(config, FEATURES, seq_len=96, time_feature_dim=4).eval()
        with torch.no_grad():
            first, diagnostic = model.forward_with_diagnostics(*inputs())
            second = model(*inputs())
        assert first.shape == (2,)
        assert diagnostic
        torch.testing.assert_close(first, second)
        assert sum(parameter.numel() for parameter in model.parameters()) <= cap


def test_pre_ln_has_final_norm():
    model = build_model(
        base_config(state_norm_first=True), FEATURES, seq_len=96, time_feature_dim=4
    )
    assert model.transformer_encoder.norm is not None
    assert model.transformer_encoder.layers[0].norm_first


def test_each_new_architecture_small_end_to_end_artifacts():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        small_features = FeatureColumns(
            forcing=("forcing",), state=("state",), static=("Lat", "Long"),
            target="GPP", time="date", land_cover="Veg_ID",
        )
        for station, offset in (("TRAIN", 0.0), ("VAL", 1.0)):
            rows = 12
            pd.DataFrame({
                "date": pd.date_range("2026-01-01", periods=rows, freq="h"),
                "forcing": np.linspace(0, 1, rows) + offset,
                "state": np.linspace(1, 2, rows),
                "Lat": np.full(rows, 30.0), "Long": np.full(rows, 120.0),
                "Veg_ID": np.ones(rows, dtype=int),
                "GPP": np.linspace(0.5, 3.0, rows) + offset,
            }).to_csv(root / f"{station}.csv", index=False)
        for kind in (ModelKind.TIMEXER, ModelKind.MODERN_TCN, ModelKind.TIME_MIXER_PP):
            output = root / kind.value
            config = ExperimentConfig(
                data_dir=root, output_dir=output,
                train_sites=("TRAIN",), val_sites=("VAL",), test_sites=(),
                features=small_features,
                window=WindowConfig(
                    seq_len=8, time_features=TimeFeatureMode.CYCLIC,
                    require_regular=True, max_gap_hours=1, max_span_hours=7,
                ),
                model=ModelConfig(
                    kind=kind, d_model=8, nhead=2, dim_feedforward=16,
                    num_layers=1, num_land_cover_classes=3,
                    land_cover_embedding_dim=2, dropout=0.0,
                    patch_length=4, patch_stride=2,
                    modern_tcn_blocks=1, mixer_blocks=1, mixer_top_k=2,
                ),
                loss=LossKind.MSE,
                training=TrainingConfig(
                    batch_size=2, epochs=1, patience=1, seed=7,
                    resume=False, optimizer="adamw", weight_decay=1e-4,
                ),
                evaluation=EvaluationConfig(
                    save_predictions=True, save_plots=False,
                    minimum_target=None, evaluate_test=False,
                ),
            )
            result = run_experiment(config)
            assert len(result["config_hash"]) == 64
            assert (output / "experiment_manifest.json").exists()
            assert (output / "evaluation" / "val_predictions.csv").exists()
            assert (output / "evaluation" / "val_metrics_by_station.csv").exists()
            assert (output / "evaluation" / "val_metrics_high_target.json").exists()
            assert (output / "architecture_profile.json").exists()
            assert (output / "architecture_diagnostics.npz").exists()
