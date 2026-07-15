"""Export a self-contained, hash-verified model package for global inference."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, ModelKind
from .contracts import FeatureContract, FeatureSpec, sha256_file
from .data import ScalingStats
from .experiments import build_model


def _unit(name: str) -> str:
    return {
        "SW_IN_F": "W m-2", "SW_IN_POT": "W m-2", "CO2_F_MDS": "ppm",
        "P_F": "mm h-1", "VPD_F": "hPa", "TA_F": "degC",
        "TS_F_MDS_1": "degC", "SWC_F_MDS_1": "percent", "WS_F": "m s-1",
        "Lat": "degree", "Long": "degree", "GPP_DT_VUT_REF": "umol CO2 m-2 s-1",
    }.get(name, "1" if "Band" in name or "Mask" in name else "degree")


def contract_from_config(config: ExperimentConfig) -> FeatureContract:
    forcing = tuple(
        FeatureSpec(name, _unit(name), "ERA5-Land/CO2 raster", "hour ending at timestamp", "contract-defined", "invalid grid cell")
        for name in config.features.forcing
    )
    state = tuple(
        FeatureSpec(name, _unit(name), "DSCOVR EPIC", "causal observed hourly slot", "nearest within 15 km", "mask=0; values=0; no forward fill")
        for name in config.features.state
    )
    static = tuple(
        FeatureSpec(name, _unit(name), "grid", "static", "cell center", "invalid")
        for name in config.features.static
    )
    target = FeatureSpec(config.features.target, _unit(config.features.target), "model", "hour ending at timestamp", "0.1 degree", "NaN plus quality flag")
    return FeatureContract(
        forcing=forcing, state=state, static=static, target=target,
        history_hours=config.window.seq_len,
    )


def export_model_package(
    config_path: str | Path,
    checkpoint_path: str | Path,
    scaler_path: str | Path,
    destination: str | Path,
    *,
    split_hash: str,
) -> Path:
    config = ExperimentConfig.from_json(config_path)
    if config.window.seq_len != 96:
        raise ValueError("Production packages require seq_len=96")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    scaler = ScalingStats.load(scaler_path)
    model = build_model(
        config.model, config.features,
        seq_len=96, time_feature_dim=config.window.time_feature_dim,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    if hasattr(model, "configure_scaling"):
        model.configure_scaling(scaler.forcing_offset, scaler.forcing_scale)
    model.eval()
    batch = 2
    inputs = (
        torch.zeros(batch, 96, len(config.features.forcing)),
        torch.zeros(batch, 96, config.features.state_dimension),
        torch.zeros(batch, 96, config.window.time_feature_dim),
        torch.zeros(batch, 96, len(config.features.static)),
        torch.zeros(batch, 96, dtype=torch.long),
    )
    production_kinds = {
        ModelKind.TCN_OBSERVATION_AWARE,
        ModelKind.TCN_MULTISCALE,
        ModelKind.HYBRID_LUE_TCN,
    }
    if config.model.kind in production_kinds:
        traced = torch.jit.script(model)
        input_arity = 6 if config.model.kind is ModelKind.TCN_MULTISCALE else 5
    else:
        traced = torch.jit.trace(model, inputs, strict=False)
        input_arity = 5
    scripted_path = destination / "model_scripted.pt"
    traced.save(str(scripted_path))
    checkpoint_copy = destination / "checkpoint.pth"
    scaler_copy = destination / "scaler.npz"
    shutil.copy2(checkpoint_path, checkpoint_copy)
    shutil.copy2(scaler_path, scaler_copy)
    global_scaler = destination / "global_scalers.npz"
    if scaler.method.value != "zscore":
        raise ValueError("Global production packages currently require zscore scaling")
    np.savez(
        global_scaler,
        feat_mean_f=np.asarray(scaler.forcing_offset, dtype=np.float32),
        feat_std_f=np.asarray(scaler.forcing_scale, dtype=np.float32),
        feat_mean_s=np.asarray(scaler.state_offset, dtype=np.float32),
        feat_std_s=np.asarray(scaler.state_scale, dtype=np.float32),
        static_mean=np.asarray(scaler.static_offset, dtype=np.float32),
        static_std=np.asarray(scaler.static_scale, dtype=np.float32),
        target_mean=np.asarray(scaler.target_offset, dtype=np.float32),
        target_std=np.asarray(scaler.target_scale, dtype=np.float32),
        forcing_cols=np.asarray(config.features.forcing, dtype=object),
        state_cols=np.asarray(config.features.state, dtype=object),
        static_cols=np.asarray(config.features.static, dtype=object),
        lc_col=np.asarray(config.features.land_cover or "Veg_ID", dtype=object),
        time_col=np.asarray(config.features.time, dtype=object),
        target_col=np.asarray(config.features.target, dtype=object),
        seq_len=np.asarray(config.window.seq_len),
        num_lc_classes=np.asarray(config.model.num_land_cover_classes or 1),
        lc_embed_dim=np.asarray(config.model.land_cover_embedding_dim),
    )
    contract = contract_from_config(config)
    contract.save(destination / "feature_contract.json")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(config_path).resolve().parent,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    payload = {
        "package_version": 1,
        "model_kind": config.model.kind.value,
        "history_hours": 96,
        "context_days": config.window.context_days,
        "daily_context_columns": list(config.window.daily_context_columns),
        "daily_context_features": config.model.daily_context_features,
        "input_arity": input_arity,
        "observation_features": {
            "endpoint_age": config.model.use_endpoint_observation_age,
            "observation_count": config.model.use_observation_count,
            "token_recency": config.model.use_token_recency,
        },
        "schema_hash": contract.schema_hash,
        "split_hash": split_hash,
        "code_commit": commit,
        "files": {
            path.name: sha256_file(path)
            for path in (
                scripted_path, checkpoint_copy, scaler_copy, global_scaler,
                destination / "feature_contract.json",
            )
        },
    }
    manifest = destination / "model_package.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
