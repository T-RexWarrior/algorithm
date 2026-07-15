"""Typed configuration shared by scripts and notebooks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ScalingMethod(str, Enum):
    ZSCORE = "zscore"
    MINMAX = "minmax"


class TimeFeatureMode(str, Enum):
    CYCLIC = "cyclic"
    IRREGULAR = "irregular"
    CDE = "cde"


class ModelKind(str, Enum):
    TCN = "tcn"
    TCN_OBSERVATION_AWARE = "tcn_observation_aware"
    TCN_MULTISCALE = "tcn_multiscale"
    HYBRID_LUE_TCN = "hybrid_lue_tcn"
    LSTM = "lstm"
    MAMBA = "mamba"
    NEURAL_CDE = "neural_cde"
    MODERN_TCN = "modern_tcn"
    TIMEXER = "timexer"
    TIME_MIXER_PP = "time_mixer_pp"


class LossKind(str, Enum):
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    WEIGHTED_HUBER = "weighted_huber"
    TAIL_AWARE = "tail_aware"


@dataclass(frozen=True)
class FeatureColumns:
    forcing: tuple[str, ...]
    state: tuple[str, ...]
    static: tuple[str, ...] = ("Lat", "Long")
    target: str = "GPP_DT_VUT_REF"
    time: str = "date"
    land_cover: str | None = "Veg_ID"
    spectral_indices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unsupported = set(self.spectral_indices) - {"NDVI", "NIRv"}
        if unsupported:
            raise ValueError(f"Unsupported spectral indices: {sorted(unsupported)}")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FeatureColumns":
        return cls(
            forcing=tuple(value["forcing"]),
            state=tuple(value["state"]),
            static=tuple(value.get("static", ("Lat", "Long"))),
            target=value.get("target", "GPP_DT_VUT_REF"),
            time=value.get("time", "date"),
            land_cover=value.get("land_cover", "Veg_ID"),
            spectral_indices=tuple(value.get("spectral_indices", ())),
        )

    @property
    def required(self) -> tuple[str, ...]:
        spectral_sources = (
            ("EPIC_Available_Mask", "Band680nm_Ref", "Band780nm_Ref")
            if self.spectral_indices else ()
        )
        columns = tuple(
            dict.fromkeys(
                (*self.forcing, *self.state, *self.static, *spectral_sources, self.target)
            )
        )
        return (*columns, self.land_cover) if self.land_cover else columns

    @property
    def state_dimension(self) -> int:
        return len(self.state) + len(self.spectral_indices)


@dataclass(frozen=True)
class WindowConfig:
    seq_len: int = 96
    time_features: TimeFeatureMode = TimeFeatureMode.CYCLIC
    require_regular: bool = True
    max_gap_hours: float | None = 1.0
    max_span_hours: float | None = 95.0
    dt_clip_hours: float = 240.0
    endpoint_stride: int = 1
    endpoint_phase: int = 0
    context_days: int = 0
    daily_context_columns: tuple[str, ...] = (
        "SW_IN_F", "TA_F", "VPD_F", "P_F", "SWC_F_MDS_1"
    )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WindowConfig":
        return cls(
            seq_len=int(value.get("seq_len", 96)),
            time_features=TimeFeatureMode(value.get("time_features", "cyclic")),
            require_regular=bool(value.get("require_regular", True)),
            max_gap_hours=value.get("max_gap_hours", 1.0),
            max_span_hours=value.get("max_span_hours", 95.0),
            dt_clip_hours=float(value.get("dt_clip_hours", 240.0)),
            endpoint_stride=int(value.get("endpoint_stride", 1)),
            endpoint_phase=int(value.get("endpoint_phase", 0)),
            context_days=int(value.get("context_days", 0)),
            daily_context_columns=tuple(
                value.get(
                    "daily_context_columns",
                    ("SW_IN_F", "TA_F", "VPD_F", "P_F", "SWC_F_MDS_1"),
                )
            ),
        )

    def __post_init__(self) -> None:
        if self.seq_len < 1:
            raise ValueError("seq_len must be positive")
        if self.endpoint_stride < 1:
            raise ValueError("endpoint_stride must be positive")
        if not 0 <= self.endpoint_phase < self.endpoint_stride:
            raise ValueError("endpoint_phase must be within endpoint_stride")
        if self.context_days < 0:
            raise ValueError("context_days cannot be negative")
        if self.context_days and not self.daily_context_columns:
            raise ValueError("daily_context_columns cannot be empty")

    @property
    def time_feature_dim(self) -> int:
        return {
            TimeFeatureMode.CYCLIC: 4,
            TimeFeatureMode.IRREGULAR: 6,
            TimeFeatureMode.CDE: 7,
        }[self.time_features]


@dataclass(frozen=True)
class ModelConfig:
    kind: ModelKind = ModelKind.TCN
    d_model: int = 64
    nhead: int = 4
    dropout: float = 0.1
    dim_feedforward: int = 128
    num_layers: int = 2
    num_mamba_layers: int = 4
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    use_native_mamba: bool | None = None
    cde_layers: int = 1
    cde_vector_field_dim: int = 128
    increment_scale: float = 1.0
    num_land_cover_classes: int | None = 13
    land_cover_embedding_dim: int = 8
    tcn_layers: int = 6
    normalized_tcn: bool = False
    cross_attention_residual: bool = False
    lag_encoding: str = "none"
    static_context_mode: str = "repeated"
    lstm_hidden_size: int = 64
    lstm_layers: int = 2
    state_norm_first: bool = False
    cross_fusion_mode: str = "replace"
    cross_direction: str = "state_to_forcing"
    temporal_pooling: str = "last"
    patch_length: int = 8
    patch_stride: int = 4
    modern_tcn_blocks: int = 4
    modern_large_kernel: int = 13
    modern_small_kernel: int = 3
    mixer_blocks: int = 2
    mixer_top_k: int = 3
    satellite_mask_index: int = 0
    no_observation_age_hours: float = 240.0
    daily_context_features: int = 5
    daily_context_hidden: int = 32
    nonnegative_output: bool = False

    def __post_init__(self) -> None:
        if self.tcn_layers < 1:
            raise ValueError("tcn_layers must be at least 1")
        if self.lag_encoding not in {"none", "continuous", "embedding"}:
            raise ValueError(
                "lag_encoding must be 'none', 'continuous', or 'embedding'"
            )
        if self.static_context_mode not in {"repeated", "film"}:
            raise ValueError("static_context_mode must be 'repeated' or 'film'")
        if self.lstm_hidden_size < 1 or self.lstm_layers < 1:
            raise ValueError("lstm_hidden_size and lstm_layers must be positive")
        if self.cross_fusion_mode not in {
            "replace", "legacy_residual", "zero_init_gated", "bidirectional_gated",
            "zero_init_bidirectional"
        }:
            raise ValueError("Unsupported cross_fusion_mode")
        if self.cross_direction not in {"state_to_forcing", "bidirectional"}:
            raise ValueError("Unsupported cross_direction")
        if self.temporal_pooling not in {"last", "gpp_query"}:
            raise ValueError("Unsupported temporal_pooling")
        if self.patch_length < 1 or self.patch_stride < 1:
            raise ValueError("patch_length and patch_stride must be positive")
        if self.modern_tcn_blocks < 1 or self.mixer_blocks < 1:
            raise ValueError("architecture block counts must be positive")
        if self.mixer_top_k < 1:
            raise ValueError("mixer_top_k must be positive")
        if self.satellite_mask_index < 0:
            raise ValueError("satellite_mask_index cannot be negative")
        if self.no_observation_age_hours <= 0:
            raise ValueError("no_observation_age_hours must be positive")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelConfig":
        fields = dict(value)
        fields["kind"] = ModelKind(fields.get("kind", "tcn"))
        return cls(**fields)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    patience: int = 10
    num_workers: int = 0
    seed: int = 42
    resume: bool = True
    selection_metric: str = "macro_rmse"
    station_balanced: bool = False
    amp: bool = False
    weight_decay: float = 0.0
    deterministic: bool = True
    optimizer: str = "adam"
    max_steps: int | None = None
    warmup_steps: int = 0
    eval_interval_steps: int = 1000
    target_balanced: bool = False
    samples_per_epoch: int | None = None
    pretrained_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.selection_metric not in {
            "val_loss",
            "micro_rmse",
            "macro_rmse",
        }:
            raise ValueError(
                "selection_metric must be val_loss, micro_rmse, or macro_rmse"
            )
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.optimizer not in {"adam", "adamw"}:
            raise ValueError("optimizer must be 'adam' or 'adamw'")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.warmup_steps < 0 or self.eval_interval_steps < 1:
            raise ValueError("Invalid step scheduler settings")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainingConfig":
        return cls(**value)


@dataclass(frozen=True)
class EvaluationConfig:
    save_predictions: bool = True
    save_plots: bool = True
    moving_average_window: int = 12
    zoom_days: int = 30
    minimum_target: float | None = None
    evaluate_test: bool = True

    def __post_init__(self) -> None:
        if self.moving_average_window < 1:
            raise ValueError("moving_average_window must be at least 1")
        if self.zoom_days < 1:
            raise ValueError("zoom_days must be at least 1")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationConfig":
        return cls(**value)


@dataclass(frozen=True)
class CrossValidationConfig:
    enabled: bool = False
    n_splits: int = 5
    seed: int = 42
    evaluate_test_each_fold: bool = False

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CrossValidationConfig":
        return cls(**value)


@dataclass(frozen=True)
class SplitProtocolConfig:
    name: str = "manual"
    split_hash: str | None = None
    blind_split_hash: str | None = None
    legacy_test_sites: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SplitProtocolConfig":
        return cls(
            name=str(value.get("name", "manual")),
            split_hash=value.get("split_hash"),
            blind_split_hash=value.get("blind_split_hash"),
            legacy_test_sites=tuple(value.get("legacy_test_sites", ())),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path
    output_dir: Path
    train_sites: tuple[str, ...]
    val_sites: tuple[str, ...]
    test_sites: tuple[str, ...]
    features: FeatureColumns
    window: WindowConfig = field(default_factory=WindowConfig)
    scaling: ScalingMethod = ScalingMethod.ZSCORE
    scale_target: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossKind = LossKind.MSE
    loss_options: dict[str, Any] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    cross_validation: CrossValidationConfig = field(
        default_factory=CrossValidationConfig
    )
    split_protocol: SplitProtocolConfig = field(default_factory=SplitProtocolConfig)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            data_dir=Path(value["data_dir"]),
            output_dir=Path(value["output_dir"]),
            train_sites=tuple(value["train_sites"]),
            val_sites=tuple(value["val_sites"]),
            test_sites=tuple(value["test_sites"]),
            features=FeatureColumns.from_dict(value["features"]),
            window=WindowConfig.from_dict(value.get("window", {})),
            scaling=ScalingMethod(value.get("scaling", "zscore")),
            scale_target=bool(value.get("scale_target", True)),
            model=ModelConfig.from_dict(value.get("model", {})),
            loss=LossKind(value.get("loss", "mse")),
            loss_options=dict(value.get("loss_options", {})),
            training=TrainingConfig.from_dict(value.get("training", {})),
            evaluation=EvaluationConfig.from_dict(value.get("evaluation", {})),
            cross_validation=CrossValidationConfig.from_dict(
                value.get("cross_validation", {})
            ),
            split_protocol=SplitProtocolConfig.from_dict(
                value.get("split_protocol", {})
            ),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
