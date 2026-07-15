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
    MAMBA = "mamba"
    NEURAL_CDE = "neural_cde"


class LossKind(str, Enum):
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    WEIGHTED_HUBER = "weighted_huber"


@dataclass(frozen=True)
class FeatureColumns:
    forcing: tuple[str, ...]
    state: tuple[str, ...]
    static: tuple[str, ...] = ("Lat", "Long")
    target: str = "GPP_DT_VUT_REF"
    time: str = "date"
    land_cover: str | None = "Veg_ID"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FeatureColumns":
        return cls(
            forcing=tuple(value["forcing"]),
            state=tuple(value["state"]),
            static=tuple(value.get("static", ("Lat", "Long"))),
            target=value.get("target", "GPP_DT_VUT_REF"),
            time=value.get("time", "date"),
            land_cover=value.get("land_cover", "Veg_ID"),
        )

    @property
    def required(self) -> tuple[str, ...]:
        columns = (*self.forcing, *self.state, *self.static, self.target)
        return (*columns, self.land_cover) if self.land_cover else columns


@dataclass(frozen=True)
class WindowConfig:
    seq_len: int = 96
    time_features: TimeFeatureMode = TimeFeatureMode.IRREGULAR
    require_regular: bool = False
    max_gap_hours: float | None = 6.0
    max_span_hours: float | None = 168.0
    dt_clip_hours: float = 240.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WindowConfig":
        return cls(
            seq_len=int(value.get("seq_len", 96)),
            time_features=TimeFeatureMode(value.get("time_features", "irregular")),
            require_regular=bool(value.get("require_regular", False)),
            max_gap_hours=value.get("max_gap_hours", 6.0),
            max_span_hours=value.get("max_span_hours", 168.0),
            dt_clip_hours=float(value.get("dt_clip_hours", 240.0)),
        )

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

    def __post_init__(self) -> None:
        if self.tcn_layers < 1:
            raise ValueError("tcn_layers must be at least 1")
        if self.lag_encoding not in {"none", "continuous", "embedding"}:
            raise ValueError(
                "lag_encoding must be 'none', 'continuous', or 'embedding'"
            )

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
    selection_metric: str = "val_loss"
    station_balanced: bool = False
    amp: bool = False
    weight_decay: float = 0.0
    deterministic: bool = True

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

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainingConfig":
        return cls(**value)


@dataclass(frozen=True)
class EvaluationConfig:
    save_predictions: bool = True
    save_plots: bool = True
    moving_average_window: int = 12
    zoom_days: int = 30
    minimum_target: float | None = 0.0
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
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
