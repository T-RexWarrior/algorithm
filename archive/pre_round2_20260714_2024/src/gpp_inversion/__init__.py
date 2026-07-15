"""Reusable components for station-scale GPP inversion experiments."""

from .config import (
    CrossValidationConfig,
    EvaluationConfig,
    ExperimentConfig,
    FeatureColumns,
    LossKind,
    ModelConfig,
    ModelKind,
    ScalingMethod,
    TimeFeatureMode,
    TrainingConfig,
    WindowConfig,
)
from .data import MultiStationWindowDataset, ScalingStats
from .dataset import TimeAwareMultiStationDataset
from .explain import perform_shap_analysis
from .losses import WeightedHuberLoss, build_loss
from .models import TCNTransformerCrossAttention, TCN_Transformer_CrossAttention
from .models_irregular import (
    NeuralCDECrossAttentionGPP,
    TimeAwareMambaTransformerCrossAttention,
)
from .pipeline import run_experiment
from .provenance import config_hash

__all__ = [
    "CrossValidationConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "FeatureColumns",
    "LossKind",
    "ModelConfig",
    "ModelKind",
    "MultiStationWindowDataset",
    "NeuralCDECrossAttentionGPP",
    "ScalingMethod",
    "ScalingStats",
    "TCNTransformerCrossAttention",
    "TimeAwareMultiStationDataset",
    "TimeAwareMambaTransformerCrossAttention",
    "TimeFeatureMode",
    "TCN_Transformer_CrossAttention",
    "TrainingConfig",
    "WeightedHuberLoss",
    "WindowConfig",
    "build_loss",
    "config_hash",
    "perform_shap_analysis",
    "run_experiment",
]

__version__ = "0.3.0"
