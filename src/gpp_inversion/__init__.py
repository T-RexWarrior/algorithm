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
    SplitProtocolConfig,
    WindowConfig,
)
from .data import MultiStationWindowDataset, ScalingStats
from .explain import perform_shap_analysis
from .ensemble import ensemble_prediction_files
from .tree_baseline import TreeBaselineConfig, run_tree_baseline
from .contracts import FeatureContract, FeatureSpec, ModelBatch, spherical_xyz
from .losses import TailAwareLoss, WeightedHuberLoss, build_loss
from .models import (
    LayerNormLSTMGPP,
    TCNTransformerCrossAttention,
    TCN_Transformer_CrossAttention,
)
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
    "LayerNormLSTMGPP",
    "ModelConfig",
    "ModelKind",
    "MultiStationWindowDataset",
    "NeuralCDECrossAttentionGPP",
    "ScalingMethod",
    "ScalingStats",
    "TCNTransformerCrossAttention",
    "TimeAwareMambaTransformerCrossAttention",
    "TimeFeatureMode",
    "TCN_Transformer_CrossAttention",
    "TrainingConfig",
    "SplitProtocolConfig",
    "WeightedHuberLoss",
    "TailAwareLoss",
    "FeatureContract",
    "FeatureSpec",
    "ModelBatch",
    "spherical_xyz",
    "WindowConfig",
    "build_loss",
    "ensemble_prediction_files",
    "TreeBaselineConfig",
    "run_tree_baseline",
    "config_hash",
    "perform_shap_analysis",
    "run_experiment",
]

__version__ = "0.3.0"
