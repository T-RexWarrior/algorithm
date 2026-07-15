"""Optional SHAP analysis shared by all models with the unified signature."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class _ForcingStateWrapper(nn.Module):
    """Explain forcing/state while holding time, static and land cover fixed."""

    def __init__(self, model, time_reference, static_reference, land_cover_reference):
        super().__init__()
        self.model = model
        self.time_reference = time_reference[:1]
        self.static_reference = static_reference[:1]
        self.land_cover_reference = land_cover_reference[:1]

    def forward(self, forcing, state):
        batch_size = forcing.shape[0]
        result = self.model(
            forcing,
            state,
            self.time_reference.expand(batch_size, -1, -1),
            self.static_reference.expand(batch_size, -1, -1),
            self.land_cover_reference.expand(batch_size, -1),
        )
        return result.unsqueeze(-1)


def perform_shap_analysis(
    model,
    dataloader,
    device,
    output_dir: str | Path,
    forcing_columns,
    state_columns,
    *,
    background_size: int = 256,
    test_size: int = 256,
):
    """Create time-flattened SHAP beeswarm and bar plots.

    This preserves the later Notebook's interpretation: only forcing and state
    variables are explained; time, static and land-cover context are fixed to
    one reference sample.
    """
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "SHAP analysis requires the 'explainability' optional dependency"
        ) from exc

    required = background_size + test_size
    loader = DataLoader(dataloader.dataset, batch_size=required, shuffle=True)
    batch = next(iter(loader))
    if batch[0].shape[0] < required:
        raise ValueError(
            f"SHAP needs at least {required} windows, got {batch[0].shape[0]}"
        )
    forcing, state, time_features, static, land_cover = (
        value.to(device) for value in batch[:5]
    )
    background_forcing = forcing[:background_size]
    background_state = state[:background_size]
    test_forcing = forcing[background_size:required]
    test_state = state[background_size:required]
    wrapper = _ForcingStateWrapper(
        model, time_features, static, land_cover
    ).to(device)
    wrapper.eval()
    explainer = shap.GradientExplainer(
        wrapper, [background_forcing, background_state]
    )
    shap_values = explainer.shap_values([test_forcing, test_state])
    if (
        isinstance(shap_values, list)
        and len(shap_values) == 1
        and isinstance(shap_values[0], list)
    ):
        shap_values = shap_values[0]
    forcing_values = np.asarray(shap_values[0])
    state_values = np.asarray(shap_values[1])
    if forcing_values.ndim == 4 and forcing_values.shape[-1] == 1:
        forcing_values = forcing_values[..., 0]
    if state_values.ndim == 4 and state_values.shape[-1] == 1:
        state_values = state_values[..., 0]

    forcing_count = len(forcing_columns)
    state_count = len(state_columns)
    combined_shap = np.concatenate(
        [
            forcing_values.reshape(-1, forcing_count),
            state_values.reshape(-1, state_count),
        ],
        axis=1,
    )
    combined_features = np.concatenate(
        [
            test_forcing.detach().cpu().numpy().reshape(-1, forcing_count),
            test_state.detach().cpu().numpy().reshape(-1, state_count),
        ],
        axis=1,
    )
    names = [*forcing_columns, *state_columns]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "SHAP_Summary_Plot.png"
    bar_path = output_dir / "SHAP_Bar_Plot.png"

    shap.summary_plot(
        combined_shap, combined_features, feature_names=names, show=False
    )
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300)
    plt.close()
    shap.summary_plot(
        combined_shap,
        combined_features,
        feature_names=names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300)
    plt.close()
    return summary_path, bar_path
