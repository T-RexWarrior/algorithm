"""Lightweight masked reconstruction and weather-to-next-EPIC pretraining."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeGuidedPretrainer(nn.Module):
    def __init__(self, model, forcing_features: int, state_features: int) -> None:
        super().__init__()
        required = ("tcn", "state_projector", "state_encoder")
        if any(not hasattr(model, name) for name in required):
            raise ValueError("Pretraining requires an observation-aware production model")
        self.model = model
        d_model = model.state_projector.out_features
        self.forcing_head = nn.Linear(d_model, forcing_features)
        self.state_head = nn.Linear(d_model, state_features)
        self.next_state_head = nn.Linear(d_model, state_features)

    def forward(self, forcing, state, time_features, *, mask_fraction: float = 0.15):
        if not 0.0 < mask_fraction <= 1.0:
            raise ValueError("mask_fraction must be in (0, 1]")
        forcing_mask = torch.rand_like(forcing) < mask_fraction
        state_mask = torch.rand_like(state) < mask_fraction
        # Tiny smoke batches can draw no masked elements. Always supervise at
        # least one scalar so the pretraining objective remains finite.
        if not forcing_mask.any():
            forcing_mask[0, 0, 0] = True
        if not state_mask.any():
            state_mask[0, 0, 0] = True
        forcing_input = forcing.masked_fill(forcing_mask, 0.0)
        state_input = state.masked_fill(state_mask, 0.0)
        forcing_latent = self.model.tcn(forcing_input.transpose(1, 2)).transpose(1, 2)
        state_latent = self.model.project_state_inputs(state_input, time_features)
        state_latent = self.model.state_encoder(state_latent)
        forcing_reconstruction = self.forcing_head(forcing_latent)
        state_reconstruction = self.state_head(state_latent)
        forcing_loss = F.mse_loss(
            forcing_reconstruction[forcing_mask], forcing[forcing_mask]
        )
        state_loss = F.mse_loss(
            state_reconstruction[state_mask], state[state_mask]
        )
        next_valid = state[:, 1:, self.model.satellite_mask_index] > 0
        next_prediction = self.next_state_head(forcing_latent[:, :-1])
        if next_valid.any():
            forecast_loss = F.mse_loss(
                next_prediction[next_valid], state[:, 1:][next_valid]
            )
        else:
            forecast_loss = forcing_loss.new_zeros(())
        return forcing_loss + state_loss + 0.5 * forecast_loss, {
            "forcing_reconstruction": forcing_loss.detach(),
            "state_reconstruction": state_loss.detach(),
            "next_epic": forecast_loss.detach(),
        }


def pretrain_model(
    model,
    loader,
    device,
    output_dir: str | Path,
    *,
    max_steps: int = 3000,
    learning_rate: float = 1e-3,
    mask_fraction: float = 0.15,
) -> dict:
    first = next(iter(loader))
    wrapper = KnowledgeGuidedPretrainer(model, first[0].shape[-1], first[1].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    step = 0
    while step < max_steps:
        for batch in loader:
            forcing, state, time_features = [value.to(device) for value in batch[:3]]
            optimizer.zero_grad(set_to_none=True)
            loss, components = wrapper(
                forcing, state, time_features, mask_fraction=mask_fraction
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 100 == 0 or step == 1:
                history.append({
                    "step": step, "loss": float(loss.detach().cpu()),
                    **{name: float(value.cpu()) for name, value in components.items()},
                })
            if step >= max_steps:
                break
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "pretrained_encoder.pth"
    torch.save({"model_state_dict": model.state_dict(), "steps": step}, checkpoint)
    manifest = {"steps": step, "mask_fraction": mask_fraction, "history": history, "checkpoint": str(checkpoint)}
    (output_dir / "pretraining_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
