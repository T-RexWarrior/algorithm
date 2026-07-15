"""Fail unless repeated CUDA training produces bitwise-identical parameters."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from gpp_inversion.models import TCNTransformerCrossAttention


def run_once(seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    model = TCNTransformerCrossAttention(
        9,
        14,
        96,
        num_static=2,
        time_feature_dim=4,
        num_lc_classes=13,
        lc_embed_dim=8,
    ).to(device)
    generator = torch.Generator().manual_seed(123)
    forcing = torch.randn(32, 96, 9, generator=generator).to(device)
    state = torch.randn(32, 96, 14, generator=generator).to(device)
    time_features = torch.randn(32, 96, 4, generator=generator).to(device)
    static = torch.randn(32, 96, 2, generator=generator).to(device)
    land_cover = torch.randint(
        0, 13, (32, 96), generator=generator
    ).to(device)
    target = torch.randn(32, generator=generator).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            forcing, state, time_features, static, land_cover
        )
        loss = torch.mean((prediction - target) ** 2)
        loss.backward()
        optimizer.step()
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def main() -> None:
    first = run_once(42)
    second = run_once(42)
    differences = {
        name: float(torch.max(torch.abs(first[name] - second[name])))
        for name in first
        if torch.is_floating_point(first[name])
    }
    maximum = max(differences.values(), default=0.0)
    print(f"maximum_parameter_difference={maximum:.12g}")
    if maximum != 0.0:
        raise SystemExit("CUDA training is not deterministic")


if __name__ == "__main__":
    main()
