"""Architecture profiling and fixed-window diagnostic artifact helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _tensor_batch(batch, device, limit=None):
    values = []
    for tensor in batch[:5]:
        if limit is not None:
            tensor = tensor[:limit]
        values.append(tensor.to(device))
    return tuple(values)


def profile_inference(model, loader, device, output_dir, *, max_batch=256):
    """Measure forward latency and peak allocated memory on one validation batch."""
    batch = next(iter(loader))
    actual_batch = min(int(batch[0].size(0)), max_batch)
    inputs = _tensor_batch(batch, device, actual_batch)
    daily_context = (
        batch[5][:actual_batch].to(device) if len(batch) == 9 else None
    )

    def run_forward():
        if daily_context is None:
            return model(*inputs)
        return model(*inputs, daily_context=daily_context)
    repeats = 10 if device.type == "cuda" else 2
    warmups = 3 if device.type == "cuda" else 1
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmups):
            run_forward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(repeats):
            run_forward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "parameter_count": parameter_count(model),
        "batch_size": actual_batch,
        "repeats": repeats,
        "mean_batch_latency_ms": elapsed * 1000.0 / repeats,
        "samples_per_second": actual_batch * repeats / elapsed,
        "peak_allocated_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else None
        ),
        "device": str(device),
    }
    path = Path(output_dir) / "architecture_profile.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, path


def _diagnostic_indices(dataset):
    rows = []
    offset = 0
    labels = ("first", "middle", "last")
    for station, count in zip(dataset.station_names, dataset.station_window_counts):
        count = int(count)
        if count:
            local = (0, count // 2, count - 1)
            for label, local_index in zip(labels, local):
                rows.append((offset + local_index, station, label, local_index))
        offset += count
    return rows


def save_fixed_window_diagnostics(model, dataset, device, output_dir):
    """Save diagnostics for the first/middle/last window of every station."""
    if not hasattr(model, "forward_with_diagnostics"):
        return None
    selections = _diagnostic_indices(dataset)
    if not selections:
        return None
    samples = [dataset[index] for index, *_ in selections]
    inputs = tuple(
        torch.stack([sample[field] for sample in samples]).to(device)
        for field in range(5)
    )
    model.eval()
    with torch.no_grad():
        predictions, diagnostics = model.forward_with_diagnostics(*inputs)
    arrays = {"prediction_scaled": predictions.detach().float().cpu().numpy()}
    for name, value in diagnostics.items():
        if torch.is_tensor(value):
            arrays[name] = value.detach().float().cpu().numpy()
    output_dir = Path(output_dir)
    values_path = output_dir / "architecture_diagnostics.npz"
    np.savez_compressed(values_path, **arrays)
    metadata = {
        "selection_rule": "first/middle/last valid window per validation station",
        "windows": [
            {
                "dataset_index": int(index),
                "station": station,
                "position": label,
                "local_window_index": int(local_index),
                "date": str(sample[6]),
            }
            for (index, station, label, local_index), sample in zip(selections, samples)
        ],
        "arrays": {name: list(value.shape) for name, value in arrays.items()},
        "values_file": str(values_path),
    }
    metadata_path = output_dir / "architecture_diagnostics.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"metadata": str(metadata_path), "values": str(values_path)}
