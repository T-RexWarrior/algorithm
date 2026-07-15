"""Target-blind ERA-style forcing perturbations for deployment stress tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor


RADIATION = {"SW_IN_F", "SW_IN_POT"}
PRECIPITATION = {"P_F"}
BOUNDS = {
    "SW_IN_F": (0.0, 1500.0), "SW_IN_POT": (0.0, 1600.0),
    "CO2_F_MDS": (250.0, 700.0), "P_F": (0.0, 100.0),
    "VPD_F": (0.0, 120.0), "TA_F": (-80.0, 70.0),
    "TS_F_MDS_1": (-80.0, 70.0), "SWC_F_MDS_1": (0.0, 100.0),
    "WS_F": (0.0, 80.0),
}


def _seed(base: int, station: str, feature: str = "") -> int:
    digest = hashlib.sha256(f"{base}|{station}|{feature}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


@dataclass(frozen=True)
class EraStressTransform:
    features: dict[str, dict]
    source_hash: str

    @classmethod
    def load(cls, path: str | Path) -> "EraStressTransform":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("uses_target", True):
            raise ValueError("ERA stress manifest must explicitly declare uses_target=false")
        return cls(dict(payload["features"]), str(payload["source_sha256"]))

    def apply(self, values: np.ndarray, columns, *, station: str, seed: int) -> np.ndarray:
        output = np.asarray(values, dtype=np.float32).copy()
        for index, name in enumerate(columns):
            if name not in self.features:
                raise ValueError(f"ERA stress manifest missing feature {name}")
            spec = self.features[name]
            raw = output[:, index].astype(np.float64)
            rng = np.random.default_rng(_seed(seed, station, name))
            if name in PRECIPITATION:
                wet = raw > 0
                simulated_wet = wet.copy()
                simulated_wet[wet] = rng.random(wet.sum()) >= float(spec["wet_to_dry_probability"])
                simulated_wet[~wet] = rng.random((~wet).sum()) < float(spec["dry_to_wet_probability"])
                base = np.expm1(float(spec["slope"]) * np.log1p(np.maximum(raw, 0)) + float(spec["intercept"]))
                quantiles = np.asarray(spec["residual_quantiles"], dtype=np.float64)
                residual = np.interp(rng.random(raw.size), np.linspace(0, 1, quantiles.size), quantiles)
                transformed = np.where(simulated_wet, np.maximum(0.0, base + residual), 0.0)
            else:
                transformed = float(spec["slope"]) * raw + float(spec["intercept"])
                quantiles = np.asarray(spec["residual_quantiles"], dtype=np.float64)
                transformed += np.interp(
                    rng.random(raw.size), np.linspace(0, 1, quantiles.size), quantiles
                )
                if name in RADIATION:
                    transformed = np.where(raw <= 1.0, 0.0, transformed)
            lower, upper = BOUNDS.get(name, (-np.inf, np.inf))
            output[:, index] = np.clip(transformed, lower, upper).astype(np.float32)
        return output


def fit_era_stress_manifest(pair_csv: str | Path, output: str | Path) -> Path:
    path = Path(pair_csv)
    frame = pd.read_csv(path)
    required = {"station", "split", "feature", "tower", "era"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Pair CSV missing columns: {sorted(required - set(frame.columns))}")
    train = frame[frame["split"] == "train"].copy()
    if train.empty:
        raise ValueError("Pair CSV has no training-site rows")
    features = {}
    for name, group in train.groupby("feature"):
        x = pd.to_numeric(group["tower"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(group["era"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if name in RADIATION:
            fit = (x > 1.0) | (y > 1.0)
        else:
            fit = np.ones(x.shape, dtype=bool)
        if name in PRECIPITATION:
            fit_x = np.log1p(np.maximum(x[fit], 0))
            fit_y = np.log1p(np.maximum(y[fit], 0))
        else:
            fit_x, fit_y = x[fit], y[fit]
        if fit_x.size < 20:
            raise ValueError(f"Insufficient ERA pairs for {name}: {fit_x.size}")
        model = HuberRegressor(epsilon=1.35, max_iter=500).fit(fit_x[:, None], fit_y)
        predicted = model.predict(fit_x[:, None])
        residual = fit_y - predicted
        spec = {
            "count": int(fit_x.size),
            "slope": float(model.coef_[0]),
            "intercept": float(model.intercept_),
            "residual_quantiles": np.quantile(residual, np.linspace(0, 1, 101)).tolist(),
        }
        if name in PRECIPITATION:
            tower_wet, era_wet = x > 0, y > 0
            spec["wet_to_dry_probability"] = float(np.mean(~era_wet[tower_wet])) if tower_wet.any() else 0.0
            spec["dry_to_wet_probability"] = float(np.mean(era_wet[~tower_wet])) if (~tower_wet).any() else 0.0
        features[str(name)] = spec
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "version": 1, "uses_target": False, "screening_only": True,
        "source": str(path.resolve()), "source_sha256": digest,
        "features": features,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
