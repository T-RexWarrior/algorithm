from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch

from gpp_inversion.blind import StationDescriptor, _previous_sites, select_blind_stations
from gpp_inversion.config import FeatureColumns, ModelConfig, ModelKind, WindowConfig
from gpp_inversion.contracts import ModelBatch, observation_metadata, spherical_xyz
from gpp_inversion.data import MultiStationWindowDataset
from gpp_inversion.ensemble import fit_nonnegative_oof_weights
from gpp_inversion.experiments import build_model
from gpp_inversion.losses import TailAwareLoss
from gpp_inversion.pretraining import KnowledgeGuidedPretrainer
from gpp_inversion.promotion import evaluate_promotion


FEATURES = FeatureColumns(
    forcing=("SW_IN_F", "TA_F", "VPD_F", "P_F", "SWC_F_MDS_1"),
    state=("EPIC_Available_Mask", "Band680nm_Ref"),
    static=("Lat", "Long"),
    target="GPP_DT_VUT_REF",
    time="date",
    land_cover="Veg_ID",
)


def test_spherical_coordinates_are_continuous_at_dateline():
    xyz = spherical_xyz([0.0, 0.0], [179.9, -179.9])
    assert np.linalg.norm(xyz[0] - xyz[1]) < 0.01


def test_observation_metadata_is_causal():
    valid, age, count = observation_metadata(np.array([[0, 1, 0, 0, 1, 0]]))
    np.testing.assert_array_equal(valid, [[False, True, False, False, True, False]])
    np.testing.assert_array_equal(age[0, 1:], [0, 1, 2, 0, 1])
    assert count.tolist() == [2]


def test_model_batch_contract_rejects_wrong_history():
    batch = ModelBatch(
        hourly_forcing=torch.zeros(2, 95, 5),
        state_values=torch.zeros(2, 95, 2),
        state_valid=torch.zeros(2, 95, dtype=torch.bool),
        state_age=torch.zeros(2, 95),
        time_features=torch.zeros(2, 95, 4),
        static_xyz=torch.zeros(2, 3),
        veg_id=torch.zeros(2, dtype=torch.long),
    )
    try:
        batch.validate()
    except ValueError as exc:
        assert "96" in str(exc)
    else:
        raise AssertionError("Expected history validation failure")


def test_observation_aware_variants_handle_all_missing_state():
    inputs = (
        torch.randn(3, 96, 5),
        torch.full((3, 96, 2), -0.5),
        torch.randn(3, 96, 4),
        torch.randn(3, 96, 2),
        torch.ones(3, 96, dtype=torch.long),
    )
    for kind in (ModelKind.TCN_OBSERVATION_AWARE, ModelKind.TCN_MULTISCALE):
        config = ModelConfig(
            kind=kind, d_model=8, nhead=2, num_layers=1, dim_feedforward=16,
            tcn_layers=2, num_land_cover_classes=3, land_cover_embedding_dim=2,
            daily_context_features=5, daily_context_hidden=4, dropout=0.0,
        )
        model = build_model(config, FEATURES, seq_len=96, time_feature_dim=4)
        kwargs = {"daily_context": torch.randn(3, 30, 5)} if kind is ModelKind.TCN_MULTISCALE else {}
        output = model(*inputs, **kwargs)
        assert output.shape == (3,)
        assert torch.isfinite(output).all()


def test_tail_loss_penalizes_high_underprediction_more():
    loss = TailAwareLoss(p50=1.0, p80=2.0, p95=3.0)
    low = loss(torch.tensor([0.0]), torch.tensor([1.0]))
    high = loss(torch.tensor([3.0]), torch.tensor([4.0]))
    assert high > low


def test_30_day_context_ends_at_same_target():
    rows = 30 * 24 + 2
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=rows, freq="h"),
            "SW_IN_F": np.arange(rows), "TA_F": 10.0, "VPD_F": 5.0,
            "P_F": 0.0, "SWC_F_MDS_1": 30.0,
            "EPIC_Available_Mask": 0.0, "Band680nm_Ref": 0.0,
            "Lat": 1.0, "Long": 2.0, "Veg_ID": 1,
            "GPP_DT_VUT_REF": np.arange(rows),
        }
    )
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        path = Path(directory) / "SITE_GRA_Merged.csv"
        frame.to_csv(path, index=False)
        dataset = MultiStationWindowDataset(
            [path], FEATURES,
            WindowConfig(seq_len=96, context_days=30, max_span_hours=95),
        )
        sample = dataset[0]
        assert len(sample) == 9
        assert sample[0].shape == (96, 5)
        assert sample[5].shape == (30, 5)
        assert sample[7].endswith("23:00:00.000000000")


def test_dataset_derives_spherical_static_coordinates_from_lat_lon():
    rows = 100
    features = replace(FEATURES, static=("Coord_X", "Coord_Y", "Coord_Z"))
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=rows, freq="h"),
            "SW_IN_F": 100.0, "TA_F": 10.0, "VPD_F": 5.0,
            "P_F": 0.0, "SWC_F_MDS_1": 30.0,
            "EPIC_Available_Mask": 0.0, "Band680nm_Ref": 0.0,
            "Lat": 0.0, "Long": 90.0, "Veg_ID": 1,
            "GPP_DT_VUT_REF": 2.0,
        }
    )
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        path = Path(directory) / "SITE_GRA_Merged.csv"
        frame.to_csv(path, index=False)
        dataset = MultiStationWindowDataset(
            [path], features, WindowConfig(seq_len=96, max_span_hours=95),
        )
        raw_xyz = (
            dataset.station_static[0][0] * dataset.scaler.static_scale
            + dataset.scaler.static_offset
        )
        np.testing.assert_allclose(raw_xyz, [0.0, 1.0, 0.0], atol=1e-6)


def test_blind_selection_is_balanced_and_excludes_previous():
    descriptors = [
        StationDescriptor(
            file=f"S{i}.csv", site=f"S{i}", rows=200, valid_target_rows=150,
            lat=float(i - 40), lon=float(i * 3 - 120), veg_id=i % 4,
            epic_fraction=0.2 + i / 1000,
            forcing_means=(float(i), 10.0, 5.0, 0.1, 20.0),
        )
        for i in range(80)
    ]
    representative, ood = select_blind_stations(
        descriptors, previous_sites={"S0", "S1"}, count=60
    )
    selected = representative + ood
    assert len(selected) == 60
    assert len({item.site for item in selected}) == 60
    assert not {"S0", "S1"} & {item.site for item in selected}


def test_previous_sites_are_recovered_from_historical_manifest():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        path = Path(directory) / "experiment_manifest.json"
        path.write_text(
            '{"config":{"experiment":{"train_sites":["A"],'
            '"val_sites":["B"],"test_sites":["C"]}}}',
            encoding="utf-8",
        )
        assert _previous_sites(path) == {"A", "B", "C"}


def test_lightweight_pretraining_objective_is_finite():
    config = ModelConfig(
        kind=ModelKind.TCN_OBSERVATION_AWARE, d_model=8, nhead=2,
        num_layers=1, dim_feedforward=16, tcn_layers=2,
        num_land_cover_classes=3, land_cover_embedding_dim=2, dropout=0.0,
    )
    model = build_model(config, FEATURES, seq_len=96, time_feature_dim=4)
    wrapper = KnowledgeGuidedPretrainer(model, forcing_features=5, state_features=2)
    forcing = torch.randn(2, 96, 5)
    state = torch.randn(2, 96, 2)
    state[:, :, 0] = 0.0
    state[:, 24, 0] = 1.0
    loss, components = wrapper(
        forcing, state, torch.randn(2, 96, 4), mask_fraction=1e-9
    )
    assert torch.isfinite(loss)
    assert set(components) == {
        "forcing_reconstruction", "state_reconstruction", "next_epic"
    }


def test_oof_weights_are_nonnegative_and_sum_to_one():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        paths = []
        target = np.linspace(0.0, 10.0, 30)
        for index, prediction in enumerate((target + 0.1, target * 0.5 + 2.0)):
            path = Path(directory) / f"model_{index}.csv"
            pd.DataFrame({
                "station": [f"S{i // 3}" for i in range(30)],
                "date": pd.date_range("2020-01-01", periods=30, freq="h"),
                "target": target,
                "prediction": prediction,
            }).to_csv(path, index=False)
            paths.append(path)
        weights = fit_nonnegative_oof_weights(paths, max_rows=30)
        assert np.all(weights >= 0)
        assert np.isclose(weights.sum(), 1.0)
        assert weights[0] > weights[1]


def test_promotion_gate_uses_strict_paired_station_comparison():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        rows = []
        for station_index in range(10):
            for hour, target in enumerate((1.0, 5.0, 10.0, 20.0)):
                rows.append({
                    "station": f"S{station_index}",
                    "date": f"2020-01-01 {hour:02d}:00:00",
                    "land_cover_id": station_index % 2,
                    "target": target,
                })
        baseline = pd.DataFrame(rows)
        candidate = baseline.copy()
        baseline["prediction"] = baseline["target"] * 0.80
        candidate["prediction"] = candidate["target"] * 0.95
        baseline_path = Path(directory) / "baseline.csv"
        candidate_path = Path(directory) / "candidate.csv"
        baseline.to_csv(baseline_path, index=False)
        candidate.to_csv(candidate_path, index=False)
        report = evaluate_promotion(
            baseline_path, candidate_path, high_target_threshold=10.0,
            bootstrap_samples=100, seed=42,
        )
        assert report["passed"]
        assert report["station_win_fraction"] == 1.0
