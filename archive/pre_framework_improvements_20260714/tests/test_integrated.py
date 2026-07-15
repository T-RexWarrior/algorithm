import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gpp_inversion.config import (
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
from gpp_inversion.data import MultiStationWindowDataset, ScalingStats
from gpp_inversion.experiments import build_model
from gpp_inversion.losses import WeightedHuberLoss
from gpp_inversion.pipeline import run_experiment
from gpp_inversion.splits import split_files_by_sites, validate_site_splits


FEATURES = FeatureColumns(
    forcing=("forcing",),
    state=("state",),
    static=("Lat", "Long"),
    target="target",
    time="date",
    land_cover="Veg_ID",
)


def write_station(path: Path, offset: float = 0.0) -> None:
    rows = 6
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows, freq="h"),
            "forcing": np.arange(rows, dtype=float) + offset,
            "state": np.arange(rows, dtype=float) + 10 + offset,
            "Lat": np.full(rows, 30.0),
            "Long": np.full(rows, 120.0),
            "Veg_ID": np.ones(rows, dtype=int),
            "target": np.arange(rows, dtype=float) + 1 + offset,
        }
    ).to_csv(path, index=False)


class IntegratedDataTest(unittest.TestCase):
    def test_cde_features_and_training_scaler_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "TRAIN.csv"
            val_path = root / "VAL.csv"
            write_station(train_path)
            write_station(val_path, offset=100.0)
            window = WindowConfig(
                seq_len=3,
                time_features=TimeFeatureMode.CDE,
                max_gap_hours=2,
                max_span_hours=4,
            )
            train = MultiStationWindowDataset(
                [train_path], FEATURES, window, scaling=ScalingMethod.ZSCORE
            )
            val = MultiStationWindowDataset(
                [val_path], FEATURES, window, scaler=train.scaler, split_name="val"
            )
            self.assertIs(val.scaler, train.scaler)
            self.assertEqual(train.time_feature_dim, 7)
            self.assertEqual(tuple(train[0][2].shape), (3, 7))
            restored = train.scaler.inverse_target(np.array([0.0]))
            self.assertAlmostEqual(restored[0], 3.5)

    def test_scaler_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.npz"
            stats = ScalingStats(
                ScalingMethod.MINMAX,
                np.array([1.0]), np.array([2.0]),
                np.array([3.0]), np.array([4.0]),
                np.array([5.0]), np.array([6.0]),
                7.0, 8.0, True,
            )
            stats.save(path)
            loaded = ScalingStats.load(path)
            self.assertEqual(loaded.method, ScalingMethod.MINMAX)
            self.assertAlmostEqual(loaded.inverse_target([1.0])[0], 15.0)


class SplitTest(unittest.TestCase):
    def test_manual_split_and_overlap_guard(self):
        files = [Path("AA.csv"), Path("BB_extra.csv"), Path("CC.csv"), Path("DD.csv")]
        result = split_files_by_sites(files, ["AA"], ["BB"], ["CC"], strict=True)
        self.assertEqual([path.name for path in result.train], ["AA.csv"])
        self.assertEqual([path.name for path in result.val], ["BB_extra.csv"])
        self.assertEqual([path.name for path in result.test], ["CC.csv"])
        self.assertEqual([path.name for path in result.ignored], ["DD.csv"])
        with self.assertRaises(ValueError):
            validate_site_splits(["AA"], ["AA"], [])


class IntegratedModelTest(unittest.TestCase):
    def _inputs(self, time_dim: int):
        return (
            torch.randn(2, 5, 1),
            torch.randn(2, 5, 1),
            torch.randn(2, 5, time_dim),
            torch.randn(2, 5, 2),
            torch.ones(2, 5, dtype=torch.long),
        )

    def test_all_model_families_forward(self):
        cases = [
            (ModelKind.TCN, 4),
            (ModelKind.MAMBA, 6),
            (ModelKind.NEURAL_CDE, 7),
        ]
        for kind, time_dim in cases:
            with self.subTest(kind=kind):
                config = ModelConfig(
                    kind=kind,
                    d_model=8,
                    nhead=2,
                    dim_feedforward=16,
                    num_layers=1,
                    num_mamba_layers=1,
                    use_native_mamba=False,
                    cde_layers=1,
                    cde_vector_field_dim=16,
                    num_land_cover_classes=3,
                    land_cover_embedding_dim=2,
                    dropout=0.0,
                )
                model = build_model(config, FEATURES, seq_len=5, time_feature_dim=time_dim)
                output = model(*self._inputs(time_dim))
                self.assertEqual(tuple(output.shape), (2,))

    def test_weighted_huber_is_finite(self):
        loss = WeightedHuberLoss()(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 3.0]))
        self.assertTrue(torch.isfinite(loss))


class PipelineTest(unittest.TestCase):
    def test_one_epoch_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, offset in (("TRAIN", 0.0), ("VAL", 10.0), ("TEST", 20.0)):
                write_station(root / f"{name}.csv", offset)
            output_dir = root / "outputs"
            config = ExperimentConfig(
                data_dir=root,
                output_dir=output_dir,
                train_sites=("TRAIN",),
                val_sites=("VAL",),
                test_sites=("TEST",),
                features=FEATURES,
                window=WindowConfig(
                    seq_len=3,
                    time_features=TimeFeatureMode.CYCLIC,
                    max_gap_hours=2,
                    max_span_hours=4,
                ),
                scaling=ScalingMethod.ZSCORE,
                scale_target=True,
                model=ModelConfig(
                    kind=ModelKind.TCN,
                    d_model=8,
                    nhead=2,
                    dim_feedforward=16,
                    num_layers=1,
                    num_land_cover_classes=3,
                    land_cover_embedding_dim=2,
                    dropout=0.0,
                ),
                loss=LossKind.MSE,
                training=TrainingConfig(
                    batch_size=2,
                    epochs=1,
                    learning_rate=1e-3,
                    patience=1,
                    seed=7,
                    resume=False,
                ),
            )
            result = run_experiment(config)
            self.assertEqual(result["split_counts"]["train_files"], 1)
            self.assertEqual(result["training"]["epochs_completed"], 1)
            self.assertTrue((output_dir / "checkpoint_best.pth").exists())
            self.assertTrue((output_dir / "scaler.npz").exists())
            self.assertGreater(result["test_metrics"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
