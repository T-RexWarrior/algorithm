import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gpp_inversion.config import (
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
from gpp_inversion.data import MultiStationWindowDataset, ScalingStats
from gpp_inversion.engine import EvaluationResult
from gpp_inversion.experiments import build_model
from gpp_inversion.losses import WeightedHuberLoss
from gpp_inversion.pipeline import run_experiment
from gpp_inversion.reporting import save_evaluation_artifacts
from gpp_inversion.splits import split_files_by_sites, validate_site_splits


FEATURES = FeatureColumns(
    forcing=("forcing",),
    state=("state",),
    static=("Lat", "Long"),
    target="target",
    time="date",
    land_cover="Veg_ID",
)


def write_station(path: Path, offset: float = 0.0, land_cover: int = 1) -> None:
    rows = 6
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows, freq="h"),
            "forcing": np.arange(rows, dtype=float) + offset,
            "state": np.arange(rows, dtype=float) + 10 + offset,
            "Lat": np.full(rows, 30.0),
            "Long": np.full(rows, 120.0),
            "Veg_ID": np.full(rows, land_cover, dtype=int),
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
            self.assertEqual(train[0][-1], "TRAIN")
            self.assertFalse(hasattr(train, "samples"))
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


class ReportingTest(unittest.TestCase):
    def test_notebook_style_evaluation_artifacts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            output_dir = Path(directory)
            dates = pd.date_range("2026-01-01", periods=8, freq="D").to_numpy()
            result = EvaluationResult(
                metrics={"rmse": 0.1, "mae": 0.08, "r2": 0.9, "count": 8},
                predictions=np.linspace(1.0, 2.0, 8),
                targets=np.linspace(1.05, 1.95, 8),
                dates=dates,
                station_names=np.array(["SITE_A"] * 8),
                land_cover_ids=np.ones(8, dtype=int),
            )
            artifacts = save_evaluation_artifacts(
                result,
                output_dir,
                prefix="test",
                config=EvaluationConfig(
                    save_predictions=True,
                    save_plots=True,
                    moving_average_window=3,
                    zoom_days=3,
                ),
            )
            self.assertTrue(Path(artifacts.predictions).exists())
            self.assertTrue(Path(artifacts.station_metrics).exists())
            plot_files = sorted(Path(artifacts.plot_directory).rglob("*.png"))
            self.assertEqual(len(plot_files), 4)


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
                evaluation=EvaluationConfig(
                    save_predictions=True,
                    save_plots=False,
                ),
            )
            result = run_experiment(config)
            self.assertEqual(result["split_counts"]["train_files"], 1)
            self.assertEqual(result["training"]["epochs_completed"], 1)
            self.assertTrue((output_dir / "checkpoint_best.pth").exists())
            self.assertTrue((output_dir / "scaler.npz").exists())
            self.assertGreater(result["test_metrics"]["count"], 0)
            self.assertEqual(len(result["config_hash"]), 64)
            self.assertTrue((output_dir / "experiment_manifest.json").exists())
            self.assertTrue((output_dir / "evaluation" / "test_predictions.csv").exists())
            self.assertTrue((output_dir / "evaluation" / "test_metrics_by_station.csv").exists())
            checkpoint = torch.load(
                output_dir / "checkpoint_latest.pth", weights_only=False
            )
            self.assertEqual(checkpoint["config_hash"], result["config_hash"])
            best_checkpoint = torch.load(
                output_dir / "checkpoint_best.pth", weights_only=False
            )
            self.assertEqual(
                best_checkpoint["config_hash"], result["config_hash"]
            )
            mismatched_config = replace(
                config,
                training=replace(
                    config.training,
                    epochs=2,
                    learning_rate=2e-3,
                    resume=True,
                ),
            )
            with self.assertRaisesRegex(ValueError, "configuration hash"):
                run_experiment(mismatched_config)

    def test_stratified_cross_validation_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stations = (
                ("DEV1", 0.0, 1),
                ("DEV2", 2.0, 1),
                ("DEV3", 4.0, 2),
                ("DEV4", 6.0, 2),
                ("TEST", 8.0, 1),
            )
            for name, offset, land_cover in stations:
                write_station(root / f"{name}.csv", offset, land_cover)
            output_dir = root / "cv_outputs"
            config = ExperimentConfig(
                data_dir=root,
                output_dir=output_dir,
                train_sites=("DEV1", "DEV3"),
                val_sites=("DEV2", "DEV4"),
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
                evaluation=EvaluationConfig(
                    save_predictions=True,
                    save_plots=False,
                ),
                cross_validation=CrossValidationConfig(
                    enabled=True,
                    n_splits=2,
                    seed=7,
                    evaluate_test_each_fold=False,
                ),
            )
            result = run_experiment(config)
            self.assertEqual(result["mode"], "stratified_kfold")
            self.assertEqual(len(result["folds"]), 2)
            self.assertEqual(result["validation_summary"]["folds"], 2)
            self.assertTrue((output_dir / "cross_validation_summary.json").exists())
            for fold_number in (1, 2):
                fold_dir = output_dir / f"fold_{fold_number:02d}"
                self.assertTrue(
                    (fold_dir / "evaluation" / "val_predictions.csv").exists()
                )


if __name__ == "__main__":
    unittest.main()
