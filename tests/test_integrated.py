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
from gpp_inversion.data import (
    BatchedWindowLoader,
    MultiStationWindowDataset,
    ScalingStats,
    StationBalancedSampler,
)
from gpp_inversion.engine import EvaluationResult
from gpp_inversion.ensemble import ensemble_prediction_files
from gpp_inversion.experiments import build_model
from gpp_inversion.losses import WeightedHuberLoss
from gpp_inversion.pipeline import run_experiment
from gpp_inversion.provenance import config_hash
from gpp_inversion.reporting import save_evaluation_artifacts
from gpp_inversion.splits import split_files_by_sites, validate_site_splits
from gpp_inversion.tree_baseline import TreeBaselineConfig, run_tree_baseline


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
            "EPIC_Available_Mask": np.ones(rows, dtype=int),
            "Band680nm_Ref": np.linspace(0.1, 0.3, rows),
            "Band780nm_Ref": np.linspace(0.4, 0.8, rows),
            "Lat": np.full(rows, 30.0),
            "Long": np.full(rows, 120.0),
            "Veg_ID": np.full(rows, land_cover, dtype=int),
            "target": np.arange(rows, dtype=float) + 1 + offset,
        }
    ).to_csv(path, index=False)


class IntegratedDataTest(unittest.TestCase):
    def test_spectral_indices_use_raw_reflectance_and_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SPECTRAL.csv"
            rows = 5
            pd.DataFrame(
                {
                    "date": pd.date_range("2026-01-01", periods=rows, freq="h"),
                    "forcing": np.arange(rows, dtype=float),
                    "state": np.arange(rows, dtype=float),
                    "Lat": np.full(rows, 30.0),
                    "Long": np.full(rows, 120.0),
                    "Veg_ID": np.ones(rows, dtype=int),
                    "target": np.arange(rows, dtype=float),
                    "EPIC_Available_Mask": [1, 1, 0, 1, 1],
                    "Band680nm_Ref": [0.2, 0.3, 0.8, 0.0, -0.2],
                    "Band780nm_Ref": [0.6, 0.3, 0.9, 0.0, 0.6],
                }
            ).to_csv(path, index=False)
            features = replace(
                FEATURES, spectral_indices=("NDVI", "NIRv")
            )
            dataset = MultiStationWindowDataset(
                [path], features, WindowConfig(seq_len=2)
            )
            raw = dataset.station_state[0] * dataset.scaler.state_scale + dataset.scaler.state_offset
            ndvi = raw[:, -2]
            nirv = raw[:, -1]
            np.testing.assert_allclose(ndvi, [0.5, 0.0, 0.0, 0.0, 1.0], atol=1e-6)
            np.testing.assert_allclose(nirv, [0.3, 0.0, 0.0, 0.0, 0.6], atol=1e-6)

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

    def test_station_balanced_sampler_returns_valid_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_station(root / "A.csv")
            write_station(root / "B.csv", offset=10.0)
            dataset = MultiStationWindowDataset(
                [root / "A.csv", root / "B.csv"],
                FEATURES,
                WindowConfig(seq_len=3, time_features=TimeFeatureMode.CYCLIC),
            )
            indices = list(
                StationBalancedSampler(dataset, num_samples=100, seed=7)
            )
            self.assertEqual(len(indices), 100)
            self.assertTrue(all(0 <= index < len(dataset) for index in indices))

    def test_vectorized_batches_match_single_sample_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_station(root / "A.csv")
            dataset = MultiStationWindowDataset(
                [root / "A.csv"],
                FEATURES,
                WindowConfig(seq_len=3, time_features=TimeFeatureMode.CDE),
            )
            batch = next(
                iter(
                    BatchedWindowLoader(
                        dataset, batch_size=3, shuffle=False, metadata="full"
                    )
                )
            )
            for sample_index in range(3):
                sample = dataset[sample_index]
                for field_index in range(6):
                    self.assertTrue(
                        torch.allclose(
                            batch[field_index][sample_index],
                            sample[field_index],
                        )
                    )
                self.assertEqual(batch[6][sample_index], sample[6])
                self.assertEqual(batch[7][sample_index], sample[7])


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
            (ModelKind.LSTM, 4),
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

    def test_static_film_and_legacy_tcn_state_dict(self):
        base = ModelConfig(
            kind=ModelKind.TCN,
            d_model=8,
            nhead=2,
            dim_feedforward=16,
            num_layers=1,
            num_land_cover_classes=3,
            land_cover_embedding_dim=2,
            dropout=0.0,
        )
        legacy = build_model(base, FEATURES, seq_len=5, time_feature_dim=4)
        reloaded = build_model(base, FEATURES, seq_len=5, time_feature_dim=4)
        reloaded.load_state_dict(legacy.state_dict(), strict=True)
        film = build_model(
            replace(base, static_context_mode="film"),
            FEATURES,
            seq_len=5,
            time_feature_dim=4,
        )
        output = film(*self._inputs(4))
        self.assertEqual(tuple(output.shape), (2,))
        self.assertTrue(torch.isfinite(output).all())

    def test_weighted_huber_is_finite(self):
        loss = WeightedHuberLoss()(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 3.0]))
        self.assertTrue(torch.isfinite(loss))

    def test_tcn_ablation_variants_forward(self):
        variants = (
            {"cross_attention_residual": True},
            {"cross_attention_residual": True, "lag_encoding": "continuous"},
            {
                "cross_attention_residual": True,
                "lag_encoding": "continuous",
                "tcn_layers": 5,
                "normalized_tcn": True,
            },
        )
        for changes in variants:
            with self.subTest(changes=changes):
                config = replace(
                    ModelConfig(
                        kind=ModelKind.TCN,
                        d_model=8,
                        nhead=2,
                        dim_feedforward=16,
                        num_layers=1,
                        num_land_cover_classes=3,
                        land_cover_embedding_dim=2,
                        dropout=0.0,
                    ),
                    **changes,
                )
                model = build_model(
                    config, FEATURES, seq_len=5, time_feature_dim=4
                )
                output = model(*self._inputs(4))
                self.assertEqual(tuple(output.shape), (2,))
                self.assertTrue(torch.isfinite(output).all())


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
                high_target_threshold=1.5,
            )
            self.assertTrue(Path(artifacts.predictions).exists())
            self.assertTrue(Path(artifacts.station_metrics).exists())
            self.assertTrue(Path(artifacts.high_target_metrics).exists())
            plot_files = sorted(Path(artifacts.plot_directory).rglob("*.png"))
            self.assertEqual(len(plot_files), 4)

    def test_equal_weight_ensemble_and_strict_alignment(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            base = pd.DataFrame(
                {
                    "station": ["A", "A", "B"],
                    "date": pd.date_range("2026-01-01", periods=3, freq="h"),
                    "land_cover_id": [1, 1, 2],
                    "target": [1.0, 2.0, 3.0],
                    "prediction": [0.0, 2.0, 4.0],
                }
            )
            first = root / "first.csv"
            second = root / "second.csv"
            base.to_csv(first, index=False)
            other = base.copy()
            other["prediction"] = [2.0, 2.0, 2.0]
            other.to_csv(second, index=False)
            manifest = ensemble_prediction_files(
                [first, second], root / "ensemble", high_target_threshold=2.5
            )
            combined = pd.read_csv(manifest["artifacts"]["predictions"])
            np.testing.assert_allclose(combined["prediction"], [1.0, 2.0, 3.0])
            np.testing.assert_allclose(combined["prediction_std"], [1.0, 0.0, 1.0])
            self.assertEqual(manifest["metrics"]["rmse"], 0.0)

            misaligned = other.iloc[[1, 0, 2]].copy()
            misaligned.to_csv(second, index=False)
            with self.assertRaisesRegex(ValueError, "alignment mismatch"):
                ensemble_prediction_files([first, second], root / "bad")


class TreeBaselineTest(unittest.TestCase):
    def test_small_end_to_end_writes_complete_artifacts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            train_path = root / "TRAIN.csv"
            val_path = root / "VAL.csv"
            write_station(train_path)
            write_station(val_path, offset=2.0)
            features = replace(
                FEATURES,
                state=(
                    "state",
                    "EPIC_Available_Mask",
                    "Band680nm_Ref",
                    "Band780nm_Ref",
                ),
                spectral_indices=("NDVI", "NIRv"),
            )
            result = run_tree_baseline(
                train_files=[train_path],
                val_files=[val_path],
                output_dir=root / "tree",
                features=features,
                window=WindowConfig(seq_len=3),
                land_cover_classes=3,
                config=TreeBaselineConfig(
                    max_windows_per_station=4,
                    batch_size=2,
                    max_iter=5,
                    seed=7,
                ),
            )
            self.assertTrue((root / "tree" / "experiment_manifest.json").exists())
            self.assertTrue(Path(result["validation"]["artifacts"]["predictions"]).exists())
            self.assertTrue(Path(result["validation"]["artifacts"]["station_metrics"]).exists())
            self.assertTrue(Path(result["validation"]["artifacts"]["high_target_metrics"]).exists())
            self.assertEqual(len(result["config_hash"]), 64)
            self.assertTrue(result["test_set_locked"])


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
            huber_one = replace(config, loss=LossKind.HUBER, loss_options={"delta": 1.0})
            huber_two = replace(config, loss=LossKind.HUBER, loss_options={"delta": 2.0})
            self.assertNotEqual(config_hash(huber_one), config_hash(huber_two))
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
