import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gpp_inversion import MultiStationWindowDataset, TCNTransformerCrossAttention
from gpp_inversion.config import FeatureColumns, TimeFeatureMode, WindowConfig


class DatasetSmokeTest(unittest.TestCase):
    def test_dataset_builds_sliding_windows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "station.csv"
            rows = 6
            pd.DataFrame(
                {
                    "date": pd.date_range("2026-01-01", periods=rows, freq="h"),
                    "forcing_a": np.arange(rows, dtype=float),
                    "forcing_b": np.arange(rows, dtype=float) + 10,
                    "state_a": np.arange(rows, dtype=float) + 20,
                    "Lat": np.full(rows, 30.0),
                    "Long": np.full(rows, 120.0),
                    "GPP_DT_VUT_REF": np.arange(rows, dtype=float) + 1,
                }
            ).to_csv(csv_path, index=False)

            dataset = MultiStationWindowDataset(
                [csv_path],
                FeatureColumns(
                    forcing=("forcing_a", "forcing_b"),
                    state=("state_a",),
                    target="GPP_DT_VUT_REF",
                    time="date",
                    land_cover=None,
                ),
                WindowConfig(seq_len=4, time_features=TimeFeatureMode.CYCLIC),
                scale_target=False,
            )

            self.assertEqual(len(dataset), 3)
            forcing, state, time_features, static, land_cover, target, target_date, station = dataset[0]
            self.assertEqual(tuple(forcing.shape), (4, 2))
            self.assertEqual(tuple(state.shape), (4, 1))
            self.assertEqual(tuple(time_features.shape), (4, 4))
            self.assertEqual(tuple(static.shape), (4, 2))
            self.assertEqual(target.item(), 4.0)
            self.assertIn("2026-01-01", target_date)
            self.assertEqual(land_cover.shape, (4,))
            self.assertEqual(station, "station")


class ModelSmokeTest(unittest.TestCase):
    def test_model_forward_shape(self):
        model = TCNTransformerCrossAttention(
            num_forcing_features=2,
            num_state_features=1,
            seq_len=8,
            num_static=2,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        output = model(
            torch.randn(3, 8, 2),
            torch.randn(3, 8, 1),
            torch.randn(3, 8, 4),
            torch.randn(3, 8, 2),
        )
        self.assertEqual(tuple(output.shape), (3,))


if __name__ == "__main__":
    unittest.main()
