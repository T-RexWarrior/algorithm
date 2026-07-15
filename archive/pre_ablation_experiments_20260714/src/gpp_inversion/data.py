"""Unified station-window dataset for regular, irregular and CDE experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import FeatureColumns, ScalingMethod, TimeFeatureMode, WindowConfig


@dataclass
class ScalingStats:
    method: ScalingMethod
    forcing_offset: np.ndarray
    forcing_scale: np.ndarray
    state_offset: np.ndarray
    state_scale: np.ndarray
    static_offset: np.ndarray
    static_scale: np.ndarray
    target_offset: float
    target_scale: float
    scale_target: bool = True

    def inverse_target(self, values):
        values = np.asarray(values)
        if not self.scale_target:
            return values
        return values * self.target_scale + self.target_offset

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            method=self.method.value,
            forcing_offset=self.forcing_offset,
            forcing_scale=self.forcing_scale,
            state_offset=self.state_offset,
            state_scale=self.state_scale,
            static_offset=self.static_offset,
            static_scale=self.static_scale,
            target_offset=self.target_offset,
            target_scale=self.target_scale,
            scale_target=int(self.scale_target),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ScalingStats":
        with np.load(path, allow_pickle=False) as values:
            return cls(
                method=ScalingMethod(str(values["method"].item())),
                forcing_offset=values["forcing_offset"],
                forcing_scale=values["forcing_scale"],
                state_offset=values["state_offset"],
                state_scale=values["state_scale"],
                static_offset=values["static_offset"],
                static_scale=values["static_scale"],
                target_offset=float(values["target_offset"]),
                target_scale=float(values["target_scale"]),
                scale_target=bool(int(values["scale_target"])),
            )


def _offset_and_scale(values: np.ndarray, method: ScalingMethod):
    if method is ScalingMethod.ZSCORE:
        offset = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
    else:
        offset = np.min(values, axis=0)
        scale = np.max(values, axis=0) - offset
    scale = np.asarray(scale).copy()
    scale[scale == 0] = 1e-8
    return offset, scale


class MultiStationWindowDataset(Dataset):
    """One dataset implementation covering the main Notebook data variants.

    The training dataset fits ``ScalingStats``. Validation and test datasets
    must receive that same object, preventing scaler leakage.
    """

    SENTINELS = (-9999, -9999.0, -999)

    def __init__(
        self,
        filepaths: Iterable[str | Path],
        features: FeatureColumns,
        window: WindowConfig | None = None,
        *,
        scaler: ScalingStats | None = None,
        scaling: ScalingMethod | str = ScalingMethod.ZSCORE,
        scale_target: bool = True,
        split_name: str = "train",
    ) -> None:
        self.features = features
        self.window = window or WindowConfig()
        self.split_name = split_name
        self.window_starts: list[np.ndarray] = []
        self.cumulative_window_counts = np.empty(0, dtype=np.int64)
        self.total_windows = 0
        self.station_forcing: list[np.ndarray] = []
        self.station_state: list[np.ndarray] = []
        self.station_static: list[np.ndarray] = []
        self.station_land_cover: list[np.ndarray] = []
        self.station_targets: list[np.ndarray] = []
        self.station_dates: list[np.ndarray] = []
        self.station_time_base: list[np.ndarray] = []
        self.station_names: list[str] = []
        self.loaded_files: list[Path] = []
        self.skipped_files: list[tuple[Path, str]] = []

        for raw_path in filepaths:
            self._load_station(Path(raw_path))
        if not self.station_forcing:
            raise ValueError(f"No usable station data for split '{split_name}'")

        scaling = ScalingMethod(scaling)
        self.scaler = scaler or self._fit_scaler(scaling, scale_target)
        self._apply_scaler()
        self._build_windows()
        if self.total_windows == 0:
            raise ValueError(
                f"No valid windows for split '{split_name}' with {self.window}"
            )

    @property
    def time_feature_dim(self) -> int:
        return self.window.time_feature_dim

    def _load_station(self, path: Path) -> None:
        selected_columns = list(
            dict.fromkeys([self.features.time, *self.features.required])
        )
        try:
            frame = pd.read_csv(
                path,
                usecols=selected_columns,
                low_memory=False,
                memory_map=True,
            )
        except (OSError, ValueError) as exc:
            self.skipped_files.append((path, str(exc)))
            return
        frame = frame.replace(list(self.SENTINELS), np.nan)
        frame[self.features.time] = pd.to_datetime(
            frame[self.features.time], errors="coerce"
        )
        clean_columns = [self.features.time, *self.features.required]
        frame = (
            frame.dropna(subset=clean_columns)
            .sort_values(self.features.time)
            .drop_duplicates(subset=[self.features.time], keep="last")
            .reset_index(drop=True)
        )
        if len(frame) < self.window.seq_len:
            self.skipped_files.append(
                (path, f"fewer than {self.window.seq_len} clean rows")
            )
            return

        dates = frame[self.features.time]
        hour = (
            dates.dt.hour.to_numpy()
            + dates.dt.minute.to_numpy() / 60.0
            + dates.dt.second.to_numpy() / 3600.0
        )
        day = dates.dt.dayofyear.to_numpy() + hour / 24.0
        cyclic = np.column_stack(
            [
                np.sin(2 * np.pi * hour / 24.0),
                np.cos(2 * np.pi * hour / 24.0),
                np.sin(2 * np.pi * day / 365.25),
                np.cos(2 * np.pi * day / 365.25),
            ]
        )
        if self.window.time_features is TimeFeatureMode.CYCLIC:
            time_base = cyclic
        else:
            dt_previous = dates.diff().dt.total_seconds().div(3600.0).to_numpy()
            dt_previous[0] = 0.0
            dt_previous = np.nan_to_num(
                dt_previous,
                nan=0.0,
                posinf=self.window.dt_clip_hours,
                neginf=0.0,
            )
            dt_previous = np.clip(dt_previous, 0.0, self.window.dt_clip_hours)
            time_base = np.column_stack([cyclic, np.log1p(dt_previous)])

        rows = len(frame)
        land_cover = (
            frame[self.features.land_cover].to_numpy(dtype=np.int64)
            if self.features.land_cover
            else np.zeros(rows, dtype=np.int64)
        )
        if land_cover.size and land_cover.min() < 0:
            raise ValueError(f"Negative land-cover ID in {path.name}")

        self.station_forcing.append(
            frame[list(self.features.forcing)].to_numpy(dtype=np.float32)
        )
        self.station_state.append(
            frame[list(self.features.state)].to_numpy(dtype=np.float32)
        )
        self.station_static.append(
            frame[list(self.features.static)].to_numpy(dtype=np.float32)
        )
        self.station_land_cover.append(land_cover)
        self.station_targets.append(
            frame[self.features.target].to_numpy(dtype=np.float32)
        )
        self.station_dates.append(dates.to_numpy(dtype="datetime64[ns]"))
        self.station_time_base.append(time_base.astype(np.float32))
        self.station_names.append(path.stem)
        self.loaded_files.append(path)

    def _fit_scaler(self, method: ScalingMethod, scale_target: bool) -> ScalingStats:
        forcing_offset, forcing_scale = _offset_and_scale(
            np.vstack(self.station_forcing), method
        )
        state_offset, state_scale = _offset_and_scale(
            np.vstack(self.station_state), method
        )
        static_offset, static_scale = _offset_and_scale(
            np.vstack(self.station_static), method
        )
        targets = np.concatenate(self.station_targets)
        if scale_target:
            target_offset_array, target_scale_array = _offset_and_scale(
                targets.reshape(-1, 1), method
            )
            target_offset = float(target_offset_array[0])
            target_scale = float(target_scale_array[0])
        else:
            target_offset, target_scale = 0.0, 1.0
        return ScalingStats(
            method=method,
            forcing_offset=forcing_offset,
            forcing_scale=forcing_scale,
            state_offset=state_offset,
            state_scale=state_scale,
            static_offset=static_offset,
            static_scale=static_scale,
            target_offset=target_offset,
            target_scale=target_scale,
            scale_target=scale_target,
        )

    def _apply_scaler(self) -> None:
        for index in range(len(self.station_forcing)):
            self.station_forcing[index] = (
                self.station_forcing[index] - self.scaler.forcing_offset
            ) / self.scaler.forcing_scale
            self.station_state[index] = (
                self.station_state[index] - self.scaler.state_offset
            ) / self.scaler.state_scale
            self.station_static[index] = (
                self.station_static[index] - self.scaler.static_offset
            ) / self.scaler.static_scale
            self.station_targets[index] = (
                self.station_targets[index] - self.scaler.target_offset
            ) / self.scaler.target_scale

    def _build_windows(self) -> None:
        length = self.window.seq_len
        if length < 1:
            raise ValueError("seq_len must be positive")
        counts: list[int] = []
        for dates in self.station_dates:
            date_ns = dates.astype("datetime64[ns]").astype(np.int64)
            candidate_count = len(dates) - length + 1
            if candidate_count <= 0:
                starts = np.empty(0, dtype=np.int64)
                self.window_starts.append(starts)
                counts.append(0)
                continue
            starts = np.arange(candidate_count, dtype=np.int64)
            valid = np.ones(candidate_count, dtype=bool)
            if length == 1:
                self.window_starts.append(starts)
                counts.append(candidate_count)
                continue

            differences = np.diff(date_ns).astype(np.float64) / 3_600_000_000_000.0
            positive = differences[differences > 0]
            expected_step = float(pd.Series(positive).mode().iloc[0]) if positive.size else 0.0

            kernel = np.ones(length - 1, dtype=np.int32)
            if self.window.require_regular:
                irregular_edges = ~np.isclose(differences, expected_step)
                valid &= np.convolve(
                    irregular_edges.astype(np.int32), kernel, mode="valid"
                ) == 0
            if self.window.max_gap_hours is not None:
                large_gaps = differences > self.window.max_gap_hours
                valid &= np.convolve(
                    large_gaps.astype(np.int32), kernel, mode="valid"
                ) == 0
            if self.window.max_span_hours is not None:
                spans = (
                    date_ns[length - 1 :] - date_ns[: -(length - 1)]
                ) / 3_600_000_000_000.0
                valid &= spans <= self.window.max_span_hours
            starts = starts[valid]
            self.window_starts.append(starts)
            counts.append(int(starts.size))

        self.cumulative_window_counts = np.cumsum(counts, dtype=np.int64)
        self.total_windows = (
            int(self.cumulative_window_counts[-1])
            if self.cumulative_window_counts.size else 0
        )

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, index: int):
        if index < 0:
            index += self.total_windows
        if index < 0 or index >= self.total_windows:
            raise IndexError(index)
        station_index = int(
            np.searchsorted(self.cumulative_window_counts, index, side="right")
        )
        previous_count = (
            0 if station_index == 0
            else int(self.cumulative_window_counts[station_index - 1])
        )
        local_index = index - previous_count
        start = int(self.window_starts[station_index][local_index])
        end = start + self.window.seq_len
        forcing = self.station_forcing[station_index][start:end]
        state = self.station_state[station_index][start:end]
        static = self.station_static[station_index][start:end]
        land_cover = self.station_land_cover[station_index][start:end]
        target = self.station_targets[station_index][end - 1]
        date_window = self.station_dates[station_index][start:end]
        date_ns = date_window.astype("datetime64[ns]").astype(np.int64)
        time_features = self.station_time_base[station_index][start:end].copy()

        if self.window.time_features is not TimeFeatureMode.CYCLIC:
            time_features[0, 4] = 0.0
            age = ((date_ns[-1] - date_ns) / 3_600_000_000_000.0).astype(
                np.float32
            )
            age = np.clip(
                np.nan_to_num(age, nan=0.0, posinf=self.window.dt_clip_hours),
                0.0,
                self.window.dt_clip_hours,
            )
            time_features = np.column_stack([time_features, np.log1p(age)])
            if self.window.time_features is TimeFeatureMode.CDE:
                relative = ((date_ns - date_ns[0]) / 3_600_000_000_000.0).astype(
                    np.float32
                )
                relative = np.clip(relative, 0.0, self.window.dt_clip_hours)
                time_features = np.column_stack(
                    [time_features, relative / self.window.dt_clip_hours]
                )

        return (
            torch.as_tensor(forcing, dtype=torch.float32),
            torch.as_tensor(state, dtype=torch.float32),
            torch.as_tensor(time_features, dtype=torch.float32),
            torch.as_tensor(static, dtype=torch.float32),
            torch.as_tensor(land_cover, dtype=torch.long),
            torch.as_tensor(target, dtype=torch.float32),
            str(self.station_dates[station_index][end - 1]),
            self.station_names[station_index],
        )
