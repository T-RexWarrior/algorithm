"""Unified station-window dataset for regular, irregular and CDE experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

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

    @property
    def station_window_counts(self) -> np.ndarray:
        """Number of valid windows for each loaded station."""
        if not self.cumulative_window_counts.size:
            return np.empty(0, dtype=np.int64)
        return np.diff(
            np.concatenate(
                [np.zeros(1, dtype=np.int64), self.cumulative_window_counts]
            )
        )

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


class StationBalancedSampler(Sampler[int]):
    """Draw stations uniformly, then draw one of that station's windows."""

    def __init__(
        self,
        dataset: MultiStationWindowDataset,
        *,
        num_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        counts = dataset.station_window_counts
        active = np.flatnonzero(counts > 0)
        if not active.size:
            raise ValueError("StationBalancedSampler requires valid windows")
        offsets = np.concatenate(
            [np.zeros(1, dtype=np.int64), dataset.cumulative_window_counts[:-1]]
        )
        self.counts = torch.as_tensor(counts[active], dtype=torch.long)
        self.offsets = torch.as_tensor(offsets[active], dtype=torch.long)
        self.num_samples = int(num_samples or len(dataset))
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        station_indices = torch.randint(
            len(self.counts),
            (self.num_samples,),
            generator=self.generator,
        )
        station_counts = self.counts[station_indices]
        local_indices = torch.floor(
            torch.rand(self.num_samples, generator=self.generator)
            * station_counts
        ).to(torch.long)
        global_indices = self.offsets[station_indices] + local_indices
        return iter(global_indices.tolist())


class BatchedWindowLoader:
    """Vectorized in-process loader for highly overlapping station windows.

    PyTorch's default loader calls ``Dataset.__getitem__`` once per window.
    With millions of 96-step windows that Python overhead dominates a small
    model. This loader groups a whole batch by station and gathers each group
    with NumPy advanced indexing while preserving exactly the same samples.
    """

    def __init__(
        self,
        dataset: MultiStationWindowDataset,
        *,
        batch_size: int,
        shuffle: bool = False,
        sampler: Sampler[int] | None = None,
        seed: int = 42,
        pin_memory: bool = False,
        metadata: str = "full",
    ) -> None:
        if shuffle and sampler is not None:
            raise ValueError("shuffle and sampler are mutually exclusive")
        if metadata not in {"none", "stations", "full"}:
            raise ValueError("metadata must be none, stations, or full")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.sampler = sampler
        self.pin_memory = pin_memory
        self.metadata = metadata
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        sample_count = len(self.sampler) if self.sampler is not None else len(self.dataset)
        return (sample_count + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.sampler is not None:
            indices = np.fromiter(
                iter(self.sampler), dtype=np.int64, count=len(self.sampler)
            )
        elif self.shuffle:
            indices = torch.randperm(
                len(self.dataset), generator=self.generator
            ).numpy()
        else:
            indices = np.arange(len(self.dataset), dtype=np.int64)
        for offset in range(0, len(indices), self.batch_size):
            yield self.dataset.get_batch(
                indices[offset : offset + self.batch_size],
                pin_memory=self.pin_memory,
                metadata=self.metadata,
            )


def _pin_if_requested(tensor: torch.Tensor, requested: bool) -> torch.Tensor:
    return tensor.pin_memory() if requested else tensor


def _dataset_get_batch(
    self: MultiStationWindowDataset,
    indices,
    *,
    pin_memory: bool = False,
    metadata: str = "full",
):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not indices.size:
        raise ValueError("indices must be a non-empty one-dimensional array")
    if indices.min() < 0 or indices.max() >= self.total_windows:
        raise IndexError("batch index out of range")
    station_indices = np.searchsorted(
        self.cumulative_window_counts, indices, side="right"
    )
    batch_size = len(indices)
    length = self.window.seq_len
    forcing = np.empty(
        (batch_size, length, self.station_forcing[0].shape[1]), dtype=np.float32
    )
    state = np.empty(
        (batch_size, length, self.station_state[0].shape[1]), dtype=np.float32
    )
    static = np.empty(
        (batch_size, length, self.station_static[0].shape[1]), dtype=np.float32
    )
    land_cover = np.empty((batch_size, length), dtype=np.int64)
    target = np.empty(batch_size, dtype=np.float32)
    time_features = np.empty(
        (batch_size, length, self.time_feature_dim), dtype=np.float32
    )
    dates = [""] * batch_size if metadata == "full" else []
    if metadata == "full":
        stations = [""] * batch_size
    elif metadata == "stations":
        stations = station_indices.copy()
    else:
        stations = []
    steps = np.arange(length, dtype=np.int64)

    for station_index in np.unique(station_indices):
        positions = np.flatnonzero(station_indices == station_index)
        previous_count = (
            0
            if station_index == 0
            else int(self.cumulative_window_counts[station_index - 1])
        )
        local_indices = indices[positions] - previous_count
        starts = self.window_starts[station_index][local_indices]
        row_indices = starts[:, None] + steps[None, :]
        forcing[positions] = self.station_forcing[station_index][row_indices]
        state[positions] = self.station_state[station_index][row_indices]
        static[positions] = self.station_static[station_index][row_indices]
        land_cover[positions] = self.station_land_cover[station_index][row_indices]
        target[positions] = self.station_targets[station_index][starts + length - 1]
        station_time = self.station_time_base[station_index][row_indices].copy()

        if self.window.time_features is not TimeFeatureMode.CYCLIC:
            station_time[:, 0, 4] = 0.0
            date_ns = self.station_dates[station_index][row_indices].astype(
                "datetime64[ns]"
            ).astype(np.int64)
            age = (date_ns[:, -1, None] - date_ns) / 3_600_000_000_000.0
            age = np.clip(
                np.nan_to_num(
                    age,
                    nan=0.0,
                    posinf=self.window.dt_clip_hours,
                ),
                0.0,
                self.window.dt_clip_hours,
            ).astype(np.float32)
            station_time = np.concatenate(
                [station_time, np.log1p(age)[:, :, None]], axis=2
            )
            if self.window.time_features is TimeFeatureMode.CDE:
                relative = (date_ns - date_ns[:, :1]) / 3_600_000_000_000.0
                relative = np.clip(
                    relative, 0.0, self.window.dt_clip_hours
                ).astype(np.float32)
                station_time = np.concatenate(
                    [
                        station_time,
                        (relative / self.window.dt_clip_hours)[:, :, None],
                    ],
                    axis=2,
                )
        time_features[positions] = station_time

        if metadata == "full":
            end_rows = starts + length - 1
            name = self.station_names[station_index]
            for position, end_row in zip(positions, end_rows):
                dates[int(position)] = str(
                    self.station_dates[station_index][int(end_row)]
                )
                stations[int(position)] = name

    tensors = (
        torch.from_numpy(forcing),
        torch.from_numpy(state),
        torch.from_numpy(time_features),
        torch.from_numpy(static),
        torch.from_numpy(land_cover),
        torch.from_numpy(target),
    )
    tensors = tuple(
        _pin_if_requested(tensor, pin_memory) for tensor in tensors
    )
    return (*tensors, dates, stations)


# Kept outside the class body to make the vectorized path easy to benchmark
# independently from the single-sample compatibility API.
MultiStationWindowDataset.get_batch = _dataset_get_batch
