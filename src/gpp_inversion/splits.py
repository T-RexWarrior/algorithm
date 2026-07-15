"""Station-level split helpers extracted from the later notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class FileSplits:
    train: tuple[Path, ...]
    val: tuple[Path, ...]
    test: tuple[Path, ...]
    ignored: tuple[Path, ...]
    missing_train_sites: tuple[str, ...] = ()
    missing_val_sites: tuple[str, ...] = ()
    missing_test_sites: tuple[str, ...] = ()


def validate_site_splits(
    train_sites: Sequence[str],
    val_sites: Sequence[str],
    test_sites: Sequence[str],
) -> None:
    """Reject overlapping station IDs before any data is loaded."""
    groups = {
        "train/val": set(train_sites) & set(val_sites),
        "train/test": set(train_sites) & set(test_sites),
        "val/test": set(val_sites) & set(test_sites),
    }
    overlaps = {name: sorted(values) for name, values in groups.items() if values}
    if overlaps:
        details = "; ".join(f"{name}: {values}" for name, values in overlaps.items())
        raise ValueError(f"Station splits overlap ({details})")


def _matching_site(stem: str, site_ids: Sequence[str]) -> str | None:
    for site_id in site_ids:
        if stem == site_id or stem.startswith(site_id + "_"):
            return site_id
    return None


def split_files_by_sites(
    all_files: Iterable[str | Path],
    train_sites: Sequence[str],
    val_sites: Sequence[str],
    test_sites: Sequence[str],
    *,
    strict: bool = True,
) -> FileSplits:
    """Match CSV filenames to mutually exclusive station lists."""
    validate_site_splits(train_sites, val_sites, test_sites)
    buckets: dict[str, list[Path]] = {"train": [], "val": [], "test": [], "ignored": []}
    matched = {"train": set(), "val": set(), "test": set()}

    for raw_path in sorted(Path(path) for path in all_files):
        stem = raw_path.stem
        # Test and validation take priority, matching the latest notebooks.
        for name, site_ids in (
            ("test", test_sites),
            ("val", val_sites),
            ("train", train_sites),
        ):
            site = _matching_site(stem, site_ids)
            if site is not None:
                buckets[name].append(raw_path)
                matched[name].add(site)
                break
        else:
            buckets["ignored"].append(raw_path)

    missing = {
        "train": tuple(sorted(set(train_sites) - matched["train"])),
        "val": tuple(sorted(set(val_sites) - matched["val"])),
        "test": tuple(sorted(set(test_sites) - matched["test"])),
    }
    if strict:
        configured = {
            "train": train_sites,
            "val": val_sites,
            "test": test_sites,
        }
        empty = [
            name for name in ("train", "val", "test")
            if configured[name] and not buckets[name]
        ]
        if empty:
            raise ValueError(f"No CSV files matched split(s): {', '.join(empty)}")
        missing_groups = {name: values for name, values in missing.items() if values}
        if missing_groups:
            raise ValueError(f"Configured stations without CSV files: {missing_groups}")

    return FileSplits(
        train=tuple(buckets["train"]),
        val=tuple(buckets["val"]),
        test=tuple(buckets["test"]),
        ignored=tuple(buckets["ignored"]),
        missing_train_sites=missing["train"],
        missing_val_sites=missing["val"],
        missing_test_sites=missing["test"],
    )


def infer_site_land_cover_labels(
    all_files: Iterable[str | Path],
    site_ids: Sequence[str],
    land_cover_column: str,
) -> tuple[int, ...]:
    """Infer one modal land-cover label per station for stratified folds."""
    paths = tuple(Path(path) for path in all_files)
    labels: list[int] = []
    for site_id in site_ids:
        matching = [path for path in paths if _matching_site(path.stem, [site_id])]
        if not matching:
            raise ValueError(f"No CSV file matched station {site_id!r}")
        station_values = []
        for path in matching:
            try:
                values = pd.read_csv(
                    path,
                    usecols=[land_cover_column],
                    low_memory=False,
                    memory_map=True,
                )[land_cover_column]
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"Cannot read {land_cover_column!r} from {path}: {exc}"
                ) from exc
            numeric = pd.to_numeric(values, errors="coerce").dropna().astype(int)
            station_values.extend(numeric.tolist())
        if not station_values:
            raise ValueError(
                f"Station {site_id!r} has no valid {land_cover_column!r} values"
            )
        modes = pd.Series(station_values).mode()
        labels.append(int(modes.iloc[0]))
    return tuple(labels)


def stratified_site_folds(
    site_ids: Sequence[str],
    land_cover_labels: Sequence[int],
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Yield station-level train/validation folds stratified by land cover."""
    if len(site_ids) != len(land_cover_labels):
        raise ValueError("site_ids and land_cover_labels must have equal length")
    site_array = np.asarray(site_ids)
    labels = np.asarray(land_cover_labels)
    _, class_counts = np.unique(labels, return_counts=True)
    if class_counts.size and int(class_counts.min()) < n_splits:
        raise ValueError(
            "Every land-cover class must contain at least n_splits stations; "
            f"minimum class count is {int(class_counts.min())}, n_splits={n_splits}"
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, val_index in splitter.split(site_array, labels):
        yield tuple(site_array[train_index]), tuple(site_array[val_index])


# Compatibility names used by the historical notebooks.
check_site_overlap = validate_site_splits


def split_files_by_manual_sites(all_files, train_sites, val_sites, test_sites):
    result = split_files_by_sites(
        all_files, train_sites, val_sites, test_sites, strict=True
    )
    return list(result.train), list(result.val), list(result.test)
