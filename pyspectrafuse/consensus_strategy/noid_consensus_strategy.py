"""Peptide-independent consensus spectrum generation for no-ID workflows."""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from pyspectrafuse.maracluster_dat import PROTON_MASS, _bin_spectrum
from pyspectrafuse.common.noid_quality import _binary_bin_cosine


def build_noid_consensus(
    spectra_df: pd.DataFrame,
    method_type: str = "most",
    min_mz: float = 100.0,
    max_mz: float = 2000.0,
    bin_size: float = 0.02,
    peak_quorum: float = 0.25,
    diff_thresh: float = 0.01,
    min_fraction: float = 0.5,
) -> pd.Series:
    """Build one peptide-independent consensus spectrum for a cluster."""
    if spectra_df.empty:
        raise ValueError("spectra_df is empty")
    if method_type == "most":
        return _most_similar_member(spectra_df)
    if method_type == "bin":
        return _binned_consensus(
            spectra_df, min_mz=min_mz, max_mz=max_mz,
            bin_size=bin_size, peak_quorum=peak_quorum,
        )
    if method_type == "average":
        return _average_consensus(
            spectra_df, diff_thresh=diff_thresh, min_fraction=min_fraction
        )
    raise ValueError(
        "No-ID consensus method must be one of: most, bin, average"
    )


def _most_similar_member(spectra_df: pd.DataFrame) -> pd.Series:
    if len(spectra_df) == 1:
        return _consensus_row_from_member(spectra_df.iloc[0])

    bin_sets = []
    for row in spectra_df.itertuples(index=False):
        neutral_mass = (
            float(row.precursor_mz) * int(row.charge)
            - PROTON_MASS * (int(row.charge) - 1)
        )
        bins = _bin_spectrum(
            np.asarray(row.mz_array, dtype=np.float64),
            np.asarray(row.intensity_array, dtype=np.float64),
            neutral_mass,
        )
        bin_sets.append(frozenset(bins))

    scores = np.zeros(len(bin_sets), dtype=np.float64)
    for i, j in combinations(range(len(bin_sets)), 2):
        similarity = _binary_bin_cosine(bin_sets[i], bin_sets[j])
        scores[i] += similarity
        scores[j] += similarity
    best_idx = int(scores.argmax())
    return _consensus_row_from_member(spectra_df.iloc[best_idx])


def _binned_consensus(
    spectra_df: pd.DataFrame,
    min_mz: float,
    max_mz: float,
    bin_size: float,
    peak_quorum: float,
) -> pd.Series:
    n_bins = int(np.ceil((max_mz - min_mz) / bin_size))
    mz_sums = np.zeros(n_bins, dtype=np.float64)
    intensity_sums = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int32)

    for row in spectra_df.itertuples(index=False):
        mz = np.asarray(row.mz_array, dtype=np.float64)
        intensity = np.asarray(row.intensity_array, dtype=np.float64)
        valid = (mz >= min_mz) & (mz < max_mz)
        bins = np.floor((mz[valid] - min_mz) / bin_size).astype(np.int64)
        for idx, bin_idx in enumerate(bins):
            mz_sums[bin_idx] += mz[valid][idx]
            intensity_sums[bin_idx] += intensity[valid][idx]
            counts[bin_idx] += 1

    required = max(1, int(np.ceil(len(spectra_df) * peak_quorum)))
    keep = counts >= required
    if not np.any(keep):
        return _most_similar_member(spectra_df)

    return _consensus_row(
        precursor_mz=float(spectra_df["precursor_mz"].median()),
        charge=int(spectra_df["charge"].iloc[0]),
        mz_array=(mz_sums[keep] / counts[keep]).astype(np.float32),
        intensity_array=(intensity_sums[keep] / counts[keep]).astype(np.float32),
    )


def _average_consensus(
    spectra_df: pd.DataFrame,
    diff_thresh: float,
    min_fraction: float,
) -> pd.Series:
    all_mz = np.concatenate([
        np.asarray(v, dtype=np.float64) for v in spectra_df["mz_array"]
    ])
    all_intensity = np.concatenate([
        np.asarray(v, dtype=np.float64) for v in spectra_df["intensity_array"]
    ])
    order = np.argsort(all_mz)
    all_mz = all_mz[order]
    all_intensity = all_intensity[order]

    mz_values = []
    intensity_values = []
    min_count = max(1, int(np.ceil(len(spectra_df) * min_fraction)))
    start = 0
    while start < len(all_mz):
        end = start + 1
        while end < len(all_mz) and all_mz[end] - all_mz[start] < diff_thresh:
            end += 1
        if end - start >= min_count:
            weights = all_intensity[start:end]
            mz_values.append(float(np.average(all_mz[start:end], weights=weights)))
            intensity_values.append(float(weights.mean()))
        start = end

    if not mz_values:
        return _most_similar_member(spectra_df)

    return _consensus_row(
        precursor_mz=float(spectra_df["precursor_mz"].median()),
        charge=int(spectra_df["charge"].iloc[0]),
        mz_array=np.asarray(mz_values, dtype=np.float32),
        intensity_array=np.asarray(intensity_values, dtype=np.float32),
    )


def _consensus_row_from_member(row: pd.Series) -> pd.Series:
    return _consensus_row(
        precursor_mz=float(row["precursor_mz"]),
        charge=int(row["charge"]),
        mz_array=np.asarray(row["mz_array"], dtype=np.float32),
        intensity_array=np.asarray(row["intensity_array"], dtype=np.float32),
    )


def _consensus_row(
    precursor_mz: float,
    charge: int,
    mz_array: Iterable[float],
    intensity_array: Iterable[float],
) -> pd.Series:
    return pd.Series({
        "precursor_mz": float(precursor_mz),
        "charge": int(charge),
        "mz_array": np.asarray(mz_array, dtype=np.float32),
        "intensity_array": np.asarray(intensity_array, dtype=np.float32),
    })
