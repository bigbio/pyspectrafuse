"""ID-free cluster quality filtering for MaRaCluster .dat results."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import re
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from pyspectrafuse.maracluster_dat import SPECTRUM_STRUCT


def parse_cluster_tsv(cluster_tsv: str) -> pd.DataFrame:
    """Parse a MaRaCluster TSV into normalized cluster assignments."""
    df = pd.read_csv(
        cluster_tsv,
        sep="\t",
        header=None,
        names=["mgf_path", "scannr", "cluster_id"],
        skip_blank_lines=False,
    ).dropna()
    df["scannr"] = df["scannr"].astype(int)
    df["cluster_id"] = df["cluster_id"].astype(str)
    df["mgf_basename"] = df["mgf_path"].apply(
        lambda p: re.sub(r"_w\d+\.mgf$", ".mgf", Path(str(p)).name)
    )
    return df[["mgf_path", "mgf_basename", "scannr", "cluster_id"]]


def read_dat_bin_index(dat_paths: Sequence[str]) -> pd.DataFrame:
    """Read binned spectra from .dat files for quality calculations."""
    rows = []
    for dat_path in dat_paths:
        path = Path(dat_path)
        if path.name.endswith(".scan_info.dat"):
            continue
        mgf_basename = f"{path.stem}.mgf"
        data = path.read_bytes()
        n_spectra = len(data) // SPECTRUM_STRUCT.size
        for i in range(n_spectra):
            fields = SPECTRUM_STRUCT.unpack_from(data, i * SPECTRUM_STRUCT.size)
            bins = tuple(int(b) for b in fields[5:] if int(b) > 0)
            rows.append({
                "mgf_basename": mgf_basename,
                "scannr": int(fields[1]),
                "charge": int(fields[2]),
                "precursor_mz": float(fields[3]),
                "frag_bins": bins,
            })
    return pd.DataFrame(rows)


def compute_cluster_quality(
    cluster_tsv: str,
    dat_paths: Sequence[str],
    min_pair_similarity: float = 0.7,
    max_pairs_per_cluster: int = 10_000,
) -> pd.DataFrame:
    """Compute an ID-free cluster quality ratio.

    The ratio is the fraction of pairwise member comparisons whose binary-bin
    cosine similarity is at least ``min_pair_similarity``. Large clusters are
    deterministically sampled to ``max_pairs_per_cluster`` pairs.
    """
    clusters = parse_cluster_tsv(cluster_tsv)
    bins = read_dat_bin_index(dat_paths)
    if clusters.empty or bins.empty:
        return pd.DataFrame(columns=[
            "cluster_id", "member_count", "pair_count",
            "mean_similarity", "cluster_quality_ratio",
        ])

    merged = clusters.merge(bins, on=["mgf_basename", "scannr"], how="inner")
    metrics = []
    for cluster_id, group in merged.groupby("cluster_id", sort=True):
        bin_sets = [frozenset(v) for v in group["frag_bins"]]
        member_count = len(bin_sets)
        similarities = _pairwise_similarities(bin_sets, max_pairs_per_cluster)
        if len(similarities) == 0:
            mean_similarity = 1.0
            quality_ratio = 1.0
            pair_count = 0
        else:
            sims = np.asarray(similarities, dtype=np.float32)
            mean_similarity = float(sims.mean())
            quality_ratio = float((sims >= min_pair_similarity).mean())
            pair_count = int(len(sims))

        metrics.append({
            "cluster_id": cluster_id,
            "member_count": int(member_count),
            "pair_count": pair_count,
            "mean_similarity": mean_similarity,
            "cluster_quality_ratio": quality_ratio,
        })

    return pd.DataFrame(metrics)


def filter_cluster_tsv_by_quality(
    cluster_tsv: str,
    dat_paths: Sequence[str],
    output_tsv: str,
    metrics_tsv: str,
    min_quality_ratio: float = 0.5,
    min_pair_similarity: float = 0.7,
    min_cluster_size: int = 1,
    max_pairs_per_cluster: int = 10_000,
) -> Tuple[str, str, int]:
    """Filter MaRaCluster assignments by ID-free quality metrics."""
    metrics = compute_cluster_quality(
        cluster_tsv=cluster_tsv,
        dat_paths=dat_paths,
        min_pair_similarity=min_pair_similarity,
        max_pairs_per_cluster=max_pairs_per_cluster,
    )
    if metrics.empty:
        Path(output_tsv).parent.mkdir(parents=True, exist_ok=True)
        Path(output_tsv).write_text("")
        metrics.to_csv(metrics_tsv, sep="\t", index=False)
        return output_tsv, metrics_tsv, 0

    metrics["pass_quality"] = (
        (metrics["member_count"] >= min_cluster_size)
        & (metrics["cluster_quality_ratio"] >= min_quality_ratio)
    )
    passing = set(metrics.loc[metrics["pass_quality"], "cluster_id"].astype(str))
    clusters = parse_cluster_tsv(cluster_tsv)
    filtered = clusters[clusters["cluster_id"].astype(str).isin(passing)]

    Path(output_tsv).parent.mkdir(parents=True, exist_ok=True)
    filtered[["mgf_path", "scannr", "cluster_id"]].to_csv(
        output_tsv, sep="\t", header=False, index=False
    )
    metrics.to_csv(metrics_tsv, sep="\t", index=False)
    return output_tsv, metrics_tsv, len(passing)


def _pairwise_similarities(
    bin_sets: Sequence[frozenset],
    max_pairs: int,
) -> List[float]:
    n = len(bin_sets)
    if n < 2:
        return []

    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        pair_iter: Iterable[Tuple[int, int]] = combinations(range(n), 2)
    else:
        rng = np.random.default_rng(0)
        sampled = set()
        while len(sampled) < max_pairs:
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n - 1))
            if j >= i:
                j += 1
            sampled.add((min(i, j), max(i, j)))
        pair_iter = sorted(sampled)

    return [_binary_bin_cosine(bin_sets[i], bin_sets[j]) for i, j in pair_iter]


def _binary_bin_cosine(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(np.sqrt(len(a) * len(b)))
