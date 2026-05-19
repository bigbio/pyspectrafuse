"""Cluster DB builders for peptide-independent mzML clustering."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pyspectrafuse.common.schemas import (
    NOID_CLUSTER_META_SCHEMA,
    SPECTRUM_MEMBERSHIP_SCHEMA,
)
from pyspectrafuse.incremental.resolve_clusters_dat import (
    resolve_incremental_clusters,
)
from pyspectrafuse.common.noid_quality import parse_cluster_tsv
from pyspectrafuse.consensus_strategy.noid_consensus_strategy import (
    build_noid_consensus,
)


def build_noid_cluster_db(
    cluster_tsv: str,
    metrics_tsv: str,
    spectra_path: str,
    species: str,
    instrument: str,
    charge_str: str,
    output_dir: str,
    method_type: str = "most",
) -> Tuple[str, str]:
    """Build fresh no-ID cluster metadata and raw-spectrum membership."""
    charge_int = int(str(charge_str).replace("charge", ""))
    clusters = parse_cluster_tsv(cluster_tsv)
    spectra = pd.read_parquet(spectra_path)
    membership = clusters.merge(spectra, on="scannr", how="inner")
    if membership.empty:
        raise ValueError("No spectra matched the no-ID cluster assignments")

    raw_to_resolved = {
        str(cluster_id): str(uuid.uuid4())
        for cluster_id in membership["cluster_id"].astype(str).unique()
    }
    membership["cluster_id"] = membership["cluster_id"].astype(str).map(raw_to_resolved)
    metrics = _read_metrics(metrics_tsv, raw_to_resolved)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    membership_path = str(out_dir / "spectrum_cluster_membership.parquet")
    meta_path = str(out_dir / "cluster_metadata.parquet")

    membership_out = membership[
        [
            "cluster_id",
            "usi",
            "project_accession",
            "reference_file_name",
            "scan",
            "charge",
            "precursor_mz",
            "species",
            "instrument",
        ]
    ].copy()
    _write_membership(membership_out, membership_path)
    meta = _build_metadata(
        spectra_membership=membership,
        metrics=metrics,
        species=species,
        instrument=instrument,
        charge_int=charge_int,
        method_type=method_type,
        reused_ids=set(),
    )
    _write_metadata(meta, meta_path)
    return meta_path, membership_path


def merge_into_existing_noid_db(
    cluster_tsv: str,
    metrics_tsv: str,
    scan_titles_files: List[str],
    spectra_path: str,
    existing_metadata_path: str,
    existing_membership_path: str,
    species: str,
    instrument: str,
    charge_str: str,
    output_dir: str,
    method_type: str = "most",
) -> Tuple[str, str]:
    """Merge one no-ID incremental round into an existing no-ID DB."""
    _validate_noid_existing_db(existing_metadata_path, existing_membership_path)
    charge_int = int(str(charge_str).replace("charge", ""))
    id_map, new_rows = resolve_incremental_clusters(cluster_tsv, scan_titles_files)
    spectra = pd.read_parquet(spectra_path)
    new_rows = new_rows.dropna(subset=["run_file_name", "orig_scan"]).copy()
    new_rows["orig_scan"] = new_rows["orig_scan"].astype(int)
    new_spectra = new_rows.merge(
        spectra,
        left_on=["run_file_name", "orig_scan"],
        right_on=["reference_file_name", "scan"],
        how="inner",
    )
    if new_spectra.empty:
        raise ValueError("No new mzML spectra matched incremental cluster assignments")
    new_spectra["cluster_id"] = new_spectra["resolved_cluster_id"].astype(str)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    membership_path = str(out_dir / "spectrum_cluster_membership.parquet")
    meta_path = str(out_dir / "cluster_metadata.parquet")

    existing_membership = pd.read_parquet(existing_membership_path)
    membership_out = pd.concat([
        existing_membership,
        new_spectra[
            [
                "cluster_id",
                "usi",
                "project_accession",
                "reference_file_name",
                "scan",
                "charge",
                "precursor_mz",
                "species",
                "instrument",
            ]
        ],
    ], ignore_index=True)
    membership_out = membership_out.drop_duplicates(subset=["usi"], keep="first")
    _write_membership(membership_out, membership_path)

    existing_meta = pd.read_parquet(existing_metadata_path).copy()
    existing_meta["cluster_id"] = existing_meta["cluster_id"].astype(str)
    existing_ids = set(existing_meta["cluster_id"])
    metrics = _read_metrics(metrics_tsv, id_map)

    current_round_rows = []
    rep_lookup = existing_meta.set_index("cluster_id")
    for cluster_id, group in new_spectra.groupby("cluster_id", sort=True):
        rows = group.copy()
        if cluster_id in rep_lookup.index:
            rep = rep_lookup.loc[cluster_id]
            rep_row = {
                "cluster_id": cluster_id,
                "precursor_mz": float(rep["precursor_mz"]),
                "charge": int(rep["charge"]),
                "mz_array": rep["consensus_mz_array"],
                "intensity_array": rep["consensus_intensity_array"],
                "project_accession": "",
                "usi": f"rep:{cluster_id}",
                "reference_file_name": "",
                "scan": -1,
                "species": species,
                "instrument": instrument,
            }
            rows = pd.concat([pd.DataFrame([rep_row]), rows], ignore_index=True)
        rows["cluster_id"] = cluster_id
        current_round_rows.append(rows)
    current_round = pd.concat(current_round_rows, ignore_index=True)

    updated_meta = _build_metadata(
        spectra_membership=current_round,
        metrics=metrics,
        species=species,
        instrument=instrument,
        charge_int=charge_int,
        method_type=method_type,
        reused_ids=existing_ids,
    )
    untouched_meta = existing_meta[
        ~existing_meta["cluster_id"].isin(set(updated_meta["cluster_id"]))
    ]
    meta = pd.concat([untouched_meta, updated_meta], ignore_index=True)
    counts = _membership_stats(membership_path)
    meta = meta.drop(columns=["member_count", "project_count", "source_datasets"])
    meta = meta.merge(counts, on="cluster_id", how="left")
    meta["member_count"] = meta["member_count"].fillna(0).astype(np.int32)
    meta["project_count"] = meta["project_count"].fillna(0).astype(np.int16)
    meta["source_datasets"] = meta["source_datasets"].apply(
        lambda x: list(x) if isinstance(x, (list, np.ndarray)) else []
    )
    _write_metadata(meta, meta_path)
    return meta_path, membership_path


def _read_metrics(metrics_tsv: str, id_map: Dict[str, str]) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_tsv, sep="\t")
    metrics["cluster_id"] = metrics["cluster_id"].astype(str).map(id_map)
    return metrics.dropna(subset=["cluster_id"])


def _build_metadata(
    spectra_membership: pd.DataFrame,
    metrics: pd.DataFrame,
    species: str,
    instrument: str,
    charge_int: int,
    method_type: str,
    reused_ids: set,
) -> pd.DataFrame:
    counts = (
        spectra_membership[spectra_membership["usi"].astype(str).str.startswith("mzspec:")]
        .groupby("cluster_id", sort=True)
        .agg(
            member_count=("usi", "count"),
            project_count=("project_accession", "nunique"),
            source_datasets=("project_accession", lambda x: sorted(set(x))),
        )
        .reset_index()
    )

    rows = []
    for cluster_id, group in spectra_membership.groupby("cluster_id", sort=True):
        consensus = build_noid_consensus(group, method_type=method_type)
        rows.append({
            "cluster_id": str(cluster_id),
            "species": species,
            "instrument": instrument,
            "charge": charge_int,
            "consensus_mz_array": consensus["mz_array"].astype(np.float32).tolist(),
            "consensus_intensity_array": (
                consensus["intensity_array"].astype(np.float32).tolist()
            ),
            "consensus_method": method_type,
            "precursor_mz": float(consensus["precursor_mz"]),
            "is_reused_cluster": str(cluster_id) in reused_ids,
        })
    meta = pd.DataFrame(rows)
    meta = meta.merge(counts, on="cluster_id", how="left")
    meta = meta.merge(
        metrics[["cluster_id", "cluster_quality_ratio", "mean_similarity"]],
        on="cluster_id",
        how="left",
    )
    meta["member_count"] = meta["member_count"].fillna(0).astype(np.int32)
    meta["project_count"] = meta["project_count"].fillna(0).astype(np.int16)
    meta["cluster_quality_ratio"] = meta["cluster_quality_ratio"].fillna(1.0)
    meta["mean_similarity"] = meta["mean_similarity"].fillna(1.0)
    meta["source_datasets"] = meta["source_datasets"].apply(
        lambda x: list(x) if isinstance(x, (list, np.ndarray)) else []
    )
    return meta


def _membership_stats(membership_path: str) -> pd.DataFrame:
    conn = duckdb.connect(":memory:")
    try:
        return conn.execute(f"""
            SELECT
                cluster_id,
                CAST(COUNT(*) AS INTEGER) AS member_count,
                CAST(COUNT(DISTINCT project_accession) AS SMALLINT) AS project_count,
                list_distinct(array_agg(project_accession)) AS source_datasets
            FROM read_parquet('{membership_path}')
            GROUP BY cluster_id
        """).fetchdf()
    finally:
        conn.close()


def _validate_noid_existing_db(
    metadata_path: str,
    membership_path: str,
) -> None:
    metadata_cols = set(pd.read_parquet(metadata_path).columns)
    membership_cols = set(pd.read_parquet(membership_path).columns)
    required_meta = set(NOID_CLUSTER_META_SCHEMA.names)
    required_membership = set(SPECTRUM_MEMBERSHIP_SCHEMA.names)
    if not required_meta.issubset(metadata_cols):
        raise ValueError("existing_cluster_db is not a no-ID cluster DB")
    if not required_membership.issubset(membership_cols):
        raise ValueError("existing_cluster_db is not a no-ID cluster DB")


def _write_metadata(df: pd.DataFrame, output_path: str) -> None:
    table = pa.Table.from_pandas(
        df[NOID_CLUSTER_META_SCHEMA.names],
        schema=NOID_CLUSTER_META_SCHEMA,
        preserve_index=False,
    )
    pq.write_table(table, output_path, compression="zstd")


def _write_membership(df: pd.DataFrame, output_path: str) -> None:
    table = pa.Table.from_pandas(
        df[SPECTRUM_MEMBERSHIP_SCHEMA.names],
        schema=SPECTRUM_MEMBERSHIP_SCHEMA,
        preserve_index=False,
    )
    pq.write_table(table, output_path, compression="zstd")
