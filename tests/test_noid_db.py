"""Tests for no-ID DB, MSP, and incremental no-ID helpers."""
from pathlib import Path
import gzip
import shutil
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pyspectrafuse.common.schemas import (
    NOID_CLUSTER_META_SCHEMA,
    NOID_SPECTRUM_SCHEMA,
    SPECTRUM_MEMBERSHIP_SCHEMA,
)
from pyspectrafuse.common.noid_cluster_db import (
    build_noid_cluster_db,
    merge_into_existing_noid_db,
)
from pyspectrafuse.common.noid_msp_utils import write_noid_msp


def _make_workdir():
    path = Path.cwd() / f"test_tmp_{uuid.uuid4().hex}"
    path.mkdir()
    return path


def _write_sidecar(path: Path, rows):
    table = pa.Table.from_pylist(rows, schema=NOID_SPECTRUM_SCHEMA)
    pq.write_table(table, path)


def _spectrum_row(scannr, scan, mz_base, cluster_suffix=""):
    return {
        "scannr": scannr,
        "usi": f"mzspec:PXD1:run01.mzML:scan:{scan}:charge2",
        "project_accession": "PXD1",
        "reference_file_name": "run01.mzML",
        "scan": scan,
        "charge": 2,
        "precursor_mz": 500.0 + scan,
        "retention_time": 12.5,
        "mz_array": [mz_base, mz_base + 100.0, mz_base + 200.0],
        "intensity_array": [1000.0, 800.0, 600.0],
        "species": "Homo sapiens",
        "instrument": "Orbitrap",
    }


def test_build_noid_cluster_db_and_msp_have_no_peptide_fields():
    workdir = _make_workdir()
    try:
        sidecar = workdir / "charge2.spectra.parquet"
        _write_sidecar(sidecar, [
            _spectrum_row(0, 10, 100.0),
            _spectrum_row(1, 11, 100.0),
        ])
        cluster_tsv = workdir / "clusters.tsv"
        cluster_tsv.write_text("charge2.mgf\t0\tA\ncharge2.mgf\t1\tA\n")
        metrics_tsv = workdir / "metrics.tsv"
        pd.DataFrame([{
            "cluster_id": "A",
            "member_count": 2,
            "pair_count": 1,
            "mean_similarity": 1.0,
            "cluster_quality_ratio": 1.0,
            "pass_quality": True,
        }]).to_csv(metrics_tsv, sep="\t", index=False)

        meta_path, membership_path = build_noid_cluster_db(
            cluster_tsv=str(cluster_tsv),
            metrics_tsv=str(metrics_tsv),
            spectra_path=str(sidecar),
            species="Homo sapiens",
            instrument="Orbitrap",
            charge_str="charge2",
            output_dir=str(workdir / "db"),
        )

        meta = pd.read_parquet(meta_path)
        membership = pd.read_parquet(membership_path)
        assert "peptidoform" not in meta.columns
        assert "best_pep" not in meta.columns
        assert set(meta.columns) == set(NOID_CLUSTER_META_SCHEMA.names)
        assert set(membership.columns) == set(SPECTRUM_MEMBERSHIP_SCHEMA.names)
        assert meta.loc[0, "member_count"] == 2

        msp_path = write_noid_msp(
            cluster_metadata_path=meta_path,
            output_dir=str(workdir / "msp"),
            dataset_name="PXD1",
        )
        with gzip.open(msp_path, "rt") as handle:
            text = handle.read()
        assert "Name:" in text
        assert "qualityRatio=1.0" in text
        assert "PEP=" not in text
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_merge_into_existing_noid_db_reuses_cluster_id():
    workdir = _make_workdir()
    try:
        existing_meta = pd.DataFrame([{
            "cluster_id": "OLD_1",
            "species": "Homo sapiens",
            "instrument": "Orbitrap",
            "charge": 2,
            "consensus_mz_array": [100.0, 200.0, 300.0],
            "consensus_intensity_array": [1000.0, 800.0, 600.0],
            "consensus_method": "most",
            "precursor_mz": 500.0,
            "member_count": 1,
            "project_count": 1,
            "cluster_quality_ratio": 1.0,
            "mean_similarity": 1.0,
            "is_reused_cluster": False,
            "source_datasets": ["PXD0"],
        }])
        existing_meta_path = workdir / "existing_meta.parquet"
        pq.write_table(
            pa.Table.from_pandas(
                existing_meta,
                schema=NOID_CLUSTER_META_SCHEMA,
                preserve_index=False,
            ),
            existing_meta_path,
        )
        existing_membership = pd.DataFrame([{
            "cluster_id": "OLD_1",
            "usi": "mzspec:PXD0:run00.mzML:scan:1:charge2",
            "project_accession": "PXD0",
            "reference_file_name": "run00.mzML",
            "scan": 1,
            "charge": 2,
            "precursor_mz": 500.0,
            "species": "Homo sapiens",
            "instrument": "Orbitrap",
        }])
        existing_membership_path = workdir / "existing_membership.parquet"
        pq.write_table(
            pa.Table.from_pandas(
                existing_membership,
                schema=SPECTRUM_MEMBERSHIP_SCHEMA,
                preserve_index=False,
            ),
            existing_membership_path,
        )

        sidecar = workdir / "charge2.spectra.parquet"
        _write_sidecar(sidecar, [_spectrum_row(0, 10, 100.0)])
        combined_titles = workdir / "combined.scan_titles.txt"
        combined_titles.write_text(
            "0\t0\trep:OLD_1\n"
            "0\t1\tid=mzspec::run01.mzML:scan:10:charge2\n"
        )
        cluster_tsv = workdir / "clusters.tsv"
        cluster_tsv.write_text(
            "combined.mgf\t0\tNEW_A\n"
            "combined.mgf\t1\tNEW_A\n"
        )
        metrics_tsv = workdir / "metrics.tsv"
        pd.DataFrame([{
            "cluster_id": "NEW_A",
            "member_count": 2,
            "pair_count": 1,
            "mean_similarity": 1.0,
            "cluster_quality_ratio": 1.0,
            "pass_quality": True,
        }]).to_csv(metrics_tsv, sep="\t", index=False)

        meta_path, membership_path = merge_into_existing_noid_db(
            cluster_tsv=str(cluster_tsv),
            metrics_tsv=str(metrics_tsv),
            scan_titles_files=[str(combined_titles)],
            spectra_path=str(sidecar),
            existing_metadata_path=str(existing_meta_path),
            existing_membership_path=str(existing_membership_path),
            species="Homo sapiens",
            instrument="Orbitrap",
            charge_str="charge2",
            output_dir=str(workdir / "merged"),
        )
        meta = pd.read_parquet(meta_path)
        membership = pd.read_parquet(membership_path)

        assert meta["cluster_id"].tolist() == ["OLD_1"]
        assert bool(meta.loc[0, "is_reused_cluster"]) is True
        assert meta.loc[0, "member_count"] == 2
        assert sorted(membership["project_accession"].tolist()) == ["PXD0", "PXD1"]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
