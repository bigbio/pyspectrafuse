"""Tests for ID-free cluster quality filtering."""
from pathlib import Path
import shutil
import uuid

from pyspectrafuse.maracluster_dat import SPECTRUM_STRUCT, _pack_spectrum
from pyspectrafuse.common.noid_quality import (
    compute_cluster_quality,
    filter_cluster_tsv_by_quality,
)


def _make_workdir():
    path = Path.cwd() / f"test_tmp_{uuid.uuid4().hex}"
    path.mkdir()
    return path


def test_filter_cluster_tsv_by_quality_keeps_coherent_clusters():
    workdir = _make_workdir()
    try:
        dat_path = workdir / "sample_charge2.dat"
        records = [
            _pack_spectrum(0, 0, 2, 500.0, 0.0, list(range(10, 30))),
            _pack_spectrum(0, 1, 2, 500.1, 0.0, list(range(10, 30))),
            _pack_spectrum(0, 2, 2, 800.0, 0.0, list(range(100, 120))),
        ]
        dat_path.write_bytes(b"".join(records))
        assert dat_path.stat().st_size == 3 * SPECTRUM_STRUCT.size

        cluster_tsv = workdir / "clusters.tsv"
        cluster_tsv.write_text(
            "sample_charge2.mgf\t0\tA\n"
            "sample_charge2.mgf\t1\tA\n"
            "sample_charge2.mgf\t2\tB\n"
        )

        metrics = compute_cluster_quality(
            str(cluster_tsv),
            [str(dat_path)],
            min_pair_similarity=0.95,
        )
        quality = dict(zip(metrics["cluster_id"], metrics["cluster_quality_ratio"]))
        assert quality["A"] == 1.0
        assert quality["B"] == 1.0

        output_tsv = workdir / "filtered.tsv"
        metrics_tsv = workdir / "quality.tsv"
        _, _, kept = filter_cluster_tsv_by_quality(
            cluster_tsv=str(cluster_tsv),
            dat_paths=[str(dat_path)],
            output_tsv=str(output_tsv),
            metrics_tsv=str(metrics_tsv),
            min_quality_ratio=0.95,
            min_pair_similarity=0.95,
            min_cluster_size=2,
        )

        assert kept == 1
        assert output_tsv.read_text() == "sample_charge2.mgf\t0\tA\nsample_charge2.mgf\t1\tA\n"
        assert Path(metrics_tsv).exists()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
