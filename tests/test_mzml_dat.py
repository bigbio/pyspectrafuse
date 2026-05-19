"""Tests for ID-free mzML to .dat conversion."""
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from pyspectrafuse.maracluster_dat import SPECTRUM_STRUCT
from pyspectrafuse.common.mzml_dat import (
    _format_noid_usi,
    _extract_charge,
    _extract_precursor_mz,
    _format_noid_scan_title,
    _scan_number_from_id,
    build_mzml_partition_table,
    merge_mzml_charge_shards,
    spectra_to_dat_by_charge,
)


def _make_workdir():
    path = Path.cwd() / f"test_tmp_{uuid.uuid4().hex}"
    path.mkdir()
    return path


def _spectrum(scan: int, charge: int, precursor_mz: float, n_peaks: int = 40):
    mzs = np.linspace(100.0, 400.0, n_peaks)
    intensities = np.linspace(1000.0, 10.0, n_peaks)
    return {
        "id": f"controllerType=0 controllerNumber=1 scan={scan}",
        "ms level": 2,
        "m/z array": mzs,
        "intensity array": intensities,
        "scanList": {"scan": [{"scan start time": 12.5}]},
        "precursorList": {
            "precursor": [{
                "selectedIonList": {
                    "selectedIon": [{
                        "selected ion m/z": precursor_mz,
                        "charge state": charge,
                    }]
                }
            }]
        },
    }


def test_extract_mzml_precursor_fields():
    spec = _spectrum(scan=42, charge=3, precursor_mz=799.5)

    assert _scan_number_from_id(spec["id"], fallback=0) == 42
    assert _extract_charge(spec) == 3
    assert _extract_precursor_mz(spec) == 799.5


def test_format_noid_scan_title_omits_unknown_peptide():
    title = _format_noid_scan_title("run01.mzML", orig_scan=7193, charge=6)

    assert title == "id=mzspec::run01.mzML:scan:7193:charge6"
    assert "UNKNOWN" not in title


def test_format_noid_usi_omits_peptide_content():
    usi = _format_noid_usi("PXD1", "run01.mzML", orig_scan=7, charge=3)

    assert usi == "mzspec:PXD1:run01.mzML:scan:7:charge3"


def test_spectra_to_dat_by_charge_writes_charge_partitions():
    spectra = [
        {"ms level": 1},
        _spectrum(scan=2, charge=2, precursor_mz=600.0),
        _spectrum(scan=3, charge=3, precursor_mz=700.0),
        _spectrum(scan=4, charge=2, precursor_mz=600.0, n_peaks=4),
    ]

    temp_dir = _make_workdir()
    try:
        result = spectra_to_dat_by_charge(
            spectra,
            source_name="run01.mzML",
            output_dir=str(temp_dir),
            file_idx=7,
            charges=(2, 3),
        )

        by_charge = {r.charge: r for r in result.charge_results}
        assert by_charge[2].written == 1
        assert by_charge[2].skipped == 1
        assert by_charge[3].written == 1

        dat_path = Path(by_charge[2].dat_path)
        assert dat_path.stat().st_size == SPECTRUM_STRUCT.size
        fields = SPECTRUM_STRUCT.unpack(dat_path.read_bytes())
        assert fields[0] == 7
        assert fields[1] == 0
        assert fields[2] == 2
        assert abs(fields[3] - 600.0) < 0.01

        titles = Path(by_charge[2].titles_path).read_text()
        assert "id=mzspec::run01.mzML:scan:2:charge2" in titles
        assert "UNKNOWN" not in titles
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_merge_mzml_charge_shards_outputs_charge_only_partitions():
    temp_dir = _make_workdir()
    try:
        shard_dir = temp_dir / "shards"
        final_dir = temp_dir / "final"
        shard_dir.mkdir()

        run01 = spectra_to_dat_by_charge(
            [_spectrum(scan=2, charge=2, precursor_mz=600.0)],
            source_name="run01.mzML",
            output_dir=str(shard_dir),
            file_idx=10,
            charges=(2,),
        )
        run02 = spectra_to_dat_by_charge(
            [_spectrum(scan=5, charge=2, precursor_mz=610.0)],
            source_name="run02.mzML",
            output_dir=str(shard_dir),
            file_idx=11,
            charges=(2,),
        )

        results = merge_mzml_charge_shards(
            [run02, run01],
            output_dir=str(final_dir),
            charges=(2,),
            file_idx_start=0,
        )

        assert len(results) == 1
        assert results[0].written == 2
        assert Path(results[0].dat_path).name == "charge2.dat"
        assert not (final_dir / "run01_charge2.dat").exists()

        data = Path(results[0].dat_path).read_bytes()
        first = SPECTRUM_STRUCT.unpack_from(data, 0)
        second = SPECTRUM_STRUCT.unpack_from(data, SPECTRUM_STRUCT.size)
        assert first[0] == 0
        assert first[1] == 0
        assert second[0] == 0
        assert second[1] == 1

        titles = Path(results[0].titles_path).read_text().splitlines()
        assert titles == [
            "0\t0\tid=mzspec::run01.mzML:scan:2:charge2",
            "0\t1\tid=mzspec::run02.mzML:scan:5:charge2",
        ]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_merge_mzml_charge_shards_writes_renumbered_sidecar():
    temp_dir = _make_workdir()
    try:
        shard_dir = temp_dir / "shards"
        final_dir = temp_dir / "final"
        shard_dir.mkdir()

        run01 = spectra_to_dat_by_charge(
            [_spectrum(scan=2, charge=2, precursor_mz=600.0)],
            source_name="run01.mzML",
            output_dir=str(shard_dir),
            file_idx=10,
            charges=(2,),
            dataset_name="PXD1",
            species="Homo sapiens",
            instrument="Orbitrap",
        )
        run02 = spectra_to_dat_by_charge(
            [_spectrum(scan=5, charge=2, precursor_mz=610.0)],
            source_name="run02.mzML",
            output_dir=str(shard_dir),
            file_idx=11,
            charges=(2,),
            dataset_name="PXD1",
            species="Homo sapiens",
            instrument="Orbitrap",
        )

        results = merge_mzml_charge_shards(
            [run02, run01],
            output_dir=str(final_dir),
            charges=(2,),
        )
        sidecar = pd.read_parquet(results[0].spectra_path)

        assert sidecar["scannr"].tolist() == [0, 1]
        assert sidecar["usi"].tolist() == [
            "mzspec:PXD1:run01.mzML:scan:2:charge2",
            "mzspec:PXD1:run02.mzML:scan:5:charge2",
        ]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_build_mzml_partition_table_uses_sdrf_then_defaults():
    temp_dir = _make_workdir()
    try:
        mzml_dir = temp_dir / "mzml"
        mzml_dir.mkdir()
        (mzml_dir / "run01.mzML").write_text("")
        (mzml_dir / "run02.mzML").write_text("")
        sdrf = temp_dir / "meta.sdrf.tsv"
        pd.DataFrame({
            "comment[data file]": ["run01.mzML"],
            "characteristics[organism]": ["Homo sapiens"],
            "comment[instrument]": ["NT=Orbitrap Fusion;"],
        }).to_csv(sdrf, sep="\t", index=False)

        table = build_mzml_partition_table(
            str(mzml_dir),
            sdrf_path=str(sdrf),
            default_species="Mus musculus",
            default_instrument="Q Exactive",
        )
        by_file = table.set_index("mzml_file")

        assert by_file.loc["run01.mzML", "species"] == "Homo sapiens"
        assert by_file.loc["run01.mzML", "instrument"] == "Orbitrap Fusion"
        assert by_file.loc["run02.mzML", "species"] == "Mus musculus"
        assert by_file.loc["run02.mzML", "instrument"] == "Q Exactive"
        assert table["partition_id"].nunique() == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
