"""Convert mzML MS2 spectra to MaRaCluster-compatible .dat files.

This module is intentionally ID-free: it reads only raw spectrum evidence from
mzML files (precursor m/z, precursor charge, scan number, retention time, and
fragment peak arrays). Peptide IDs, PEPs, q-values, and search-engine fields are
not required or used.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
import logging
from pathlib import Path
import re
import shutil
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyteomics import mzml

from pyspectrafuse.common.schemas import NOID_SPECTRUM_SCHEMA
from pyspectrafuse.maracluster_dat import (
    MIN_SCORING_PEAKS,
    PROTON_MASS,
    SCANINFO_STRUCT,
    SPECTRUM_STRUCT,
    _bin_spectrum,
    _pack_scaninfo,
    _pack_spectrum,
)

logger = logging.getLogger(__name__)

DEFAULT_CHARGES = (2, 3, 4, 5, 6)


@dataclass(frozen=True)
class ChargeDatResult:
    """Summary for one charge-specific .dat output."""

    charge: int
    written: int
    skipped: int
    dat_path: str
    scaninfo_path: str
    titles_path: str
    spectra_path: Optional[str] = None


@dataclass(frozen=True)
class MzmlDatResult:
    """Summary for one converted mzML file."""

    mzml_path: str
    file_idx: int
    charge_results: Tuple[ChargeDatResult, ...]
    skipped_missing_charge: int


def find_mzml_files(mzml_dir: str) -> List[str]:
    """Find mzML files below a directory, case-insensitively."""
    root = Path(mzml_dir)
    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() == ".mzml"
    )
    return [str(p) for p in files]


def build_mzml_partition_table(
    mzml_dir: str,
    sdrf_path: Optional[str] = None,
    default_species: Optional[str] = None,
    default_instrument: Optional[str] = None,
    skip_instrument: bool = False,
) -> pd.DataFrame:
    """Map local mzML files to SpectraFuse partition metadata."""
    mzml_files = find_mzml_files(mzml_dir)
    if not mzml_files:
        raise FileNotFoundError(f"No mzML files found in {mzml_dir}")

    sdrf_df = read_sdrf_mzml_metadata(sdrf_path) if sdrf_path else pd.DataFrame()
    by_stem = {
        str(row.run_stem).lower(): row
        for row in sdrf_df.itertuples(index=False)
    } if not sdrf_df.empty else {}

    rows = []
    for mzml_file in mzml_files:
        path = Path(mzml_file)
        row = by_stem.get(path.stem.lower())
        species = (
            str(getattr(row, "species", "") or "").strip()
            or str(default_species or "").strip()
            or "Unknown"
        )
        instrument = "" if skip_instrument else (
            str(getattr(row, "instrument", "") or "").strip()
            or str(default_instrument or "").strip()
            or "Unknown"
        )
        partition_id = _partition_slug(species, instrument, skip_instrument)
        rows.append({
            "mzml_path": str(path),
            "mzml_file": path.name,
            "species": species,
            "instrument": instrument,
            "partition_id": partition_id,
        })
    return pd.DataFrame(rows)


def _partition_slug(species: str, instrument: str, skip_instrument: bool) -> str:
    parts = [species] if skip_instrument else [species, instrument]
    normalized = [
        re.sub(r"[^A-Za-z0-9]+", "_", part).strip("_") or "Unknown"
        for part in parts
    ]
    return "__".join(normalized)


def parse_charges(charges: str) -> Tuple[int, ...]:
    """Parse a charge list such as ``2,3,4`` or ``2-6``."""
    parsed: List[int] = []
    for part in str(charges).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            parsed.extend(range(int(start), int(end) + 1))
        else:
            parsed.append(int(part))
    unique = tuple(sorted(set(parsed)))
    if not unique:
        raise ValueError("At least one charge must be provided")
    if any(c <= 0 for c in unique):
        raise ValueError(f"Charges must be positive integers: {unique}")
    return unique


def _scan_number_from_id(spectrum_id: str, fallback: int) -> int:
    """Extract a scan number from common native IDs."""
    for pattern in (r"(?:^|\s)scan=(\d+)", r"(?:^|\s)index=(\d+)"):
        match = re.search(pattern, spectrum_id or "")
        if match:
            return int(match.group(1))
    return int(fallback)


def _first_selected_ion(spec: dict) -> dict:
    precursors = spec.get("precursorList", {}).get("precursor", [])
    if not precursors:
        return {}
    selected = precursors[0].get("selectedIonList", {}).get("selectedIon", [])
    return selected[0] if selected else {}


def _first_precursor(spec: dict) -> dict:
    precursors = spec.get("precursorList", {}).get("precursor", [])
    return precursors[0] if precursors else {}


def _extract_precursor_mz(spec: dict) -> Optional[float]:
    selected = _first_selected_ion(spec)
    prec_mz = selected.get("selected ion m/z")
    if prec_mz is not None:
        return float(prec_mz)

    precursor = _first_precursor(spec)
    isolation = precursor.get("isolationWindow", {})
    target_mz = isolation.get("isolation window target m/z")
    if target_mz is not None:
        return float(target_mz)

    scans = spec.get("scanList", {}).get("scan", [])
    if scans:
        thermo_mono = scans[0].get("[Thermo Trailer Extra]Monoisotopic M/Z:")
        if thermo_mono is not None:
            return float(thermo_mono)
    return None


def _extract_charge(spec: dict) -> Optional[int]:
    selected = _first_selected_ion(spec)
    charge = selected.get("charge state")
    if charge is None:
        return None
    try:
        charge_int = int(charge)
    except (TypeError, ValueError):
        return None
    return charge_int if charge_int > 0 else None


def _extract_retention_time(spec: dict) -> float:
    scans = spec.get("scanList", {}).get("scan", [])
    if not scans:
        return 0.0
    rt = scans[0].get("scan start time")
    return float(rt) if rt is not None else 0.0


def _format_noid_scan_title(source_name: str, orig_scan: int, charge: int) -> str:
    """Format an ID-free mzML scan title while preserving charge."""
    return f"id=mzspec::{source_name}:scan:{orig_scan}:charge{charge}"


def _format_noid_usi(
    dataset_name: str,
    source_name: str,
    orig_scan: int,
    charge: int,
) -> str:
    """Format a project-aware USI without peptide identification content."""
    return f"mzspec:{dataset_name}:{source_name}:scan:{orig_scan}:charge{charge}"


def spectra_to_dat_by_charge(
    spectra: Iterable[dict],
    source_name: str,
    output_dir: str,
    file_idx: int = 0,
    charges: Sequence[int] = DEFAULT_CHARGES,
    dataset_name: Optional[str] = None,
    species: str = "",
    instrument: str = "",
) -> MzmlDatResult:
    """Write charge-partitioned .dat files from an iterable of mzML spectra."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(source_name).stem
    charge_set = tuple(sorted(set(int(c) for c in charges)))
    state: Dict[int, dict] = {}

    with ExitStack() as stack:
        for charge in charge_set:
            out_stem = f"{stem}_charge{charge}"
            dat_path = str(out_dir / f"{out_stem}.dat")
            scaninfo_path = str(out_dir / f"{out_stem}.scan_info.dat")
            titles_path = str(out_dir / f"{out_stem}.scan_titles.txt")
            spectra_path = (
                str(out_dir / f"{out_stem}.spectra.parquet")
                if dataset_name else None
            )
            state[charge] = {
                "written": 0,
                "skipped": 0,
                "dat_path": dat_path,
                "scaninfo_path": scaninfo_path,
                "titles_path": titles_path,
                "spectra_path": spectra_path,
                "dat": stack.enter_context(open(dat_path, "wb")),
                "scaninfo": stack.enter_context(open(scaninfo_path, "wb")),
                "titles": stack.enter_context(open(titles_path, "w")),
                "dat_buf": bytearray(),
                "scaninfo_buf": bytearray(),
                "title_lines": [],
                "sidecar_rows": [],
                "sidecar_writer": None,
            }

        skipped_missing_charge = 0

        for fallback_idx, spec in enumerate(spectra):
            if spec.get("ms level") != 2:
                continue

            charge = _extract_charge(spec)
            if charge is None:
                skipped_missing_charge += 1
                continue
            if charge not in state:
                continue

            charge_state = state[charge]
            prec_mz = _extract_precursor_mz(spec)
            mz_arr = spec.get("m/z array")
            int_arr = spec.get("intensity array")
            if prec_mz is None or mz_arr is None or int_arr is None:
                charge_state["skipped"] += 1
                continue

            mz_arr = np.asarray(mz_arr, dtype=np.float64)
            int_arr = np.asarray(int_arr, dtype=np.float64)
            if len(mz_arr) == 0 or len(mz_arr) != len(int_arr):
                charge_state["skipped"] += 1
                continue

            finite = np.isfinite(mz_arr) & np.isfinite(int_arr)
            if not np.all(finite):
                mz_arr = mz_arr[finite]
                int_arr = int_arr[finite]
            if len(mz_arr) == 0:
                charge_state["skipped"] += 1
                continue

            prec_mass = float(prec_mz) * charge - PROTON_MASS * (charge - 1)
            peak_bins = _bin_spectrum(mz_arr, int_arr, prec_mass)
            if len(peak_bins) < MIN_SCORING_PEAKS:
                charge_state["skipped"] += 1
                continue

            scannr = charge_state["written"]
            retention_time = _extract_retention_time(spec)
            orig_scan = _scan_number_from_id(spec.get("id", ""), fallback_idx)

            charge_state["dat_buf"].extend(
                _pack_spectrum(file_idx, scannr, charge, float(prec_mz),
                               retention_time, peak_bins)
            )
            charge_state["scaninfo_buf"].extend(
                _pack_scaninfo(file_idx, scannr, float(prec_mz), float(prec_mz))
            )
            charge_state["title_lines"].append(
                f"{file_idx}\t{scannr}\t"
                f"{_format_noid_scan_title(source_name, orig_scan, charge)}\n"
            )
            if dataset_name:
                charge_state["sidecar_rows"].append({
                    "scannr": scannr,
                    "usi": _format_noid_usi(
                        dataset_name, source_name, orig_scan, charge
                    ),
                    "project_accession": dataset_name,
                    "reference_file_name": source_name,
                    "scan": orig_scan,
                    "charge": charge,
                    "precursor_mz": float(prec_mz),
                    "retention_time": retention_time,
                    "mz_array": mz_arr.astype(np.float32).tolist(),
                    "intensity_array": int_arr.astype(np.float32).tolist(),
                    "species": species,
                    "instrument": instrument,
                })
            charge_state["written"] += 1

            if len(charge_state["dat_buf"]) >= 1_000_000:
                _flush_charge_state(charge_state)

        for charge_state in state.values():
            _flush_charge_state(charge_state)
            if charge_state["sidecar_writer"] is not None:
                charge_state["sidecar_writer"].close()

    results = tuple(
        ChargeDatResult(
            charge=charge,
            written=int(state[charge]["written"]),
            skipped=int(state[charge]["skipped"]),
            dat_path=str(state[charge]["dat_path"]),
            scaninfo_path=str(state[charge]["scaninfo_path"]),
            titles_path=str(state[charge]["titles_path"]),
            spectra_path=state[charge]["spectra_path"],
        )
        for charge in charge_set
    )

    return MzmlDatResult(
        mzml_path=source_name,
        file_idx=file_idx,
        charge_results=results,
        skipped_missing_charge=skipped_missing_charge,
    )


def _flush_charge_state(charge_state: dict) -> None:
    if charge_state["dat_buf"]:
        charge_state["dat"].write(charge_state["dat_buf"])
        charge_state["scaninfo"].write(charge_state["scaninfo_buf"])
        charge_state["titles"].writelines(charge_state["title_lines"])
        charge_state["dat_buf"].clear()
        charge_state["scaninfo_buf"].clear()
        charge_state["title_lines"].clear()
    if charge_state["sidecar_rows"]:
        table = pa.Table.from_pylist(
            charge_state["sidecar_rows"], schema=NOID_SPECTRUM_SCHEMA
        )
        if charge_state["sidecar_writer"] is None:
            charge_state["sidecar_writer"] = pq.ParquetWriter(
                charge_state["spectra_path"],
                schema=NOID_SPECTRUM_SCHEMA,
                compression="zstd",
            )
        charge_state["sidecar_writer"].write_table(table)
        charge_state["sidecar_rows"].clear()


def mzml_to_dat_by_charge(
    mzml_path: str,
    output_dir: str,
    file_idx: int = 0,
    charges: Sequence[int] = DEFAULT_CHARGES,
    dataset_name: Optional[str] = None,
    species: str = "",
    instrument: str = "",
) -> MzmlDatResult:
    """Convert one mzML file into charge-specific .dat outputs."""
    source_name = Path(mzml_path).name
    with mzml.MzML(mzml_path) as reader:
        result = spectra_to_dat_by_charge(
            reader,
            source_name=source_name,
            output_dir=output_dir,
            file_idx=file_idx,
            charges=charges,
            dataset_name=dataset_name,
            species=species,
            instrument=instrument,
        )
    logger.info("Converted %s", mzml_path)
    return MzmlDatResult(
        mzml_path=str(mzml_path),
        file_idx=file_idx,
        charge_results=result.charge_results,
        skipped_missing_charge=result.skipped_missing_charge,
    )


def merge_mzml_charge_shards(
    source_results: Sequence[MzmlDatResult],
    output_dir: str,
    charges: Sequence[int] = DEFAULT_CHARGES,
    file_idx_start: int = 0,
) -> List[ChargeDatResult]:
    """Merge per-mzML shards into one final .dat triplet per charge.

    mzML inputs are already split by run/sample, so the final ID-free clustering
    input is partitioned only by precursor charge. During merge, ``scannr`` is
    renumbered so every charge file has a single continuous scan index.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_source = sorted(source_results, key=lambda result: result.file_idx)
    charge_set = tuple(sorted(set(int(c) for c in charges)))
    merged_results: List[ChargeDatResult] = []

    for charge in charge_set:
        out_stem = f"charge{charge}"
        dat_path = out_dir / f"{out_stem}.dat"
        scaninfo_path = out_dir / f"{out_stem}.scan_info.dat"
        titles_path = out_dir / f"{out_stem}.scan_titles.txt"
        spectra_path = out_dir / f"{out_stem}.spectra.parquet"
        final_file_idx = file_idx_start
        written = 0
        skipped = 0
        has_sidecars = any(
            charge_result is not None
            and charge_result.spectra_path
            and Path(charge_result.spectra_path).exists()
            for source_result in by_source
            for charge_result in [_charge_result_for(source_result, charge)]
        )

        with open(dat_path, "wb") as f_dat, \
             open(scaninfo_path, "wb") as f_scaninfo, \
             open(titles_path, "w") as f_titles:
            spectra_writer = (
                pq.ParquetWriter(
                    spectra_path,
                    schema=NOID_SPECTRUM_SCHEMA,
                    compression="zstd",
                )
                if has_sidecars else None
            )
            for source_result in by_source:
                charge_result = _charge_result_for(source_result, charge)
                if charge_result is None:
                    continue
                skipped += charge_result.skipped
                written = _append_charge_shard(
                    charge_result=charge_result,
                    final_file_idx=final_file_idx,
                    next_scannr=written,
                    dat_out=f_dat,
                    scaninfo_out=f_scaninfo,
                    titles_out=f_titles,
                )
                if spectra_writer is not None and charge_result.spectra_path:
                    _append_sidecar_shard(
                        spectra_path=charge_result.spectra_path,
                        writer=spectra_writer,
                        scannr_offset=written - charge_result.written,
                    )
            if spectra_writer is not None:
                spectra_writer.close()

        merged_results.append(
            ChargeDatResult(
                charge=charge,
                written=written,
                skipped=skipped,
                dat_path=str(dat_path),
                scaninfo_path=str(scaninfo_path),
                titles_path=str(titles_path),
                spectra_path=str(spectra_path) if has_sidecars else None,
            )
        )

    return merged_results


def _charge_result_for(
    source_result: MzmlDatResult,
    charge: int,
) -> Optional[ChargeDatResult]:
    for charge_result in source_result.charge_results:
        if charge_result.charge == charge:
            return charge_result
    return None


def _append_charge_shard(
    charge_result: ChargeDatResult,
    final_file_idx: int,
    next_scannr: int,
    dat_out,
    scaninfo_out,
    titles_out,
) -> int:
    dat_path = Path(charge_result.dat_path)
    scaninfo_path = Path(charge_result.scaninfo_path)
    titles_path = Path(charge_result.titles_path)
    if not dat_path.exists() or dat_path.stat().st_size == 0:
        return next_scannr

    with open(dat_path, "rb") as f_dat, \
         open(scaninfo_path, "rb") as f_scaninfo, \
         open(titles_path) as f_titles:
        while True:
            dat_record = f_dat.read(SPECTRUM_STRUCT.size)
            if not dat_record:
                break
            if len(dat_record) != SPECTRUM_STRUCT.size:
                raise ValueError(f"Truncated .dat record in {dat_path}")

            scaninfo_record = f_scaninfo.read(SCANINFO_STRUCT.size)
            if len(scaninfo_record) != SCANINFO_STRUCT.size:
                raise ValueError(f"Truncated scan_info record in {scaninfo_path}")

            title_line = f_titles.readline()
            if not title_line:
                raise ValueError(f"Missing scan title for record in {titles_path}")

            fields = SPECTRUM_STRUCT.unpack(dat_record)
            scaninfo_fields = SCANINFO_STRUCT.unpack(scaninfo_record)
            title = _title_payload(title_line)

            dat_out.write(
                SPECTRUM_STRUCT.pack(
                    final_file_idx,
                    next_scannr,
                    int(fields[2]),
                    float(fields[3]),
                    float(fields[4]),
                    *fields[5:],
                )
            )
            scaninfo_out.write(
                _pack_scaninfo(
                    final_file_idx,
                    next_scannr,
                    float(scaninfo_fields[2]),
                    float(scaninfo_fields[3]),
                )
            )
            titles_out.write(f"{final_file_idx}\t{next_scannr}\t{title}\n")
            next_scannr += 1

    return next_scannr


def _append_sidecar_shard(
    spectra_path: str,
    writer: pq.ParquetWriter,
    scannr_offset: int,
) -> None:
    path = Path(spectra_path)
    if not path.exists() or path.stat().st_size == 0:
        return
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=50_000):
        df = batch.to_pandas()
        df["scannr"] = df["scannr"].astype(np.int32) + int(scannr_offset)
        table = pa.Table.from_pandas(
            df,
            schema=NOID_SPECTRUM_SCHEMA,
            preserve_index=False,
        )
        writer.write_table(table)


def _title_payload(line: str) -> str:
    parts = line.rstrip("\n").split("\t", 2)
    return parts[2] if len(parts) == 3 else line.strip()


def convert_mzml_dataset_to_dat(
    mzml_dir: str,
    output_dir: str,
    file_idx_start: int = 0,
    charges: Sequence[int] = DEFAULT_CHARGES,
    workers: int = 1,
) -> Tuple[List[ChargeDatResult], int, int]:
    """Convert all mzML files to final charge-only .dat partitions."""
    mzml_files = find_mzml_files(mzml_dir)
    if not mzml_files:
        raise FileNotFoundError(f"No mzML files found in {mzml_dir}")
    return convert_mzml_files_to_dat(
        mzml_files=mzml_files,
        output_dir=output_dir,
        file_idx_start=file_idx_start,
        charges=charges,
        workers=workers,
    )


def convert_mzml_files_to_dat(
    mzml_files: Sequence[str],
    output_dir: str,
    file_idx_start: int = 0,
    charges: Sequence[int] = DEFAULT_CHARGES,
    workers: int = 1,
    dataset_name: Optional[str] = None,
    species: str = "",
    instrument: str = "",
) -> Tuple[List[ChargeDatResult], int, int]:
    """Convert explicit mzML files into final charge-only .dat partitions."""
    if not mzml_files:
        raise FileNotFoundError("No mzML files were provided")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / f"_mzml_charge_shards_{uuid.uuid4().hex}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            path,
            str(shard_dir),
            file_idx_start + i,
            tuple(charges),
            dataset_name,
            species,
            instrument,
        )
        for i, path in enumerate(mzml_files)
    ]

    try:
        if workers <= 1:
            shard_results = [_convert_task(task) for task in tasks]
        else:
            shard_results = []
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(_convert_task, task): task for task in tasks
                }
                for future in as_completed(future_map):
                    shard_results.append(future.result())

        shard_results.sort(key=lambda r: r.file_idx)
        charge_results = merge_mzml_charge_shards(
            shard_results,
            output_dir=output_dir,
            charges=charges,
            file_idx_start=file_idx_start,
        )
        total_written = sum(result.written for result in charge_results)
        total_skipped = sum(result.skipped for result in charge_results) + sum(
            result.skipped_missing_charge for result in shard_results
        )
        return charge_results, total_written, total_skipped
    finally:
        shutil.rmtree(shard_dir, ignore_errors=True)


def convert_mzml_partitions_to_dat(
    mzml_dir: str,
    output_dir: str,
    dataset_name: str,
    sdrf_path: Optional[str] = None,
    default_species: Optional[str] = None,
    default_instrument: Optional[str] = None,
    skip_instrument: bool = False,
    charges: Sequence[int] = DEFAULT_CHARGES,
    workers: int = 1,
) -> str:
    """Convert mzML inputs into species/instrument/charge no-ID partitions.

    Returns the path to ``partitions.tsv``. File paths in the manifest are
    relative to ``output_dir`` so workflow engines can relocate the directory.
    """
    partition_df = build_mzml_partition_table(
        mzml_dir=mzml_dir,
        sdrf_path=sdrf_path,
        default_species=default_species,
        default_instrument=default_instrument,
        skip_instrument=skip_instrument,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for partition_id, group in partition_df.groupby("partition_id", sort=True):
        species = str(group["species"].iloc[0])
        instrument = str(group["instrument"].iloc[0])
        partition_dir = out_dir / partition_id
        charge_results, _, _ = convert_mzml_files_to_dat(
            mzml_files=group["mzml_path"].tolist(),
            output_dir=str(partition_dir),
            charges=charges,
            workers=workers,
            dataset_name=dataset_name,
            species=species,
            instrument=instrument,
        )
        for result in charge_results:
            if result.written == 0:
                continue
            manifest_rows.append({
                "partition_id": partition_id,
                "species": species,
                "instrument": instrument,
                "charge": f"charge{result.charge}",
                "dat_path": _relative_to(out_dir, result.dat_path),
                "scaninfo_path": _relative_to(out_dir, result.scaninfo_path),
                "titles_path": _relative_to(out_dir, result.titles_path),
                "spectra_path": (
                    _relative_to(out_dir, result.spectra_path)
                    if result.spectra_path else ""
                ),
            })

    if not manifest_rows:
        raise ValueError(
            "No mzML spectra passed conversion filters for the requested charges"
        )

    manifest_path = out_dir / "partitions.tsv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, sep="\t", index=False)
    return str(manifest_path)


def _relative_to(root: Path, path: str) -> str:
    return str(Path(path).relative_to(root)).replace("\\", "/")


def _convert_task(task: Tuple) -> MzmlDatResult:
    mzml_path, output_dir, file_idx, charges, dataset_name, species, instrument = task
    return mzml_to_dat_by_charge(
        mzml_path=mzml_path,
        output_dir=output_dir,
        file_idx=file_idx,
        charges=charges,
        dataset_name=dataset_name,
        species=species,
        instrument=instrument,
    )


def read_sdrf_mzml_metadata(sdrf_path: str) -> pd.DataFrame:
    """Read SDRF run metadata used to annotate mzML raw clustering inputs."""
    df = pd.read_csv(sdrf_path, sep="\t")
    data_col = "comment[data file]"
    species_col = _find_column(df, "characteristics[organism]")
    instrument_col = _find_column(df, "comment[instrument]")
    if data_col not in df.columns:
        raise ValueError(f"SDRF is missing required column: {data_col}")

    records = []
    for _, row in df.iterrows():
        data_file = str(row.get(data_col, "")).strip()
        if not data_file:
            continue
        records.append({
            "run_stem": Path(data_file).stem,
            "sdrf_data_file": data_file,
            "species": str(row.get(species_col, "")).strip() if species_col else "",
            "instrument": _extract_nt(row.get(instrument_col, "")) if instrument_col else "",
        })
    return pd.DataFrame(records).drop_duplicates(subset=["run_stem"])


def write_mzml_metadata_manifest(
    mzml_files: Sequence[str],
    sdrf_path: Optional[str],
    output_path: str,
) -> str:
    """Write a small TSV manifest mapping local mzML files to SDRF metadata."""
    rows = []
    sdrf_df = read_sdrf_mzml_metadata(sdrf_path) if sdrf_path else pd.DataFrame()
    by_stem = {
        str(row.run_stem).lower(): row
        for row in sdrf_df.itertuples(index=False)
    } if not sdrf_df.empty else {}

    for mzml_file in mzml_files:
        path = Path(mzml_file)
        row = by_stem.get(path.stem.lower())
        rows.append({
            "mzml_file": path.name,
            "sdrf_data_file": getattr(row, "sdrf_data_file", ""),
            "species": getattr(row, "species", ""),
            "instrument": getattr(row, "instrument", ""),
        })

    manifest = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, sep="\t", index=False)
    return output_path


def _find_column(df: pd.DataFrame, wanted_lower: str) -> Optional[str]:
    wanted = wanted_lower.lower()
    for col in df.columns:
        if str(col).lower() == wanted:
            return str(col)
    return None


def _extract_nt(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    match = re.search(r"NT=(.+?)(;|$)", text)
    return match.group(1).strip() if match else text
