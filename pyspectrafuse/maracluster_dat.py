"""Generate MaRaCluster-compatible .dat binary files directly from parquet.

Reads QPX .psm.parquet → writes compact .dat binary structs (100 bytes per
spectrum). MaRaCluster's -D flag reads pre-existing .dat files and starts
directly from the pvalue computation step.

Size: ~100 bytes/spectrum (1M PSMs = ~100 MB, 500M PSMs = ~50 GB)

How it fits into the pipeline:
  Normally MaRaCluster converts MGF/ms2 text files into .dat binaries as its
  first step, then clusters the .dat data. This module bypasses that conversion
  by producing .dat files directly from parquet — same binary format, but
  without the intermediate MGF text stage. MaRaCluster's ``-D dat_dir`` flag
  tells it to skip conversion and use existing .dat files in that directory.

  MaRaCluster still requires a list of .mgf paths via ``-b`` (hardcoded in its
  CLI parser), so the Nextflow process creates minimal dummy .mgf files (one
  fake spectrum each) whose basenames match the .dat files. MaRaCluster matches
  ``dummy.mgf`` → ``dummy.dat`` and reads the .dat data. The dummy MGF content
  is never used for actual spectra.

The .dat format is a flat array of Spectrum structs:
  struct Spectrum {  // 100 bytes total
    uint32 fileIdx;         // 0-based file index
    uint32 scannr;          // scan number
    uint32 charge;          // precursor charge
    float  precMz;          // precursor m/z
    float  retentionTime;   // retention time
    int16  fragBins[40];    // top-40 binned peak indices, 0-padded
  };

Reference: github.com/statisticalbiotechnology/maracluster
  - src/Spectrum.h, src/BinSpectra.cpp, src/BinaryInterface.cpp
"""
import logging
import struct
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ── MaRaCluster constants ──
# Senko bin width (mass of one nucleon at ~1 Da resolution)
BIN_WIDTH = 1.000508
BIN_SHIFT = 0.32
MAX_SCORING_PEAKS = 40
MIN_SCORING_PEAKS = 15
PROTON_MASS = 1.00727646677

# Struct format: 2×uint32 (ScanId) + uint32 (charge) + 2×float32 + 40×int16
# Little-endian, total = 4+4+4+4+4+80 = 100 bytes
SPECTRUM_STRUCT = struct.Struct('<IIIff40h')
SCANINFO_STRUCT = struct.Struct('<IIff')

assert SPECTRUM_STRUCT.size == 100
assert SCANINFO_STRUCT.size == 16


def _bin_spectrum(mz_array: np.ndarray, intensity_array: np.ndarray,
                  prec_mass: float, n_peaks: int = MAX_SCORING_PEAKS) -> List[int]:
    """Replicate MaRaCluster's binBinaryTruncated algorithm.

    1. Sort peaks by intensity descending
    2. For each peak (highest intensity first):
       - Skip if mz >= precursor mass (neutral mass)
       - Compute bin index: floor(mz / 1.000508 + 0.32)
       - If bin not seen before, add to result
       - Stop at n_peaks unique bins
    3. Sort bins ascending

    Args:
        mz_array: Fragment m/z values
        intensity_array: Fragment intensities
        prec_mass: Neutral precursor mass (NOT m/z — charge-deconvolved)
        n_peaks: Maximum number of bins to keep (default 40)

    Returns:
        Sorted list of bin indices (may be shorter than n_peaks)
    """
    if len(mz_array) == 0:
        return []

    # Sort by intensity descending
    order = np.argsort(-intensity_array)

    seen = set()
    peak_bins = []
    for idx in order:
        mz = float(mz_array[idx])
        if mz >= prec_mass:
            continue
        b = int(mz / BIN_WIDTH + BIN_SHIFT)
        if b not in seen and b > 0:
            seen.add(b)
            peak_bins.append(b)
            if len(peak_bins) >= n_peaks:
                break

    peak_bins.sort()
    return peak_bins


def _pack_spectrum(file_idx: int, scannr: int, charge: int,
                   prec_mz: float, retention_time: float,
                   peak_bins: List[int]) -> bytes:
    """Pack one Spectrum struct into 100 bytes."""
    bins = list(peak_bins[:MAX_SCORING_PEAKS])
    bins.extend([0] * (MAX_SCORING_PEAKS - len(bins)))
    return SPECTRUM_STRUCT.pack(
        file_idx, scannr, charge,
        float(prec_mz), float(retention_time),
        *bins
    )


def _pack_scaninfo(file_idx: int, scannr: int,
                   min_prec_mz: float, max_prec_mz: float) -> bytes:
    """Pack one ScanInfo struct into 16 bytes."""
    return SCANINFO_STRUCT.pack(file_idx, scannr,
                                float(min_prec_mz), float(max_prec_mz))


def parquet_to_dat(
    parquet_path,
    output_dir: str,
    file_idx: int = 0,
    charge_filter: Optional[int] = None,
    batch_size: int = 50_000,
    name: Optional[str] = None,
) -> Tuple[int, int, str]:
    """Convert one QPX .psm.parquet file (or a list of them) to MaRaCluster .dat binary format.

    Writes three files:
      - {name}.dat          — array of 100-byte Spectrum structs
      - {name}.scan_info.dat — array of 16-byte ScanInfo structs
      - {name}.scan_titles.txt — TSV of (fileIdx, scannr, title)

    A list of parquet files (e.g. the parts of one project) is written into one
    set of files with scannr continuing across parts, so scannr stays unique
    within the partition.

    Args:
        parquet_path: Path to a .psm.parquet file, or a list of paths
        output_dir: Directory to write .dat files
        file_idx: 0-based file index for this file in the batch
        charge_filter: If set, only include spectra with this charge
        batch_size: Arrow batch size for streaming reads
        name: Output stem (default: the parquet file stem)

    Returns:
        Tuple of (spectra_written, spectra_skipped, dat_path)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    paths = [parquet_path] if isinstance(parquet_path, (str, Path)) else list(parquet_path)
    if not paths:
        raise ValueError("no parquet files given")
    name = name or Path(paths[0]).stem
    if charge_filter is not None:
        name = f"{name}_charge{charge_filter}"
    dat_path = str(Path(output_dir) / f"{name}.dat")
    scaninfo_path = str(Path(output_dir) / f"{name}.scan_info.dat")
    titles_path = str(Path(output_dir) / f"{name}.scan_titles.txt")

    written = 0
    skipped = 0

    with open(dat_path, 'wb') as f_dat, \
         open(scaninfo_path, 'wb') as f_scaninfo, \
         open(titles_path, 'w') as f_titles:
        for one_path in paths:
            w, sk = _write_parquet_spectra(one_path, f_dat, f_scaninfo, f_titles, file_idx,
                                           charge_filter, batch_size, written)
            written += w
            skipped += sk

    dat_size = Path(dat_path).stat().st_size
    logger.info(f"Wrote {written} spectra to {dat_path} "
                f"({dat_size / 1e6:.1f} MB, {skipped} skipped)")

    return written, skipped, dat_path


def _write_parquet_spectra(parquet_path, f_dat, f_scaninfo, f_titles, file_idx: int,
                           charge_filter: Optional[int], batch_size: int,
                           scannr_start: int) -> Tuple[int, int]:
    """Append the spectra of one parquet file to open .dat/.scan_info/.titles handles."""
    pf = pq.ParquetFile(parquet_path)
    columns = ['mz_array', 'intensity_array', 'charge',
               'observed_mz', 'calculated_mz', 'rt', 'scan',
               'run_file_name', 'peptidoform']

    # Read available columns (QPX format may use different names)
    schema_names = set(pf.schema_arrow.names)
    read_cols = [c for c in columns if c in schema_names]

    written = 0
    skipped = 0

    for batch in pf.iter_batches(batch_size=batch_size, columns=read_cols):
        df = batch.to_pandas()

        if charge_filter is not None and 'charge' in df.columns:
            df = df[df['charge'] == charge_filter]
            if df.empty:
                continue

        # Vectorized column access
        mz_arrays = df['mz_array'].values if 'mz_array' in df.columns else None
        int_arrays = df['intensity_array'].values if 'intensity_array' in df.columns else None
        if mz_arrays is None or int_arrays is None:
            skipped += len(df)
            continue

        charges = df['charge'].values if 'charge' in df.columns else np.zeros(len(df), dtype=np.int32)
        obs_mz = df['observed_mz'].values if 'observed_mz' in df.columns else None
        calc_mz = df['calculated_mz'].values if 'calculated_mz' in df.columns else None
        runs = df['run_file_name'].values if 'run_file_name' in df.columns else [''] * len(df)
        peps = df['peptidoform'].values if 'peptidoform' in df.columns else [''] * len(df)
        scans = df['scan'].values if 'scan' in df.columns else [0] * len(df)

        # Pre-allocate buffers for batch writing
        dat_buf = bytearray()
        scaninfo_buf = bytearray()
        title_lines = []

        for i in range(len(df)):
            mz_arr = mz_arrays[i]
            int_arr = int_arrays[i]
            if mz_arr is None or int_arr is None:
                skipped += 1
                continue

            mz_arr = np.asarray(mz_arr, dtype=np.float64)
            int_arr = np.asarray(int_arr, dtype=np.float64)

            if len(mz_arr) == 0 or len(int_arr) == 0:
                skipped += 1
                continue

            charge = int(charges[i])
            if charge <= 0:
                skipped += 1
                continue

            # Precursor m/z: prefer observed_mz, fall back to calculated_mz
            prec_mz = obs_mz[i] if obs_mz is not None else None
            if prec_mz is None or np.isnan(prec_mz):
                prec_mz = calc_mz[i] if calc_mz is not None else 0.0
            prec_mz = float(prec_mz)

            # Neutral mass for peak filtering
            prec_mass = prec_mz * charge - PROTON_MASS * (charge - 1)

            # Bin the spectrum
            peak_bins = _bin_spectrum(mz_arr, int_arr, prec_mass)
            if len(peak_bins) < MIN_SCORING_PEAKS:
                skipped += 1
                continue

            scannr = scannr_start + written

            # Buffer writes instead of per-row I/O
            dat_buf.extend(_pack_spectrum(
                file_idx, scannr, charge, prec_mz, 0.0, peak_bins))
            scaninfo_buf.extend(_pack_scaninfo(
                file_idx, scannr, prec_mz, prec_mz))

            orig_scan = scans[i]
            if orig_scan is not None:
                if isinstance(orig_scan, (list, np.ndarray)):
                    orig_scan = int(orig_scan[0]) if len(orig_scan) > 0 else 0
                else:
                    orig_scan = int(orig_scan)
            else:
                orig_scan = 0
            run = runs[i] if runs[i] is not None else ''
            pep = peps[i] if peps[i] is not None else ''
            title_lines.append(f"{file_idx}\t{scannr}\tid=mzspec::{run}:scan:{orig_scan}:{pep}/{charge}\n")

            written += 1

        # Flush batch buffers
        if dat_buf:
            f_dat.write(dat_buf)
            f_scaninfo.write(scaninfo_buf)
            f_titles.writelines(title_lines)

    return written, skipped


def convert_dataset_to_dat(
    parquet_dir: str,
    output_dir: str,
    file_idx_start: int = 0,
    charge_filter: Optional[int] = None,
) -> Tuple[List[str], int]:
    """Convert all .psm.parquet files in a directory to .dat format.

    Args:
        parquet_dir: Directory containing .psm.parquet files
        output_dir: Directory to write .dat files
        file_idx_start: Starting file index
        charge_filter: If set, only include spectra with this charge

    Returns:
        Tuple of (list of .dat file paths, total spectra written)
    """
    from pyspectrafuse.commands.spectrum2msp import find_target_ext_files

    parquet_files = find_target_ext_files(parquet_dir, '.parquet')
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    dat_paths = []
    total_written = 0

    for i, pf in enumerate(parquet_files):
        written, skipped, dat_path = parquet_to_dat(
            pf, output_dir,
            file_idx=file_idx_start + i,
            charge_filter=charge_filter,
        )
        dat_paths.append(dat_path)
        total_written += written
        logger.info(f"  [{i+1}/{len(parquet_files)}] {Path(pf).name}: "
                     f"{written} written, {skipped} skipped")

    return dat_paths, total_written


def write_dat_file_list(dat_paths: List[str], output_path: str) -> str:
    """Write the file list that MaRaCluster's --datFNfile expects.

    Args:
        dat_paths: List of .dat file paths
        output_path: Where to write the file list

    Returns:
        output_path
    """
    with open(output_path, 'w') as f:
        for p in dat_paths:
            f.write(p + '\n')
    return output_path


def mz_windows(prec_mzs: np.ndarray, window_size: float = 300.0, window_overlap: float = 1.0,
               min_mz: float = 200.0, max_mz: float = 2000.0,
               max_spectra: Optional[int] = None) -> List[Tuple[float, float]]:
    """Precursor m/z windows: fixed-width, then any window holding more than ``max_spectra`` spectra is cut
    into equal-count sub-windows. Adjacent windows overlap by ``window_overlap`` Da so every pair of spectra
    within MaRaCluster's precursor tolerance shares at least one window."""
    if window_overlap >= window_size:
        raise ValueError(
            f"window_overlap ({window_overlap}) must be less than "
            f"window_size ({window_size}) to avoid infinite loop")
    fixed = []
    start = min_mz
    while start < max_mz:
        end = min(start + window_size, max_mz)
        fixed.append((start, end))
        if end >= max_mz:
            break
        start = end - window_overlap
    if not max_spectra:
        return fixed
    srt = np.sort(prec_mzs)
    out = []
    for wmin, wmax in fixed:
        lo, hi = np.searchsorted(srt, wmin), np.searchsorted(srt, wmax)
        n = hi - lo
        k = int(np.ceil(n / max_spectra)) if n > max_spectra else 1
        cuts = [wmin]
        for j in range(1, k):
            pos = lo + (j * n) // k
            cut = float(srt[pos - 1] + srt[pos]) / 2
            if cut - window_overlap > cuts[-1]:
                cuts.append(cut)
        cuts.append(wmax)
        for j in range(len(cuts) - 1):
            out.append((cuts[j] if j == 0 else cuts[j] - window_overlap, cuts[j + 1]))
    return out


def split_dat_by_mz_window(
    dat_path: str,
    scaninfo_path: str,
    titles_path: str,
    output_dir: str,
    window_size: float = 300.0,
    window_overlap: float = 1.0,
    min_mz: float = 200.0,
    max_mz: float = 2000.0,
    max_spectra_per_window: Optional[int] = None,
) -> List[Dict]:
    """Split a .dat file into precursor m/z windows for parallel MaRaCluster.

    MaRaCluster uses 20 ppm precursor tolerance internally — spectra far apart
    in m/z can NEVER cluster together. This function formalizes that by splitting
    spectra into overlapping m/z windows, enabling Nextflow to run MaRaCluster
    on each window in parallel. With ``max_spectra_per_window`` dense windows are
    cut further into equal-count sub-windows, which bounds MaRaCluster's memory.

    Args:
        dat_path: Path to the .dat file (100 bytes/spectrum).
        scaninfo_path: Path to the .scan_info.dat file (16 bytes/spectrum).
        titles_path: Path to the .scan_titles.txt file.
        output_dir: Directory to write per-window .dat files.
        window_size: Width of each m/z window in Da (default 300).
        window_overlap: Overlap between adjacent windows in Da (default 1.0).
        min_mz: Minimum precursor m/z (default 200).
        max_mz: Maximum precursor m/z (default 2000).
        max_spectra_per_window: Cut windows holding more spectra than this (default: no cut).

    Returns:
        List of dicts, each with keys:
            - 'window_idx': int
            - 'mz_min': float
            - 'mz_max': float
            - 'dat_path': str
            - 'scaninfo_path': str
            - 'titles_path': str
            - 'n_spectra': int
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dat = np.fromfile(dat_path, dtype=np.uint8)
    n_spectra = len(dat) // SPECTRUM_STRUCT.size
    if n_spectra == 0:
        return []
    dat = dat[:n_spectra * SPECTRUM_STRUCT.size].reshape(n_spectra, SPECTRUM_STRUCT.size)
    # precMz is the float32 after three uint32 (fileIdx, scannr, charge)
    prec_mzs = np.ascontiguousarray(dat[:, 12:16]).view('<f4').ravel()
    scaninfo = np.fromfile(scaninfo_path, dtype=np.uint8)
    n_si = len(scaninfo) // SCANINFO_STRUCT.size
    scaninfo = scaninfo[:n_si * SCANINFO_STRUCT.size].reshape(n_si, SCANINFO_STRUCT.size)
    with open(titles_path) as f:
        title_lines = f.readlines()

    windows = mz_windows(prec_mzs, window_size, window_overlap, min_mz, max_mz, max_spectra_per_window)
    stem = Path(dat_path).stem
    results = []
    idx = -1
    for wmin, wmax in windows:
        idx += 1
        indices = np.nonzero((prec_mzs >= wmin) & (prec_mzs < wmax))[0]
        if len(indices) == 0:
            continue
        w_dat_path = str(Path(output_dir) / f"{stem}_w{idx}.dat")
        w_scaninfo_path = str(Path(output_dir) / f"{stem}_w{idx}.scan_info.dat")
        w_titles_path = str(Path(output_dir) / f"{stem}_w{idx}.scan_titles.txt")
        # original structs, scaninfo and titles copied unchanged (original scannr kept)
        dat[indices].tofile(w_dat_path)
        scaninfo[indices[indices < n_si]].tofile(w_scaninfo_path)
        with open(w_titles_path, 'w') as f_titles:
            f_titles.writelines(title_lines[i] for i in indices if i < len(title_lines))
        results.append({
            'window_idx': idx,
            'mz_min': wmin,
            'mz_max': wmax,
            'dat_path': w_dat_path,
            'scaninfo_path': w_scaninfo_path,
            'titles_path': w_titles_path,
            'n_spectra': len(indices),
        })
        logger.info(f"Window {idx} [{wmin:.1f}-{wmax:.1f} Da]: "
                     f"{len(indices)} spectra → {w_dat_path}")

    logger.info(f"Split {n_spectra} spectra into {len(results)} m/z windows")
    return results
