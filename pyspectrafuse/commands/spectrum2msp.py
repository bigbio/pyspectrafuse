import glob
import logging
import re
import uuid
from pathlib import Path
from typing import List, Tuple

import click
import numpy as np
import pandas as pd

from pyspectrafuse.common.constant import ParquetSchemaAdapter
from pyspectrafuse.common.duckdb_ops import _detect_scan_expr
from pyspectrafuse.common.msp_utils import MspUtil
from pyspectrafuse.common.parquet_utils import ParquetPathHandler
from pyspectrafuse.consensus_strategy.average_spectrum_strategy import AverageSpectrumStrategy
from pyspectrafuse.consensus_strategy.best_spetrum_strategy import BestSpectrumStrategy
from pyspectrafuse.consensus_strategy.binning_strategy import BinningStrategy
from pyspectrafuse.consensus_strategy.most_similar_strategy import MostSimilarStrategy

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

logger = logging.getLogger(__name__)


def find_target_ext_files(directory: str, extensions: str) -> List[str]:
    """Find files with specified extension in directory tree.

    Args:
        directory: Root directory to search
        extensions: File extension to match (e.g., '.parquet')

    Returns:
        List of file paths matching the extension, excluding generated output
        files (psm_cluster_membership, cluster_metadata) and output directories
        (cluster_db/, msp/, mgf_output/).
    """
    # Directories that contain pipeline outputs, not inputs
    exclude_dirs = {'cluster_db', 'msp', 'mgf_output'}
    exclude_names = {'psm_cluster_membership', 'cluster_metadata'}
    # QPX metadata parquets are not PSM data; MSNet parquets are old format
    exclude_suffixes = {'.run.parquet', '.sample.parquet',
                        '.dataset.parquet', '.ontology.parquet',
                        '-MSNet.parquet'}

    res = []
    # Use glob.glob (follows symlinks) instead of Path.rglob (does not)
    for path_str in glob.glob(str(Path(directory) / f'**/*{extensions}'), recursive=True):
        file = Path(path_str)
        if not file.is_file():
            continue
        stem = file.stem
        if any(ex in stem for ex in exclude_names):
            continue
        if any(part in exclude_dirs for part in file.parts):
            continue
        if any(str(file).endswith(s) for s in exclude_suffixes):
            continue
        if extensions == '.parquet' and not file.name.endswith('.psm.parquet'):
            continue
        res.append(str(file))
    return sorted(res)


def create_consensus_strategy(method_type: str, sim: str = 'dot',
                             fragment_mz_tolerance: float = 0.02,
                             min_mz: float = 100., max_mz: float = 2000.,
                             bin_size: float = 0.02, peak_quorum: float = 0.25,
                             edge_case_threshold: float = 0.5,
                             diff_thresh: float = 0.01, dyn_range: int = 1000,
                             min_fraction: float = 0.5, pepmass: str = 'lower_median',
                             msms_avg: str = 'weighted'):
    """Create consensus strategy instance based on method type."""
    if method_type == "best":
        return BestSpectrumStrategy()
    elif method_type == "most":
        return MostSimilarStrategy(sim=sim, fragment_mz_tolerance=fragment_mz_tolerance)
    elif method_type == 'bin':
        return BinningStrategy(min_mz=min_mz, max_mz=max_mz, bin_size=bin_size,
                              peak_quorum=peak_quorum, edge_case_threshold=edge_case_threshold)
    elif method_type == 'average':
        return AverageSpectrumStrategy(DIFF_THRESH=diff_thresh, DYN_RANGE=dyn_range,
                                      MIN_FRACTION=min_fraction, pepmass=pepmass,
                                      msms_avg=msms_avg)
    else:
        raise ValueError(f"Unknown strategy type: {method_type}. "
                         f"Must be one of [best, most, bin, average]")


def write_spectra_to_msp(consensus_df, single_df, output_path: str) -> None:
    """Write consensus and single spectra to MSP file."""
    for spectrum_df in [consensus_df, single_df]:
        if spectrum_df.empty:
            logger.info("No spectra in this group, skipping")
            continue
        logger.info(f"Writing {len(spectrum_df)} spectra to MSP")
        spectrum_df.loc[:, 'msp_fmt'] = spectrum_df.apply(
            lambda row: MspUtil.get_msp_fmt(row), axis=1)
        MspUtil.write2msp(output_path, '\n\n'.join(spectrum_df['msp_fmt']) + '\n\n')


def parse_cluster_tsv(tsv_path: str) -> pd.DataFrame:
    """Parse MaRaCluster cluster TSV into (mgf_basename, scannr, cluster_id)."""
    df = pd.read_csv(tsv_path, sep='\t', header=None,
                     names=['mgf_path', 'scannr', 'cluster_id'],
                     skip_blank_lines=False)
    df = df.dropna()
    df['scannr'] = df['scannr'].astype(int)
    df['cluster_id'] = df['cluster_id'].astype(str)
    df['mgf_basename'] = df['mgf_path'].apply(
        lambda p: re.sub(r'_w\d+\.mgf$', '.mgf', Path(p).name))
    return df[['mgf_basename', 'scannr', 'cluster_id']]


def parse_scan_titles(titles_path: str, dataset_name: str) -> pd.DataFrame:
    """Parse scan_titles.txt -> DataFrame mapping scannr to original spectrum identity."""
    rows = []
    title_pattern = re.compile(r'id=mzspec::(.+?):scan:(\d+):(.+?)$')

    with open(titles_path) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t', 2)
            if len(parts) < 3:
                continue
            scannr = int(parts[1])
            title = parts[2]

            m = title_pattern.match(title)
            if m:
                run_file = m.group(1)
                orig_scan = int(m.group(2))
                pep_charge = m.group(3)
                last_slash = pep_charge.rfind('/')
                if last_slash > 0:
                    peptidoform = pep_charge[:last_slash]
                else:
                    peptidoform = pep_charge
                rows.append((scannr, run_file, orig_scan, peptidoform, dataset_name))

    return pd.DataFrame(rows, columns=['scannr', 'run_file_name', 'orig_scan',
                                        'peptidoform', 'dataset'])


def build_membership_from_scan_titles(
    cluster_tsv: str,
    scan_titles_files: List[str],
    dataset_names: List[str],
) -> pd.DataFrame:
    """Build cluster membership by joining cluster TSV with scan_titles.

    Returns DataFrame with columns: cluster_id, run_file_name, orig_scan,
    peptidoform, dataset, mgf_basename, scannr.
    """
    cluster_df = parse_cluster_tsv(cluster_tsv)
    logger.info(f"Cluster TSV: {len(cluster_df)} entries, "
                f"{cluster_df['cluster_id'].nunique()} clusters")

    titles_dfs = []
    for titles_file, ds_name in zip(scan_titles_files, dataset_names):
        titles_df = parse_scan_titles(titles_file, ds_name)
        stem = Path(titles_file).name.replace('.scan_titles.txt', '')
        titles_df['mgf_basename'] = f'{stem}.mgf'
        titles_dfs.append(titles_df)

    all_titles = pd.concat(titles_dfs, ignore_index=True)
    membership = cluster_df.merge(all_titles, on=['mgf_basename', 'scannr'], how='inner')

    if len(membership) < len(cluster_df):
        logger.warning(f"{len(cluster_df) - len(membership)} spectra lost in title join")
    logger.info(f"Membership: {len(membership)} PSMs in "
                f"{membership['cluster_id'].nunique()} clusters")
    return membership


def _build_parquet_union_sql(
    parquet_files: List[str],
    charge_int: int,
    conn,
) -> str:
    """Build a UNION ALL SQL query across all parquet files with column normalization."""
    parts = []
    for ppath in parquet_files:
        cols_df = conn.execute(f"SELECT name FROM parquet_schema('{ppath}')").fetchdf()
        available = set(cols_df['name'])

        run_col = 'run_file_name' if 'run_file_name' in available else 'reference_file_name'
        charge_col = 'charge' if 'charge' in available else 'precursor_charge'

        if 'mz_array' not in available or 'intensity_array' not in available:
            continue

        mz_parts = []
        if 'observed_mz' in available: mz_parts.append('observed_mz')
        if 'calculated_mz' in available: mz_parts.append('calculated_mz')
        if 'exp_mass_to_charge' in available: mz_parts.append('exp_mass_to_charge')
        mz_expr = f"COALESCE({', '.join(mz_parts)}, 0.0)" if mz_parts else '0.0'

        pep_col = 'posterior_error_probability' if 'posterior_error_probability' in available else 'NULL'
        scan_expr = _detect_scan_expr(conn, ppath)

        if 'global_qvalue' in available:
            qvalue_expr = 'global_qvalue'
        elif 'additional_scores' in available:
            qvalue_expr = """(
                SELECT s.score_value
                FROM UNNEST(additional_scores) AS t(s)
                WHERE s.score_name = 'global_qvalue'
                LIMIT 1
            )"""
        else:
            qvalue_expr = 'NULL'

        parts.append(f"""
            SELECT
                {run_col} AS run_file_name,
                {scan_expr} AS orig_scan,
                mz_array,
                intensity_array,
                CAST({charge_col} AS INTEGER) AS charge,
                {mz_expr} AS pepmass,
                {pep_col} AS posterior_error_probability,
                {qvalue_expr} AS global_qvalue
            FROM read_parquet('{ppath}')
            WHERE {charge_col} = {charge_int}
        """)

    return " UNION ALL ".join(parts) if parts else ""


def load_spectra_for_clusters(
    cluster_membership: pd.DataFrame,
    parquet_files: List[str],
    charge_int: int,
    union_sql: str,
    conn,
) -> pd.DataFrame:
    """Load spectrum arrays for a subset of clusters.

    Args:
        cluster_membership: Membership DataFrame (must have run_file_name, orig_scan,
            cluster_id, peptidoform, dataset columns).
        parquet_files: List of parquet paths (for column detection).
        charge_int: Charge filter value.
        union_sql: Pre-built UNION ALL SQL from _build_parquet_union_sql().
        conn: DuckDB connection.

    Returns DataFrame with columns needed by consensus strategies.
    """
    if cluster_membership.empty or not union_sql:
        return pd.DataFrame()

    conn.register('chunk_membership', cluster_membership)

    result = conn.execute(f"""
        SELECT
            m.cluster_id AS cluster_accession,
            s.mz_array,
            s.intensity_array,
            s.charge,
            m.peptidoform,
            s.pepmass,
            s.posterior_error_probability,
            s.global_qvalue,
            s.run_file_name AS reference_file_name,
            s.orig_scan AS scan,
            'mzspec:' || m.dataset || ':' || s.run_file_name || ':scan:'
                || CAST(s.orig_scan AS VARCHAR) || ':'
                || m.peptidoform || '/' || CAST(s.charge AS VARCHAR) AS usi
        FROM ({union_sql}) s
        JOIN chunk_membership m
            ON s.run_file_name = m.run_file_name
            AND s.orig_scan = m.orig_scan
    """).fetchdf()

    # Unregister to allow re-registration in next chunk
    try:
        conn.execute("DROP VIEW IF EXISTS chunk_membership")
    except Exception:
        pass

    return result


@click.command("msp", short_help="get msp format file")
@click.option('--parquet_dir', required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='Project directory containing PSM parquet files.')
@click.option('--method_type', required=True, type=click.Choice(['best', 'most', 'bin', 'average']),
              help='Consensus spectrum generation method')
@click.option('--cluster_tsv_file', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='MaRaCluster merged cluster TSV file')
@click.option('--scan_titles', required=True, multiple=True,
              help='scan_titles.txt file(s) from parquet-to-dat conversion (repeatable)')
@click.option('--species', required=True, help='species name')
@click.option('--instrument', required=True, help='instrument name')
@click.option('--charge', required=True, help='charge name (e.g. charge2)')
@click.option('--sim', default='dot', help='Similarity measure method (for most_similar)')
@click.option('--fragment_mz_tolerance', default=0.02, help='Fragment m/z tolerance (for most_similar)')
@click.option('--min_mz', default=100, help='Minimum m/z (for bin method)')
@click.option('--max_mz', default=2000, help='Maximum m/z (for bin method)')
@click.option('--bin_size', default=0.02, help='Bin size in m/z (for bin method)')
@click.option('--peak_quorum', default=0.25, help='Peak quorum threshold (for bin method)')
@click.option('--edge_case_threshold', default=0.5, help='Edge case threshold (for bin method)')
@click.option('--diff_thresh', default=0.01, help='Minimum distance between MS/MS peak clusters')
@click.option('--dyn_range', default=1000, help='Dynamic range to apply')
@click.option('--min_fraction', default=0.5, help='Minimum fraction of cluster spectra')
@click.option('--pepmass', type=click.Choice(['naive_average', 'neutral_average', 'lower_median']),
              default='lower_median')
@click.option('--msms_avg', type=click.Choice(['naive', 'weighted']), default='weighted')
def spectrum2msp(parquet_dir: str, method_type: str, cluster_tsv_file: str,
                 scan_titles: Tuple[str, ...],
                 species: str, instrument: str, charge: str,
                 sim: str = 'dot', fragment_mz_tolerance: float = 0.02,
                 min_mz: float = 100., max_mz: float = 2000., bin_size: float = 0.02,
                 peak_quorum: float = 0.25, edge_case_threshold: float = 0.5,
                 diff_thresh: float = 0.01, dyn_range: int = 1000,
                 min_fraction: float = 0.5, pepmass: str = 'lower_median',
                 msms_avg: str = 'weighted') -> None:
    """Generate MSP consensus spectral library from dat-bypass pipeline output.

    Uses scan_titles files to map MaRaCluster cluster assignments back to
    original spectra in the parquet files, then applies a consensus strategy
    to generate the spectral library in MSP format.
    """
    import gc
    import duckdb

    charge_int = int(charge.replace('charge', ''))
    chunk_size = 100_000  # clusters per chunk

    # Find parquet files
    path_parquet_lst = find_target_ext_files(parquet_dir, '.parquet')
    if not path_parquet_lst:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    logger.info(f"Found {len(path_parquet_lst)} parquet file(s)")

    # Derive dataset names from scan_titles filenames
    dataset_names = []
    for t in scan_titles:
        stem = Path(t).name.split('.')[0]
        dataset_names.append(stem)

    # Build cluster membership via scan_titles join (scalars only — small)
    membership = build_membership_from_scan_titles(
        cluster_tsv_file, list(scan_titles), dataset_names)

    if membership.empty:
        logger.warning("No cluster membership found, no MSP to generate")
        return

    # Setup output directory and file. An empty instrument (passed by the
    # Nextflow workflow when --skip_instrument is set) collapses the
    # species/instrument/charge layout to species/charge.
    if instrument and instrument.strip():
        output_dir = Path(parquet_dir) / 'msp' / species / instrument / charge
    else:
        output_dir = Path(parquet_dir) / 'msp' / species / charge
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = ParquetPathHandler(path_parquet_lst[0]).get_item_info()
    output = output_dir / f"{basename}_{uuid.uuid4()}.msp.gz"

    # Create consensus strategy
    consensus_strategy = create_consensus_strategy(
        method_type=method_type, sim=sim, fragment_mz_tolerance=fragment_mz_tolerance,
        min_mz=min_mz, max_mz=max_mz, bin_size=bin_size, peak_quorum=peak_quorum,
        edge_case_threshold=edge_case_threshold, diff_thresh=diff_thresh,
        dyn_range=dyn_range, min_fraction=min_fraction, pepmass=pepmass,
        msms_avg=msms_avg)

    # Pre-build the UNION SQL (reused across chunks)
    conn = duckdb.connect(':memory:')
    try:
        conn.execute("SET memory_limit='4GB'")
        conn.execute("SET threads=4")
        union_sql = _build_parquet_union_sql(path_parquet_lst, charge_int, conn)

        if not union_sql:
            logger.warning("No parquet files with spectrum arrays found")
            return

        # Process clusters in chunks to avoid OOM
        all_cluster_ids = membership['cluster_id'].unique()
        n_clusters = len(all_cluster_ids)
        logger.info(f"Processing {n_clusters} clusters in chunks of {chunk_size} "
                    f"({len(membership)} PSMs) using '{method_type}' strategy")

        total_consensus = 0
        total_single = 0

        for chunk_start in range(0, n_clusters, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_clusters)
            chunk_ids = all_cluster_ids[chunk_start:chunk_end]
            chunk_membership = membership[membership['cluster_id'].isin(chunk_ids)]

            logger.info(f"Chunk {chunk_start // chunk_size + 1}: "
                       f"clusters {chunk_start}-{chunk_end} "
                       f"({len(chunk_membership)} PSMs)")

            # Load spectrum arrays for this chunk only
            df = load_spectra_for_clusters(
                chunk_membership, path_parquet_lst, charge_int, union_sql, conn)

            if df.empty:
                logger.warning(f"No spectra loaded for chunk, skipping")
                continue

            # Apply consensus strategy
            consensus_df, single_df = consensus_strategy.consensus_spectrum_aggregation(df)

            # Write to MSP (appends)
            write_spectra_to_msp(consensus_df, single_df, str(output))
            total_consensus += len(consensus_df) if not consensus_df.empty else 0
            total_single += len(single_df) if not single_df.empty else 0

            del df, consensus_df, single_df, chunk_membership
            gc.collect()

        logger.info(f"MSP file generated: {output} "
                    f"({total_consensus} consensus + {total_single} single spectra)")
    finally:
        conn.close()
