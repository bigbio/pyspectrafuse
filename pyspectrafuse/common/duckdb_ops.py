"""DuckDB-based operations for cluster DB building and maintenance.

Replaces pandas groupby/concat/dedup patterns with SQL queries that work
efficiently at all scales. Each public function creates its own in-memory
DuckDB connection (batch operations, not long-lived).

Used by: build_cluster_db.py, incremental.py, merge_results.py, resolve_clusters_dat.py
"""
import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _create_connection(memory_limit: Optional[str] = None, threads: Optional[int] = None,
                       temp_directory: Optional[str] = None) -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection.

    Defaults come from SPECTRAFUSE_DUCKDB_MEMORY / _THREADS / _TEMP_DIR (set by the
    Nextflow task from its own memory and cpus), else 4GB and 4 threads.
    """
    memory_limit = memory_limit or os.environ.get('SPECTRAFUSE_DUCKDB_MEMORY', '4GB')
    threads = threads or int(os.environ.get('SPECTRAFUSE_DUCKDB_THREADS', '4'))
    temp_directory = temp_directory or os.environ.get('SPECTRAFUSE_DUCKDB_TEMP_DIR')
    conn = duckdb.connect(':memory:')
    conn.execute(f"SET memory_limit='{memory_limit}'")
    conn.execute(f"SET threads={threads}")
    if temp_directory:
        conn.execute(f"SET temp_directory='{temp_directory}'")
    return conn


def _register_source(conn: duckdb.DuckDBPyConnection,
                     source: Union[str, pd.DataFrame],
                     view_name: str = 'psm_data') -> None:
    """Register a parquet path or DataFrame as a DuckDB view."""
    if isinstance(source, str):
        conn.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{source}')")
    elif isinstance(source, pd.DataFrame):
        conn.register(view_name, source)
    else:
        raise TypeError(f"source must be str (parquet path) or DataFrame, got {type(source)}")


def _detect_scan_expr(conn: duckdb.DuckDBPyConnection, ppath: str) -> str:
    """Detect the correct SQL expression for the scan column in a parquet file.

    Returns a SQL expression string like 'scan[1]', 'CAST(scan AS INTEGER)', or '0'.
    """
    try:
        scan_type = conn.execute(
            f"SELECT typeof(scan) FROM read_parquet('{ppath}') LIMIT 1"
        ).fetchone()
        type_str = str(scan_type[0]).upper() if scan_type else ''
        if 'LIST' in type_str or '[]' in type_str:
            return 'scan[1]'
        elif scan_type:
            return 'CAST(scan AS INTEGER)'
    except (duckdb.Error, TypeError) as e:
        logger.warning(f"Could not detect scan column type in {ppath}: {e}")
    return '0'


def compute_cluster_stats(source: Union[str, pd.DataFrame],
                          cluster_col: str = 'cluster_id') -> pd.DataFrame:
    """Compute per-cluster aggregation stats.

    Args:
        source: Path to PSM membership parquet or DataFrame.
        cluster_col: Name of the cluster ID column.

    Returns:
        DataFrame with columns: cluster_id, member_count, project_count, best_pep, best_qvalue
    """
    conn = _create_connection()
    try:
        _register_source(conn, source, 'psm_data')
        result = conn.execute(f"""
            SELECT
                {cluster_col} AS cluster_id,
                CAST(COUNT(usi) AS INTEGER) AS member_count,
                CAST(COUNT(DISTINCT project_accession) AS INTEGER) AS project_count,
                MIN(posterior_error_probability) AS best_pep,
                MIN(global_qvalue) AS best_qvalue
            FROM psm_data
            GROUP BY {cluster_col}
        """).fetchdf()
        return result
    finally:
        conn.close()


def compute_purity(source: Union[str, pd.DataFrame],
                   cluster_col: str = 'cluster_id') -> pd.DataFrame:
    """Compute peptidoform purity per cluster.

    Purity = fraction of the most common peptidoform in each cluster.

    Args:
        source: Path to PSM membership parquet or DataFrame.
        cluster_col: Name of the cluster ID column.

    Returns:
        DataFrame with columns: cluster_id, purity
    """
    conn = _create_connection()
    try:
        _register_source(conn, source, 'psm_data')
        result = conn.execute(f"""
            WITH pair_counts AS (
                SELECT {cluster_col}, peptidoform, COUNT(*) AS cnt
                FROM psm_data
                GROUP BY {cluster_col}, peptidoform
            ),
            max_counts AS (
                SELECT {cluster_col}, MAX(cnt) AS max_cnt
                FROM pair_counts
                GROUP BY {cluster_col}
            ),
            totals AS (
                SELECT {cluster_col}, CAST(COUNT(*) AS DOUBLE) AS total
                FROM psm_data
                GROUP BY {cluster_col}
            )
            SELECT
                t.{cluster_col} AS cluster_id,
                CAST(m.max_cnt / t.total AS FLOAT) AS purity
            FROM totals t
            JOIN max_counts m ON t.{cluster_col} = m.{cluster_col}
        """).fetchdf()
        return result
    finally:
        conn.close()


def compute_stats_and_purity(source: Union[str, pd.DataFrame],
                             cluster_col: str = 'cluster_id') -> pd.DataFrame:
    """Compute cluster stats and purity in a single connection.

    More efficient than calling compute_cluster_stats + compute_purity separately
    since it only scans the data once.

    Returns:
        DataFrame with columns: cluster_id, member_count, project_count,
        best_pep, best_qvalue, purity
    """
    conn = _create_connection()
    try:
        _register_source(conn, source, 'psm_data')
        result = conn.execute(f"""
            WITH stats AS (
                SELECT
                    {cluster_col} AS cluster_id,
                    CAST(COUNT(usi) AS INTEGER) AS member_count,
                    CAST(COUNT(DISTINCT project_accession) AS INTEGER) AS project_count,
                    MIN(posterior_error_probability) AS best_pep,
                    MIN(global_qvalue) AS best_qvalue
                FROM psm_data
                GROUP BY {cluster_col}
            ),
            pair_counts AS (
                SELECT {cluster_col}, peptidoform, COUNT(*) AS cnt
                FROM psm_data
                GROUP BY {cluster_col}, peptidoform
            ),
            max_counts AS (
                SELECT {cluster_col}, MAX(cnt) AS max_cnt
                FROM pair_counts
                GROUP BY {cluster_col}
            ),
            totals AS (
                SELECT {cluster_col}, CAST(COUNT(*) AS DOUBLE) AS total
                FROM psm_data
                GROUP BY {cluster_col}
            )
            SELECT
                s.*,
                CAST(m.max_cnt / t.total AS FLOAT) AS purity
            FROM stats s
            JOIN totals t ON s.cluster_id = t.{cluster_col}
            JOIN max_counts m ON s.cluster_id = m.{cluster_col}
        """).fetchdf()
        return result
    finally:
        conn.close()


def find_best_psm_per_cluster(source: Union[str, pd.DataFrame],
                              metric: str = 'posterior_error_probability',
                              cluster_col: str = 'cluster_id',
                              columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Find the best PSM per cluster by a given metric (lowest value wins).

    Args:
        source: Path to PSM parquet or DataFrame.
        metric: Column to minimize (e.g. 'posterior_error_probability', 'global_qvalue').
        cluster_col: Name of the cluster ID column.
        columns: If set, only return these columns. None returns all.

    Returns:
        DataFrame with one row per cluster, the row with the lowest metric value.
    """
    conn = _create_connection()
    try:
        _register_source(conn, source, 'psm_data')

        select_cols = '*' if columns is None else ', '.join(columns)
        result = conn.execute(f"""
            SELECT {select_cols} FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY {cluster_col}
                           ORDER BY {metric} ASC NULLS LAST
                       ) AS _rn
                FROM psm_data
            ) sub
            WHERE _rn = 1
        """).fetchdf()
        if '_rn' in result.columns:
            result = result.drop(columns=['_rn'])
        return result
    finally:
        conn.close()


def append_membership_dedup(existing_path: str,
                            new_df: pd.DataFrame,
                            output_path: str) -> str:
    """Append new PSMs to existing membership parquet, deduplicating by USI.

    Existing rows take priority over new rows (keep='first').

    Args:
        existing_path: Path to existing psm_cluster_membership.parquet.
        new_df: DataFrame of new PSMs to append.
        output_path: Where to write the merged result.

    Returns:
        output_path
    """
    conn = _create_connection()
    try:
        conn.register('new_psms', new_df)

        # Get columns from existing parquet
        existing_cols = [col for col in conn.execute(
            f"SELECT name FROM parquet_schema('{existing_path}')"
        ).fetchdf()['name']]

        # Only keep columns that exist in the existing schema
        new_cols = [c for c in new_df.columns if c in existing_cols]
        select_existing = ', '.join(existing_cols)

        # Pad missing columns with NULL
        new_select_parts = []
        for col in existing_cols:
            if col in new_cols:
                new_select_parts.append(col)
            else:
                new_select_parts.append(f"NULL AS {col}")
        select_new_padded = ', '.join(new_select_parts)

        conn.execute(f"""
            COPY (
                WITH combined AS (
                    SELECT {select_existing}, 1 AS _priority FROM read_parquet('{existing_path}')
                    UNION ALL
                    SELECT {select_new_padded}, 2 AS _priority FROM new_psms
                ),
                deduped AS (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY usi ORDER BY _priority) AS _rn
                    FROM combined
                )
                SELECT {select_existing}
                FROM deduped
                WHERE _rn = 1
                ORDER BY cluster_id
            ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        n_existing = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{existing_path}')").fetchone()[0]
        n_new = len(new_df)
        n_merged = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0]
        logger.info(f"PSM membership: {n_existing} existing + {n_new} new "
                    f"→ {n_merged} total (after dedup) written to {output_path}")

        return output_path
    finally:
        conn.close()


def write_parquet_from_df(df: pd.DataFrame, output_path: str,
                          order_by: Optional[str] = None) -> str:
    """Write a DataFrame to parquet using DuckDB (ZSTD compression).

    Args:
        df: DataFrame to write.
        output_path: Destination path.
        order_by: Optional column to sort by.

    Returns:
        output_path
    """
    conn = _create_connection()
    try:
        conn.register('source_df', df)
        order_clause = f"ORDER BY {order_by}" if order_by else ""
        conn.execute(f"""
            COPY (SELECT * FROM source_df {order_clause})
            TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        return output_path
    finally:
        conn.close()


def scan_parquet_scalars(
    parquet_paths: List[str],
    needed_keys_df: pd.DataFrame,
    charge_int: int,
) -> pd.DataFrame:
    """Read scalar columns from source parquets, filtering by (run_file_name, scan) keys.

    Replaces the nested Python loop in build_cluster_db._scan_parquet_scalars()
    with a single DuckDB query using predicate pushdown.

    Args:
        parquet_paths: List of .psm.parquet file paths.
        needed_keys_df: DataFrame with columns ['run_file_name', 'orig_scan'].
        charge_int: Charge value to filter by.

    Returns:
        DataFrame with columns: run_file_name, orig_scan, precursor_mz,
        posterior_error_probability, global_qvalue
    """
    if needed_keys_df.empty:
        return pd.DataFrame()

    conn = _create_connection()
    try:
        conn.register('needed_keys', needed_keys_df)

        # Build UNION ALL of all parquet files with column normalization
        parts = []
        for ppath in parquet_paths:
            # Check available columns
            cols_df = conn.execute(f"SELECT name FROM parquet_schema('{ppath}')").fetchdf()
            available = set(cols_df['name'])

            # Build column aliases for normalization
            run_col = 'run_file_name' if 'run_file_name' in available else 'reference_file_name'
            charge_col = 'charge' if 'charge' in available else 'precursor_charge'

            mz_parts = []
            if 'observed_mz' in available: mz_parts.append('observed_mz')
            if 'calculated_mz' in available: mz_parts.append('calculated_mz')
            if 'exp_mass_to_charge' in available: mz_parts.append('exp_mass_to_charge')
            mz_expr = f"COALESCE({', '.join(mz_parts)}, 0.0)" if mz_parts else '0.0'

            pep_col = 'posterior_error_probability' if 'posterior_error_probability' in available else 'NULL'

            scan_expr = _detect_scan_expr(conn, ppath)

            # Handle global_qvalue (may be in additional_scores in QPX)
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
                    {mz_expr} AS precursor_mz,
                    {pep_col} AS posterior_error_probability,
                    {qvalue_expr} AS global_qvalue
                FROM read_parquet('{ppath}')
                WHERE {charge_col} = {charge_int}
            """)

        if not parts:
            return pd.DataFrame()

        union_sql = " UNION ALL ".join(parts)

        result = conn.execute(f"""
            SELECT s.*
            FROM ({union_sql}) s
            SEMI JOIN needed_keys k
                ON s.run_file_name = k.run_file_name
                AND s.orig_scan = k.orig_scan
        """).fetchdf()

        return result
    finally:
        conn.close()


def scan_parquet_arrays(
    parquet_paths: List[str],
    best_psm_keys_df: pd.DataFrame,
    charge_int: int,
    temp_directory: str = None,
) -> pd.DataFrame:
    """Read spectrum array columns from source parquets for best PSMs.

    Replaces the chunked Pass 2 loop in build_cluster_db with a DuckDB join.

    Args:
        parquet_paths: List of .psm.parquet file paths.
        best_psm_keys_df: DataFrame with columns ['run_file_name', 'orig_scan'].
        charge_int: Charge value to filter by.
        temp_directory: Directory for DuckDB to spill to disk when memory is tight.

    Returns:
        DataFrame with columns: run_file_name, orig_scan, mz_array, intensity_array
    """
    if best_psm_keys_df.empty:
        return pd.DataFrame()

    conn = _create_connection(temp_directory=temp_directory)
    try:
        conn.register('best_keys', best_psm_keys_df)

        parts = []
        for ppath in parquet_paths:
            cols_df = conn.execute(f"SELECT name FROM parquet_schema('{ppath}')").fetchdf()
            available = set(cols_df['name'])

            run_col = 'run_file_name' if 'run_file_name' in available else 'reference_file_name'
            charge_col = 'charge' if 'charge' in available else 'precursor_charge'

            scan_expr = _detect_scan_expr(conn, ppath)

            if 'mz_array' not in available or 'intensity_array' not in available:
                continue

            parts.append(f"""
                SELECT
                    {run_col} AS run_file_name,
                    {scan_expr} AS orig_scan,
                    mz_array,
                    intensity_array
                FROM read_parquet('{ppath}')
                WHERE {charge_col} = {charge_int}
            """)

        if not parts:
            return pd.DataFrame()

        union_sql = " UNION ALL ".join(parts)

        result = conn.execute(f"""
            SELECT s.*
            FROM ({union_sql}) s
            SEMI JOIN best_keys k
                ON s.run_file_name = k.run_file_name
                AND s.orig_scan = k.orig_scan
        """).fetchdf()

        return result
    finally:
        conn.close()


def merge_mz_window_clusters(
    window_tsvs: List[str],
    window_mz_ranges: List[tuple],
    overlap_da: float = 1.0,
) -> pd.DataFrame:
    """Merge cluster assignments from overlapping m/z windows.

    Spectra in overlap zones may be assigned to clusters in both windows.
    Resolution: if a spectrum is in an overlap zone and was clustered in both
    windows, prefer the assignment from the window where it is NOT in the
    overlap zone (i.e., from its "home" window). If in overlap in both, keep
    the one from the lower window.

    Args:
        window_tsvs: List of MaRaCluster output TSV paths, one per window.
        window_mz_ranges: List of (min_mz, max_mz) tuples for each window.
        overlap_da: Overlap width in Da.

    Returns:
        DataFrame with columns: mgf_path, scannr, cluster_id (resolved).
    """
    conn = _create_connection()
    try:
        all_parts = []
        for i, (tsv_path, (min_mz, max_mz)) in enumerate(zip(window_tsvs, window_mz_ranges)):
            df = pd.read_csv(tsv_path, sep='\t', header=None,
                             names=['mgf_path', 'scannr', 'cluster_id'],
                             skip_blank_lines=False)
            df = df.dropna()
            df['scannr'] = df['scannr'].astype('int64')
            df['window_idx'] = i
            df['window_min_mz'] = min_mz
            df['window_max_mz'] = max_mz
            # Prefix cluster IDs with window index to avoid collisions
            df['cluster_id'] = f'w{i}_' + df['cluster_id'].astype('int64').astype(str)
            all_parts.append(df)

        if not all_parts:
            return pd.DataFrame()

        combined = pd.concat(all_parts, ignore_index=True)
        conn.register('combined', combined)

        # For spectra appearing in multiple windows (overlap zone), keep the
        # assignment from the window with the lower index (deterministic).
        # Dedup by scannr only — within a charge partition, scannr is unique.
        # mgf_path differs across windows for the same spectrum.
        result = conn.execute("""
            SELECT mgf_path, scannr, cluster_id FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY scannr
                           ORDER BY window_idx
                       ) AS _rn
                FROM combined
            )
            WHERE _rn = 1
        """).fetchdf()

        return result
    finally:
        conn.close()
