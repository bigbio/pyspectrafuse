"""Resolve incremental clustering results using scan_titles (dat-mode).

Reads scan_titles.txt files to identify representative spectra (titles starting
with ``rep:``) and map new MaRaCluster cluster IDs back to existing ones.

Terminology:
- *old cluster*: a cluster from the previous round (has a ``cluster_id`` in
  ``cluster_metadata.parquet``).
- *representative*: the consensus spectrum written to .dat with scan_title
  ``rep:{cluster_id}``.
- *new cluster*: a cluster assigned by MaRaCluster in the incremental round.
- *resolved cluster*: the final cluster ID — reuses the old ID when a
  representative is present, or mints a new UUID when no representative maps.
"""
import logging
import re
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_REP_PREFIX = "rep:"
_TITLE_PATTERN = re.compile(r'id=mzspec::(.+?):scan:(\d+):(.+?)$')


def parse_cluster_tsv(tsv_path: str) -> pd.DataFrame:
    """Read a MaRaCluster TSV into a DataFrame.

    Returns:
        DataFrame with columns [mgf_path, scannr, new_cluster_id, mgf_basename].
    """
    df = pd.read_csv(tsv_path, sep='\t', header=None,
                     names=['mgf_path', 'scannr', 'new_cluster_id'])
    df.dropna(inplace=True)
    df['scannr'] = df['scannr'].astype(int)
    # Strip window suffix from MGF basename for matching with scan_titles
    df['mgf_basename'] = df['mgf_path'].apply(
        lambda p: re.sub(r'_w\d+\.mgf$', '.mgf', Path(p).name))
    return df


def parse_all_scan_titles(scan_titles_files: List[str]) -> pd.DataFrame:
    """Parse scan_titles.txt files into a unified lookup table.

    For representative titles (``rep:{cluster_id}``), ``old_cluster_id`` is set.
    For new-data titles (``id=mzspec::{run}:scan:{n}:{peptidoform}/{charge}``),
    the run_file_name / orig_scan / peptidoform / parsed_charge columns are set.

    Returns:
        DataFrame with columns [mgf_basename, scannr, title, is_rep,
        old_cluster_id, run_file_name, orig_scan, peptidoform, parsed_charge].
        Missing values in string columns are preserved as ``None`` (not NaN).
    """
    # Build column-wise to preserve None in object columns across pandas
    # versions. Tuple-of-rows construction coerces None → NaN on newer pandas
    # when neighbouring columns are numeric.
    cols = {k: [] for k in (
        'mgf_basename', 'scannr', 'title', 'is_rep', 'old_cluster_id',
        'run_file_name', 'orig_scan', 'peptidoform', 'parsed_charge')}

    for titles_path in scan_titles_files:
        stem = Path(titles_path).name.replace('.scan_titles.txt', '')
        mgf_basename = f'{stem}.mgf'

        with open(titles_path) as f:
            for line in f:
                parts = line.rstrip('\n').split('\t', 2)
                if len(parts) < 3:
                    continue
                scannr = int(parts[1])
                title = parts[2]

                cols['mgf_basename'].append(mgf_basename)
                cols['scannr'].append(scannr)
                cols['title'].append(title)

                if title.startswith(_REP_PREFIX):
                    cols['is_rep'].append(True)
                    cols['old_cluster_id'].append(title[len(_REP_PREFIX):])
                    cols['run_file_name'].append(None)
                    cols['orig_scan'].append(None)
                    cols['peptidoform'].append(None)
                    cols['parsed_charge'].append(None)
                    continue

                cols['is_rep'].append(False)
                cols['old_cluster_id'].append(None)

                m = _TITLE_PATTERN.match(title)
                if m:
                    run_file = m.group(1)
                    orig_scan = int(m.group(2))
                    pep_charge = m.group(3)
                    if pep_charge.startswith("charge") and pep_charge[6:].isdigit():
                        peptidoform = None
                        parsed_charge = int(pep_charge[6:])
                    else:
                        last_slash = pep_charge.rfind('/')
                        if last_slash > 0:
                            peptidoform = pep_charge[:last_slash]
                            parsed_charge = int(pep_charge[last_slash + 1:])
                        else:
                            peptidoform = pep_charge
                            parsed_charge = 0
                    cols['run_file_name'].append(run_file)
                    cols['orig_scan'].append(orig_scan)
                    cols['peptidoform'].append(peptidoform)
                    cols['parsed_charge'].append(parsed_charge)
                else:
                    cols['run_file_name'].append(None)
                    cols['orig_scan'].append(None)
                    cols['peptidoform'].append(None)
                    cols['parsed_charge'].append(None)

    df = pd.DataFrame(cols)
    # Force object dtype and restore None for string columns — some pandas
    # versions still coerce Python None to NaN inside object-dtype columns
    # when the column is constructed from a list containing None alongside
    # strings. Explicit restoration keeps `value is None` semantics.
    for col in ('old_cluster_id', 'run_file_name', 'peptidoform'):
        df[col] = df[col].astype(object)
        na_mask = df[col].isna()
        if na_mask.any():
            df.loc[na_mask, col] = None
    return df


def resolve_incremental_clusters(
    cluster_tsv_path: str,
    scan_titles_files: List[str],
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Map new MaRaCluster cluster IDs to resolved (old or fresh) IDs.

    Algorithm:
    1. Parse the MaRaCluster TSV to get (mgf_basename, scannr) → new_cluster_id.
    2. Join with scan_titles to identify which spectra are representatives.
    3. For representatives, map new_cluster_id → old_cluster_id.
    4. For new clusters with no representative, mint a fresh UUID.

    If multiple old clusters merge into one new cluster, the first old cluster
    ID wins (deterministic).

    Returns:
        Tuple of:
        - id_map: {new_cluster_id → resolved_cluster_id}
        - new_spectra_df: DataFrame of non-representative spectra with columns
          [mgf_basename, scannr, new_cluster_id, resolved_cluster_id, title,
           run_file_name, orig_scan, peptidoform, parsed_charge].
    """
    cluster_df = parse_cluster_tsv(cluster_tsv_path)
    titles_df = parse_all_scan_titles(scan_titles_files)

    logger.info(f"Cluster TSV: {len(cluster_df)} rows")
    logger.info(f"Scan titles: {len(titles_df)} entries "
                f"({titles_df['is_rep'].sum()} representatives)")

    merged = cluster_df.merge(titles_df, on=['mgf_basename', 'scannr'], how='left')

    rep_rows = merged[merged['is_rep'] == True].copy()
    new_rows = merged[merged['is_rep'] != True].copy()

    logger.info(f"Matched: {len(rep_rows)} representatives, {len(new_rows)} new spectra")

    new_to_old: Dict[str, list] = {}
    for _, row in rep_rows.iterrows():
        new_cid = str(row['new_cluster_id'])
        old_cid = row['old_cluster_id']
        if old_cid is not None:
            new_to_old.setdefault(new_cid, []).append(old_cid)

    id_map: Dict[str, str] = {}
    merge_count = 0

    for new_cid in cluster_df['new_cluster_id'].unique():
        new_cid_str = str(new_cid)
        old_ids = new_to_old.get(new_cid_str, [])
        if len(old_ids) == 0:
            id_map[new_cid_str] = str(uuid.uuid4())
        elif len(old_ids) == 1:
            id_map[new_cid_str] = old_ids[0]
        else:
            id_map[new_cid_str] = old_ids[0]
            merge_count += 1
            logger.info(f"Cluster merge: new {new_cid_str} <- old {old_ids}")

    if merge_count > 0:
        logger.info(f"Total cluster merges: {merge_count}")

    new_rows['resolved_cluster_id'] = (
        new_rows['new_cluster_id'].astype(str).map(id_map))

    return id_map, new_rows


def load_new_spectra_with_arrays(
    new_rows: pd.DataFrame,
    parquet_paths: List[str],
    charge_int: int,
    project_accession: str,
) -> pd.DataFrame:
    """Enrich resolved new-spectra rows with PSM data from source parquets.

    Joins ``new_rows`` (from :func:`resolve_incremental_clusters`) with the
    project's PSM parquet files by ``(run_file_name, orig_scan)`` via DuckDB,
    pulling in precursor_mz, PEP, q-value, mz_array, and intensity_array.
    Builds USIs in the same ``mzspec:{dataset}:{run}:scan:{n}:{peptidoform}/{z}``
    format as :mod:`build_cluster_db`.

    Args:
        new_rows: Non-representative spectra with resolved_cluster_id.
        parquet_paths: PSM parquet files for the new project.
        charge_int: Charge state to filter by.
        project_accession: Dataset name used in the USI prefix.

    Returns:
        DataFrame with columns [cluster_accession, usi, reference_file_name,
        scan, peptidoform, charge, pepmass, posterior_error_probability,
        global_qvalue, mz_array, intensity_array].
    """
    if new_rows.empty or not parquet_paths:
        return pd.DataFrame()

    from pyspectrafuse.common.duckdb_ops import _create_connection, _detect_scan_expr

    keys = new_rows[['run_file_name', 'orig_scan']].dropna().drop_duplicates().copy()
    keys['orig_scan'] = keys['orig_scan'].astype(int)

    conn = _create_connection()
    try:
        conn.register('needed_keys', keys)

        parts = []
        for ppath in parquet_paths:
            cols_df = conn.execute(
                f"SELECT name FROM parquet_schema('{ppath}')").fetchdf()
            available = set(cols_df['name'])

            if 'mz_array' not in available or 'intensity_array' not in available:
                continue

            run_col = 'run_file_name' if 'run_file_name' in available else 'reference_file_name'
            charge_col = 'charge' if 'charge' in available else 'precursor_charge'

            mz_candidates = [c for c in ('observed_mz', 'calculated_mz',
                                         'exp_mass_to_charge') if c in available]
            mz_expr = (f"COALESCE({', '.join(mz_candidates)}, 0.0)"
                       if mz_candidates else '0.0')

            pep_col = ('posterior_error_probability'
                       if 'posterior_error_probability' in available else 'NULL')

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
                    {mz_expr} AS precursor_mz,
                    {pep_col} AS posterior_error_probability,
                    {qvalue_expr} AS global_qvalue,
                    mz_array,
                    intensity_array
                FROM read_parquet('{ppath}')
                WHERE {charge_col} = {charge_int}
            """)

        if not parts:
            return pd.DataFrame()

        union_sql = " UNION ALL ".join(parts)
        psm_data = conn.execute(f"""
            SELECT s.*
            FROM ({union_sql}) s
            SEMI JOIN needed_keys k
                ON s.run_file_name = k.run_file_name
                AND s.orig_scan = k.orig_scan
        """).fetchdf()
    finally:
        conn.close()

    if psm_data.empty:
        return pd.DataFrame()

    psm_data['orig_scan'] = psm_data['orig_scan'].astype(int)
    merged = new_rows.merge(psm_data, on=['run_file_name', 'orig_scan'], how='inner')

    if merged.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        'cluster_accession': merged['resolved_cluster_id'].astype(str),
        'reference_file_name': merged['run_file_name'].astype(str),
        'scan': merged['orig_scan'].astype(int),
        'peptidoform': merged['peptidoform'].astype(str),
        'charge': merged['parsed_charge'].fillna(charge_int).astype(int),
        'pepmass': pd.to_numeric(merged['precursor_mz'], errors='coerce').astype(float),
        'posterior_error_probability': pd.to_numeric(
            merged['posterior_error_probability'], errors='coerce').astype(float),
        'global_qvalue': pd.to_numeric(
            merged['global_qvalue'], errors='coerce').astype(float),
        'mz_array': merged['mz_array'],
        'intensity_array': merged['intensity_array'],
    })
    out['usi'] = ('mzspec:' + project_accession + ':'
                  + out['reference_file_name'] + ':scan:'
                  + out['scan'].astype(str) + ':'
                  + out['peptidoform'] + '/'
                  + out['charge'].astype(str))
    return out
