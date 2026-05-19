"""Centralized Parquet schemas for cluster DB output files.

These schemas define the canonical column names and types for the two output
parquet files produced by the clustering pipeline:
- cluster_metadata.parquet: one row per cluster (consensus spectrum + stats)
- psm_cluster_membership.parquet: one row per PSM (cluster assignment + scores)
"""
import pyarrow as pa

CLUSTER_META_SCHEMA = pa.schema([
    ('cluster_id', pa.string()),
    ('species', pa.string()),
    ('instrument', pa.string()),
    ('charge', pa.int8()),
    ('peptidoform', pa.string()),
    ('peptide_sequence', pa.string()),
    ('consensus_mz_array', pa.list_(pa.float32())),
    ('consensus_intensity_array', pa.list_(pa.float32())),
    ('consensus_method', pa.string()),
    ('precursor_mz', pa.float64()),
    ('member_count', pa.int32()),
    ('project_count', pa.int16()),
    ('best_pep', pa.float64()),
    ('best_qvalue', pa.float64()),
    ('purity', pa.float32()),
    # Provenance: True when this cluster's ID was inherited from a previous
    # round (i.e. a representative spectrum from an existing cluster DB was
    # passed in and merged into this run).
    ('is_reused_cluster', pa.bool_()),
    # Provenance: distinct project accessions that contribute PSMs to this
    # cluster, accumulated across all rounds.
    ('source_datasets', pa.list_(pa.string())),
])

PSM_MEMBERSHIP_SCHEMA = pa.schema([
    ('cluster_id', pa.string()),
    ('usi', pa.string()),
    ('project_accession', pa.string()),
    ('reference_file_name', pa.string()),
    ('scan', pa.int32()),
    ('peptidoform', pa.string()),
    ('charge', pa.int8()),
    ('precursor_mz', pa.float64()),
    ('posterior_error_probability', pa.float64()),
    ('global_qvalue', pa.float64()),
    ('species', pa.string()),
    ('instrument', pa.string()),
])

NOID_SPECTRUM_SCHEMA = pa.schema([
    ('scannr', pa.int32()),
    ('usi', pa.string()),
    ('project_accession', pa.string()),
    ('reference_file_name', pa.string()),
    ('scan', pa.int32()),
    ('charge', pa.int8()),
    ('precursor_mz', pa.float64()),
    ('retention_time', pa.float64()),
    ('mz_array', pa.list_(pa.float32())),
    ('intensity_array', pa.list_(pa.float32())),
    ('species', pa.string()),
    ('instrument', pa.string()),
])

NOID_CLUSTER_META_SCHEMA = pa.schema([
    ('cluster_id', pa.string()),
    ('species', pa.string()),
    ('instrument', pa.string()),
    ('charge', pa.int8()),
    ('consensus_mz_array', pa.list_(pa.float32())),
    ('consensus_intensity_array', pa.list_(pa.float32())),
    ('consensus_method', pa.string()),
    ('precursor_mz', pa.float64()),
    ('member_count', pa.int32()),
    ('project_count', pa.int16()),
    ('cluster_quality_ratio', pa.float64()),
    ('mean_similarity', pa.float64()),
    ('is_reused_cluster', pa.bool_()),
    ('source_datasets', pa.list_(pa.string())),
])

SPECTRUM_MEMBERSHIP_SCHEMA = pa.schema([
    ('cluster_id', pa.string()),
    ('usi', pa.string()),
    ('project_accession', pa.string()),
    ('reference_file_name', pa.string()),
    ('scan', pa.int32()),
    ('charge', pa.int8()),
    ('precursor_mz', pa.float64()),
    ('species', pa.string()),
    ('instrument', pa.string()),
])
