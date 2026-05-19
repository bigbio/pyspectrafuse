"""Convert MSNet parquet files to QPX format.

Remaps column names, types, and structures from the legacy MSNet schema
to the QPX PSM parquet schema used by spectrafuse. Also generates
.run.parquet, .sample.parquet, .dataset.parquet, and .ontology.parquet
from the accompanying SDRF file.

Usage:
    pyspectrafuse convert-to-qpx --input <msnet.parquet> --sdrf <sdrf.tsv> --output_dir <dir>
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

logger = logging.getLogger(__name__)

# QPX PSM schema definition
QPX_PSM_SCHEMA = pa.schema([
    ("sequence", pa.string()),
    ("peptidoform", pa.string()),
    ("modifications", pa.list_(pa.struct([
        ("name", pa.string()),
        ("accession", pa.string()),
        ("positions", pa.list_(pa.struct([
            ("position", pa.int32()),
            ("amino_acid", pa.string()),
            ("scores", pa.list_(pa.struct([
                ("score_name", pa.string()),
                ("score_value", pa.float64()),
                ("higher_better", pa.bool_()),
            ]))),
        ]))),
    ]))),
    ("charge", pa.int16()),
    ("posterior_error_probability", pa.float64()),
    ("is_decoy", pa.bool_()),
    ("calculated_mz", pa.float32()),
    ("observed_mz", pa.float32()),
    ("mass_error_ppm", pa.float32()),
    ("additional_scores", pa.list_(pa.struct([
        ("score_name", pa.string()),
        ("score_value", pa.float64()),
        ("higher_better", pa.bool_()),
    ]))),
    ("predicted_rt", pa.float32()),
    ("run_file_name", pa.string()),
    ("cv_params", pa.list_(pa.struct([
        ("cv_name", pa.string()),
        ("cv_value", pa.string()),
    ]))),
    ("scan", pa.list_(pa.int32())),
    ("rt", pa.float32()),
    ("ion_mobility", pa.float32()),
    ("missed_cleavages", pa.int16()),
    ("protein_accessions", pa.list_(pa.string())),
    ("cross_links", pa.list_(pa.struct([
        ("xl_type", pa.string()),
        ("partner_sequence", pa.string()),
        ("partner_peptidoform", pa.string()),
        ("donor_position", pa.int32()),
        ("acceptor_position", pa.int32()),
        ("linker_name", pa.string()),
        ("linker_accession", pa.string()),
        ("linker_mass", pa.float64()),
    ]))),
    ("mz_array", pa.list_(pa.float32())),
    ("intensity_array", pa.list_(pa.float32())),
    ("charge_array", pa.list_(pa.int32())),
    ("ion_type_array", pa.list_(pa.string())),
    ("ion_mobility_array", pa.list_(pa.float32())),
])


def _convert_modifications(msnet_mods_col):
    """Convert MSNet modifications to QPX format.

    MSNet: list<struct<name, positions: list<struct<position: string>>>>
    QPX:   list<struct<name, accession, positions: list<struct<position: int32, amino_acid, scores>>>>
    """
    n = len(msnet_mods_col)
    qpx_mods = []

    for i in range(n):
        row_mods = msnet_mods_col[i].as_py()
        if row_mods is None:
            qpx_mods.append(None)
            continue

        converted = []
        for mod in row_mods:
            positions = []
            if mod.get("positions"):
                for pos in mod["positions"]:
                    pos_val = pos.get("position", "0") if pos else "0"
                    try:
                        pos_int = int(pos_val)
                    except (ValueError, TypeError):
                        pos_int = 0
                    positions.append({
                        "position": pos_int,
                        "amino_acid": None,
                        "scores": None,
                    })
            converted.append({
                "name": mod.get("name"),
                "accession": None,
                "positions": positions if positions else None,
            })
        qpx_mods.append(converted)

    return pa.array(qpx_mods, type=QPX_PSM_SCHEMA.field("modifications").type)


def _convert_cv_params(msnet_cv_col):
    """Convert MSNet cv_params struct to QPX list<struct<cv_name, cv_value>>.

    MSNet: struct<Instrument, Fragmentation Method, Collision Energy, Mass Analyzer, isotope_error>
    QPX:   list<struct<cv_name, cv_value>>
    """
    n = len(msnet_cv_col)
    qpx_cv = []

    for i in range(n):
        row = msnet_cv_col[i].as_py()
        if row is None:
            qpx_cv.append(None)
            continue

        entries = []
        for key, val in row.items():
            if val is not None:
                entries.append({"cv_name": str(key), "cv_value": str(val)})
        qpx_cv.append(entries if entries else None)

    return pa.array(qpx_cv, type=QPX_PSM_SCHEMA.field("cv_params").type)


def _build_additional_scores(global_qvalue_col):
    """Wrap global_qvalue into QPX additional_scores array."""
    n = len(global_qvalue_col)
    scores = []

    for i in range(n):
        val = global_qvalue_col[i].as_py()
        if val is None:
            scores.append(None)
        else:
            scores.append([{
                "score_name": "global_qvalue",
                "score_value": float(val),
                "higher_better": False,
            }])

    return pa.array(scores, type=QPX_PSM_SCHEMA.field("additional_scores").type)


def _wrap_scan_as_list(scan_col):
    """Convert scalar int32 scan to list<int32>."""
    n = len(scan_col)
    result = []
    for i in range(n):
        val = scan_col[i].as_py()
        result.append([val] if val is not None else None)
    return pa.array(result, type=pa.list_(pa.int32()))


def _convert_batch(batch: pa.RecordBatch, col_names: set) -> pa.RecordBatch:
    """Convert a single batch from MSNet to QPX schema."""
    table = pa.Table.from_batches([batch])
    n = table.num_rows

    columns = {}

    columns["sequence"] = table.column("sequence").cast(pa.string())
    columns["peptidoform"] = table.column("peptidoform").cast(pa.string())
    columns["posterior_error_probability"] = table.column("posterior_error_probability").cast(pa.float64())
    columns["is_decoy"] = table.column("is_decoy").cast(pa.bool_())
    columns["mz_array"] = table.column("mz_array")
    columns["intensity_array"] = table.column("intensity_array")

    columns["ion_type_array"] = table.column("ion_type_array") if "ion_type_array" in col_names else pa.nulls(n, type=pa.list_(pa.string()))
    columns["charge_array"] = table.column("charge_array") if "charge_array" in col_names else pa.nulls(n, type=pa.list_(pa.int32()))

    columns["charge"] = table.column("precursor_charge").cast(pa.int16())
    columns["observed_mz"] = table.column("exp_mass_to_charge").cast(pa.float32())
    columns["run_file_name"] = table.column("reference_file_name").cast(pa.string())
    columns["calculated_mz"] = table.column("cal_mass_to_charge").cast(pa.float32()) if "cal_mass_to_charge" in col_names else pa.nulls(n, type=pa.float32())

    columns["scan"] = _wrap_scan_as_list(table.column("scan").cast(pa.int32()) if pa.types.is_string(table.column("scan").type) else table.column("scan"))

    # columns["scan"] = _wrap_scan_as_list(table.column("scan"))
    columns["rt"] = table.column("retention_time").cast(pa.float32()) if "retention_time" in col_names else pa.nulls(n, type=pa.float32())
    columns["additional_scores"] = _build_additional_scores(table.column("global_qvalue")) if "global_qvalue" in col_names else pa.nulls(n, type=QPX_PSM_SCHEMA.field("additional_scores").type)
    columns["modifications"] = _convert_modifications(table.column("modifications")) if "modifications" in col_names else pa.nulls(n, type=QPX_PSM_SCHEMA.field("modifications").type)
    columns["cv_params"] = _convert_cv_params(table.column("cv_params")) if "cv_params" in col_names else pa.nulls(n, type=QPX_PSM_SCHEMA.field("cv_params").type)

    columns["mass_error_ppm"] = pa.nulls(n, type=pa.float32())
    columns["predicted_rt"] = pa.nulls(n, type=pa.float32())
    columns["ion_mobility"] = pa.nulls(n, type=pa.float32())
    columns["missed_cleavages"] = pa.nulls(n, type=pa.int16())
    columns["protein_accessions"] = pa.nulls(n, type=pa.list_(pa.string()))
    columns["cross_links"] = pa.nulls(n, type=QPX_PSM_SCHEMA.field("cross_links").type)
    columns["ion_mobility_array"] = pa.nulls(n, type=pa.list_(pa.float32()))

    arrays = [columns[field.name] for field in QPX_PSM_SCHEMA]
    return pa.table(arrays, schema=QPX_PSM_SCHEMA).to_batches()[0]


def convert_msnet_to_qpx(input_path: str, output_path: str,
                          batch_size: int = 100_000) -> None:
    """Convert an MSNet parquet file to QPX PSM format.

    Processes in batches to avoid OOM on large files (e.g. 6G+).
    """
    parquet_file = pq.ParquetFile(input_path)
    col_names = set(f.name for f in parquet_file.schema_arrow)
    total_rows = parquet_file.metadata.num_rows

    logger.info(f"Converting {input_path}: {total_rows:,} rows, batch_size={batch_size}")

    writer = None
    rows_written = 0

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            qpx_batch = _convert_batch(batch, col_names)

            if writer is None:
                writer = pq.ParquetWriter(output_path, QPX_PSM_SCHEMA, compression="zstd")

            writer.write_batch(qpx_batch)
            rows_written += len(batch)

            if rows_written % (batch_size * 5) == 0:
                logger.info(f"  Converted {rows_written:,}/{total_rows:,} rows")
    finally:
        if writer is not None:
            writer.close()

    logger.info(f"Done: {rows_written:,} rows written to {output_path}")


# -----------------------------------------------------------------------
# QPX run.parquet and sample.parquet generation from SDRF
# -----------------------------------------------------------------------

# Sentinel values that indicate an SDRF field has no meaningful content
_SENTINELS = {"not applicable", "not available", "none", "na", "n/a", ""}

# PyArrow schema for sample_channel struct (used in run.parquet)
_SAMPLE_CHANNEL_TYPE = pa.struct([
    ("sample_accession", pa.string()),
    ("label", pa.string()),
    ("biological_replicate", pa.int32()),
    ("technical_replicate", pa.int32()),
])

# PyArrow schema for modification_param struct (used in run.parquet)
_MOD_PARAM_TYPE = pa.struct([
    ("accession", pa.string()),
    ("name", pa.string()),
    ("fixed", pa.bool_()),
    ("position", pa.string()),
    ("target_amino_acid", pa.string()),
])

QPX_RUN_SCHEMA = pa.schema([
    ("run_accession", pa.string()),
    ("run_file_name", pa.string()),
    ("file_name", pa.string()),
    ("samples", pa.list_(_SAMPLE_CHANNEL_TYPE)),
    ("fraction", pa.string()),
    ("instrument", pa.string()),
    ("enzymes", pa.list_(pa.string())),
    ("dissociation_method", pa.string()),
    ("modification_parameters", pa.list_(_MOD_PARAM_TYPE)),
])

# sample.parquet: required fields only (optional fields added dynamically)
QPX_SAMPLE_BASE_FIELDS = [
    ("sample_accession", pa.string()),
    ("organism", pa.string()),
    ("organism_part", pa.string()),
]


def _extract_nt(value) -> str:
    """Extract the NT= name from a complex SDRF value like 'NT=Trypsin;AC=MS:1001251'."""
    if pd.isna(value):
        return ""
    s = str(value)
    if "NT=" in s:
        m = re.search(r"NT=(.+?)(;|$)", s)
        return m.group(1).strip() if m else s.strip()
    return s.strip()


def _is_sentinel(value) -> bool:
    if value is None or pd.isna(value):
        return True
    return str(value).strip().lower() in _SENTINELS


def _parse_modification_params(sdrf_df: pd.DataFrame):
    """Extract modification parameters from SDRF comment columns."""
    mod_cols = [c for c in sdrf_df.columns if c.startswith("comment[modification parameter")]
    if not mod_cols:
        return None

    results = []
    for col in mod_cols:
        non_null = sdrf_df[col].dropna()
        if non_null.empty:
            continue
        raw = str(non_null.iloc[0])
        parts = raw.split(";")
        kv = {}
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                kv[k.strip()] = v.strip()
        results.append({
            "accession": kv.get("AC", ""),
            "name": kv.get("NT", ""),
            "fixed": kv.get("MT", "").lower() in ("fixed", "fix"),
            "position": kv.get("PP"),
            "target_amino_acid": kv.get("TA"),
        })
    return results if results else None


def generate_sample_parquet(sdrf_df: pd.DataFrame, output_path: str) -> None:
    """Generate .sample.parquet from an SDRF DataFrame."""
    col_source = "source name"
    col_organism = "characteristics[organism]"
    col_organism_part = "characteristics[organism part]"

    # Optional sample fields and their SDRF column names
    optional_fields = {
        "disease": "characteristics[disease]",
        "cell_line": "characteristics[cell line]",
        "cell_type": "characteristics[cell type]",
        "sex": "characteristics[sex]",
        "age": "characteristics[age]",
        "developmental_stage": "characteristics[developmental stage]",
        "ancestry": "characteristics[ancestry category]",
        "individual": "characteristics[individual]",
    }

    # Determine which optional fields have real data
    active_optional = {}
    for field, col in optional_fields.items():
        if col in sdrf_df.columns:
            vals = sdrf_df[col].dropna().unique()
            if not all(_is_sentinel(v) for v in vals):
                active_optional[field] = col

    records = []
    for sample_name, group in sdrf_df.groupby(col_source):
        rec = {"sample_accession": str(sample_name)}

        # Required fields
        if col_organism in group.columns:
            vals = group[col_organism].dropna().unique()
            rec["organism"] = "; ".join(str(v) for v in vals) if len(vals) > 0 else None
        else:
            rec["organism"] = None

        if col_organism_part in group.columns:
            vals = group[col_organism_part].dropna().unique()
            rec["organism_part"] = "; ".join(str(v) for v in vals) if len(vals) > 0 else None
        else:
            rec["organism_part"] = None

        # Active optional fields
        for field, col in active_optional.items():
            vals = group[col].dropna().unique()
            rec[field] = "; ".join(str(v) for v in vals) if len(vals) > 0 else None

        records.append(rec)

    # Build schema: base fields + active optional fields
    schema_fields = list(QPX_SAMPLE_BASE_FIELDS)
    for field in active_optional:
        schema_fields.append((field, pa.string()))
    schema = pa.schema(schema_fields)

    table = pa.table({f: [r.get(f) for r in records] for f, _ in schema_fields}, schema=schema)
    pq.write_table(table, output_path, compression="zstd")
    logger.info(f"Wrote {len(records)} samples to {output_path}")


def generate_run_parquet(sdrf_df: pd.DataFrame, output_path: str) -> None:
    """Generate .run.parquet from an SDRF DataFrame."""
    col_data_file = "comment[data file]"
    col_source = "source name"
    col_label = "comment[label]"
    col_fraction = "comment[fraction identifier]"
    col_tech_rep = "comment[technical replicate]"
    col_bio_rep = "characteristics[biological replicate]"
    col_instrument = "comment[instrument]"
    col_enzyme = "comment[cleavage agent details]"
    col_dissoc = "comment[dissociation method]"

    df = sdrf_df.copy()
    df["_run_file"] = df[col_data_file].astype(str).str.split(".").str[0]
    df["_file_name"] = df[col_data_file].astype(str).str.strip()

    mod_params = _parse_modification_params(df)

    records = []
    for run_file, group in df.groupby("_run_file"):
        run_file = str(run_file)

        # Build samples list
        samples = []
        for _, row in group.iterrows():
            label_raw = row.get(col_label, "")
            label = _extract_nt(label_raw) if pd.notna(label_raw) else ""
            sample_acc = str(row.get(col_source, ""))

            bio_rep = None
            if col_bio_rep in row.index and pd.notna(row.get(col_bio_rep)):
                try:
                    bio_rep = int(row[col_bio_rep])
                except (ValueError, TypeError):
                    pass

            tech_rep = None
            if col_tech_rep in row.index and pd.notna(row.get(col_tech_rep)):
                try:
                    tech_rep = int(row[col_tech_rep])
                except (ValueError, TypeError):
                    pass

            samples.append({
                "sample_accession": sample_acc,
                "label": label,
                "biological_replicate": bio_rep,
                "technical_replicate": tech_rep,
            })

        # Instrument
        instrument = None
        if col_instrument in group.columns:
            inst_vals = group[col_instrument].dropna().unique()
            if len(inst_vals) > 0:
                instrument = _extract_nt(inst_vals[0])

        # Fraction
        fraction = None
        if col_fraction in group.columns:
            frac_vals = group[col_fraction].dropna().unique()
            if len(frac_vals) > 0:
                fraction = str(frac_vals[0])

        # Enzymes
        enzymes = None
        if col_enzyme in group.columns:
            enz_vals = group[col_enzyme].dropna().unique()
            if len(enz_vals) > 0:
                enzymes = [_extract_nt(v) for v in enz_vals if pd.notna(v)]

        # Dissociation method
        dissoc = None
        if col_dissoc in group.columns:
            dissoc_vals = group[col_dissoc].dropna().unique()
            if len(dissoc_vals) > 0:
                dissoc = _extract_nt(dissoc_vals[0])

        # Original file name
        file_name_vals = group["_file_name"].dropna().unique()
        file_name = str(file_name_vals[0]) if len(file_name_vals) > 0 else None

        records.append({
            "run_accession": run_file,
            "run_file_name": run_file,
            "file_name": file_name,
            "samples": samples,
            "fraction": fraction,
            "instrument": instrument,
            "enzymes": enzymes,
            "dissociation_method": dissoc,
            "modification_parameters": mod_params,
        })

    # Build pyarrow table
    table = pa.table({
        "run_accession": [r["run_accession"] for r in records],
        "run_file_name": [r["run_file_name"] for r in records],
        "file_name": [r["file_name"] for r in records],
        "samples": pa.array([r["samples"] for r in records], type=pa.list_(_SAMPLE_CHANNEL_TYPE)),
        "fraction": [r["fraction"] for r in records],
        "instrument": [r["instrument"] for r in records],
        "enzymes": pa.array([r["enzymes"] for r in records], type=pa.list_(pa.string())),
        "dissociation_method": [r["dissociation_method"] for r in records],
        "modification_parameters": pa.array(
            [r["modification_parameters"] for r in records],
            type=pa.list_(_MOD_PARAM_TYPE),
        ),
    }, schema=QPX_RUN_SCHEMA)

    pq.write_table(table, output_path, compression="zstd")
    logger.info(f"Wrote {len(records)} runs to {output_path}")


# -----------------------------------------------------------------------
# QPX dataset.parquet generation
# -----------------------------------------------------------------------

QPX_DATASET_SCHEMA = pa.schema([
    ("project_accession", pa.string()),
    ("project_title", pa.string()),
    ("project_description", pa.string()),
    ("pubmed_id", pa.string()),
    ("software_name", pa.string()),
    ("software_version", pa.string()),
    ("creation_date", pa.string()),
    ("file_checksums", pa.map_(pa.string(), pa.string())),
    ("file_row_counts", pa.map_(pa.string(), pa.int64())),
    ("file_sizes_bytes", pa.map_(pa.string(), pa.int64())),
    ("total_structures", pa.int32()),
    ("packaged_at", pa.string()),
])


def generate_dataset_parquet(project_accession: str, output_dir: str,
                              output_path: str,
                              psm_path: str = None) -> None:
    """Generate .dataset.parquet with project-level metadata.

    Args:
        project_accession: Project ID (e.g., PXD014877).
        output_dir: Directory containing the QPX parquet files.
        output_path: Path to write the dataset parquet.
        psm_path: Optional path to PSM parquet for row count.
    """
    # Collect file inventory from the output directory
    file_checksums = []
    file_row_counts = []
    file_sizes = []
    total_structures = 0

    out_dir = Path(output_dir)
    for pq_file in out_dir.glob("*.parquet"):
        rel_name = pq_file.name
        size = pq_file.stat().st_size
        file_sizes.append((rel_name, size))

        try:
            pf = pq.ParquetFile(str(pq_file))
            row_count = pf.metadata.num_rows
            file_row_counts.append((rel_name, row_count))
            total_structures += 1
        except Exception:
            file_row_counts.append((rel_name, 0))

    table = pa.table({
        "project_accession": [project_accession],
        "project_title": [None],
        "project_description": [None],
        "pubmed_id": [None],
        "software_name": ["pyspectrafuse"],
        "software_version": ["0.0.3"],
        "creation_date": [datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")],
        "file_checksums": pa.array([file_checksums if file_checksums else None],
                                    type=pa.map_(pa.string(), pa.string())),
        "file_row_counts": pa.array([file_row_counts if file_row_counts else None],
                                     type=pa.map_(pa.string(), pa.int64())),
        "file_sizes_bytes": pa.array([file_sizes if file_sizes else None],
                                      type=pa.map_(pa.string(), pa.int64())),
        "total_structures": [total_structures],
        "packaged_at": [datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")],
    }, schema=QPX_DATASET_SCHEMA)

    pq.write_table(table, output_path, compression="zstd")
    logger.info(f"Wrote dataset metadata to {output_path}")


# -----------------------------------------------------------------------
# QPX ontology.parquet generation
# -----------------------------------------------------------------------

QPX_ONTOLOGY_SCHEMA = pa.schema([
    ("field_name", pa.string()),
    ("ontology_name", pa.string()),
    ("ontology_accession", pa.string()),
    ("ontology_source", pa.string()),
    ("ontology_version", pa.string()),
    ("view", pa.string()),
    ("description", pa.string()),
    ("source_column_name", pa.string()),
    ("source_tool", pa.string()),
])

# Known PSI-MS CV mappings for common instruments, enzymes, dissociation methods
_KNOWN_CV_TERMS = {
    # Instruments
    "Q Exactive HF-X": ("MS:1002877", "Q Exactive HF-X", "MS", "mass spectrometer"),
    "Q Exactive HF": ("MS:1002523", "Q Exactive HF", "MS", "mass spectrometer"),
    "Q Exactive": ("MS:1001911", "Q Exactive", "MS", "mass spectrometer"),
    "Q Exactive Plus": ("MS:1002634", "Q Exactive Plus", "MS", "mass spectrometer"),
    "Orbitrap Fusion": ("MS:1002416", "Orbitrap Fusion", "MS", "mass spectrometer"),
    "Orbitrap Fusion Lumos": ("MS:1002732", "Orbitrap Fusion Lumos", "MS", "mass spectrometer"),
    "Orbitrap Eclipse": ("MS:1003294", "Orbitrap Eclipse", "MS", "mass spectrometer"),
    "Orbitrap Exploris 480": ("MS:1003028", "Orbitrap Exploris 480", "MS", "mass spectrometer"),
    "LTQ Orbitrap Elite": ("MS:1001742", "LTQ Orbitrap Elite", "MS", "mass spectrometer"),
    "LTQ Orbitrap Velos": ("MS:1001742", "LTQ Orbitrap Velos", "MS", "mass spectrometer"),
    "timsTOF Pro": ("MS:1003005", "timsTOF Pro", "MS", "mass spectrometer"),
    # Enzymes
    "Trypsin": ("MS:1001251", "Trypsin", "MS", "cleavage agent name"),
    "Trypsin/P": ("MS:1001313", "Trypsin/P", "MS", "cleavage agent name"),
    "Lys-C": ("MS:1001309", "Lys-C", "MS", "cleavage agent name"),
    "Chymotrypsin": ("MS:1001306", "Chymotrypsin", "MS", "cleavage agent name"),
    "Asp-N": ("MS:1001303", "Asp-N", "MS", "cleavage agent name"),
    "Glu-C": ("MS:1001917", "Glu-C", "MS", "cleavage agent name"),
    # Dissociation methods
    "HCD": ("MS:1000422", "beam-type collision-induced dissociation", "MS", "dissociation method"),
    "CID": ("MS:1000133", "collision-induced dissociation", "MS", "dissociation method"),
    "ETD": ("MS:1000598", "electron transfer dissociation", "MS", "dissociation method"),
    "EThcD": ("MS:1002631", "electron transfer higher-energy collision dissociation", "MS", "dissociation method"),
}


def generate_ontology_parquet(sdrf_df: pd.DataFrame, output_path: str) -> None:
    """Generate .ontology.parquet with field-to-CV mappings from SDRF metadata.

    Resolves instrument, enzyme, and dissociation method names found in the SDRF
    to PSI-MS controlled vocabulary accessions.
    """
    col_instrument = "comment[instrument]"
    col_enzyme = "comment[cleavage agent details]"
    col_dissoc = "comment[dissociation method]"

    records = []
    seen = set()

    # Collect unique values from SDRF
    terms_by_view = []
    if col_instrument in sdrf_df.columns:
        for v in sdrf_df[col_instrument].dropna().unique():
            name = _extract_nt(v)
            if name:
                terms_by_view.append((name, "run"))
    if col_enzyme in sdrf_df.columns:
        for v in sdrf_df[col_enzyme].dropna().unique():
            name = _extract_nt(v)
            if name:
                terms_by_view.append((name, "run"))
    if col_dissoc in sdrf_df.columns:
        for v in sdrf_df[col_dissoc].dropna().unique():
            name = _extract_nt(v)
            if name:
                terms_by_view.append((name, "run"))

    # Also map organism names
    col_organism = "characteristics[organism]"
    if col_organism in sdrf_df.columns:
        for v in sdrf_df[col_organism].dropna().unique():
            name = str(v).strip()
            if name and not _is_sentinel(name):
                terms_by_view.append((name, "sample"))

    for name, view in terms_by_view:
        key = (name, view)
        if key in seen:
            continue
        seen.add(key)

        cv = _KNOWN_CV_TERMS.get(name)
        records.append({
            "field_name": name,
            "ontology_name": cv[1] if cv else name,
            "ontology_accession": cv[0] if cv else None,
            "ontology_source": cv[2] if cv else None,
            "ontology_version": None,
            "view": view,
            "description": cv[3] if cv else None,
            "source_column_name": None,
            "source_tool": None,
        })

    if not records:
        # Write empty table with correct schema
        table = pa.table({f: [] for f in QPX_ONTOLOGY_SCHEMA.names}, schema=QPX_ONTOLOGY_SCHEMA)
    else:
        table = pa.table({
            f: [r[f] for r in records] for f in QPX_ONTOLOGY_SCHEMA.names
        }, schema=QPX_ONTOLOGY_SCHEMA)

    pq.write_table(table, output_path, compression="zstd")
    logger.info(f"Wrote {len(records)} ontology entries to {output_path}")


# -----------------------------------------------------------------------
# Top-level SDRF -> QPX conversion
# -----------------------------------------------------------------------

def convert_sdrf_to_qpx(sdrf_path: str, output_dir: str, project_id: str = None):
    """Convert an SDRF file to the full QPX metadata set.

    Produces: .sample.parquet, .run.parquet, .ontology.parquet, .dataset.parquet.

    Args:
        sdrf_path: Path to the SDRF TSV file.
        output_dir: Directory to write output files.
        project_id: Project identifier for file naming. If None, derived from SDRF filename.

    Returns:
        Tuple of (sample_path, run_path, ontology_path, dataset_path).
    """
    if project_id is None:
        project_id = Path(sdrf_path).name.split('.sdrf')[0]

    sdrf_df = pd.read_csv(sdrf_path, sep="\t")
    sdrf_df.columns = sdrf_df.columns.str.lower()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sample_path = str(Path(output_dir) / f"{project_id}.sample.parquet")
    run_path = str(Path(output_dir) / f"{project_id}.run.parquet")
    ontology_path = str(Path(output_dir) / f"{project_id}.ontology.parquet")
    dataset_path = str(Path(output_dir) / f"{project_id}.dataset.parquet")

    generate_sample_parquet(sdrf_df, sample_path)
    generate_run_parquet(sdrf_df, run_path)
    generate_ontology_parquet(sdrf_df, ontology_path)

    # Extract project accession (PXDnnnnn) from project_id
    acc_match = re.search(r'(PXD\d+)', project_id)
    project_accession = acc_match.group(1) if acc_match else project_id
    generate_dataset_parquet(project_accession, output_dir, dataset_path)

    return sample_path, run_path, ontology_path, dataset_path


@click.command("convert-to-qpx", short_help="Convert MSNet parquet + SDRF to QPX format")
@click.option("--input", "input_path", required=True,
              type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help="Path to MSNet parquet file")
@click.option("--sdrf", "sdrf_path", required=False, default=None,
              type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help="Path to SDRF TSV file (generates .run.parquet and .sample.parquet)")
@click.option("--output", "output_path", required=True,
              type=click.Path(file_okay=True, dir_okay=False),
              help="Output QPX PSM parquet path (e.g., PXD014877.psm.parquet)")
def convert_to_qpx_cmd(input_path: str, sdrf_path: str, output_path: str) -> None:
    """Convert an MSNet parquet file to QPX format.

    Produces .psm.parquet from the MSNet parquet. If --sdrf is provided,
    also generates .run.parquet and .sample.parquet alongside the PSM file.
    """
    convert_msnet_to_qpx(input_path, output_path)
    click.echo(f"Converted PSM: {input_path} -> {output_path}")

    if sdrf_path:
        output_dir = str(Path(output_path).parent)
        sample_path, run_path, ontology_path, dataset_path = convert_sdrf_to_qpx(sdrf_path, output_dir)
        click.echo(f"Generated sample: {sample_path}")
        click.echo(f"Generated run: {run_path}")
        click.echo(f"Generated ontology: {ontology_path}")
        click.echo(f"Generated dataset: {dataset_path}")
