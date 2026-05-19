"""CLI commands for ID-free mzML clustering inputs and outputs."""
from pathlib import Path

import click
import pandas as pd

from pyspectrafuse.common.mzml_dat import (
    convert_mzml_dataset_to_dat,
    convert_mzml_partitions_to_dat,
    find_mzml_files,
    parse_charges,
    read_sdrf_mzml_metadata,
    write_mzml_metadata_manifest,
)
from pyspectrafuse.common.noid_cluster_db import (
    build_noid_cluster_db,
    merge_into_existing_noid_db,
)
from pyspectrafuse.common.noid_msp_utils import write_noid_msp
from pyspectrafuse.common.noid_quality import filter_cluster_tsv_by_quality

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(
    "convert-mzml-dat",
    short_help="Convert mzML MS2 spectra to MaRaCluster .dat without IDs",
)
@click.option("--mzml_dir", "-m", required=True,
              type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help="Directory containing mzML files")
@click.option("--output_dir", "-o", required=True,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Output directory for .dat, .scan_info.dat, and scan_titles files")
@click.option("--charges", default="2-6", show_default=True,
              help="Charge partitions to emit, e.g. '2-6' or '2,3,4'")
@click.option("--workers", "-w", default=1, show_default=True, type=int,
              help="Number of mzML files to convert in parallel")
@click.option("--file_idx", default=0, show_default=True, type=int,
              help="File index written into final charge-partition .dat files")
@click.option("--sdrf", default=None,
              type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help="Optional SDRF TSV for species/instrument manifest")
@click.option("--dataset_name", default=None,
              help="Dataset/project accession for no-ID sidecar and manifests")
@click.option("--default_species", default=None,
              help="Fallback species when SDRF does not provide one")
@click.option("--default_instrument", default=None,
              help="Fallback instrument when SDRF does not provide one")
@click.option("--skip_instrument", is_flag=True, default=False,
              help="Partition by species/charge only")
@click.option("--partitioned/--flat", default=False, show_default=True,
              help="Write species/instrument partitions plus partitions.tsv")
def convert_mzml_dat_cmd(
    mzml_dir: str,
    output_dir: str,
    charges: str,
    workers: int,
    file_idx: int,
    sdrf: str,
    dataset_name: str,
    default_species: str,
    default_instrument: str,
    skip_instrument: bool,
    partitioned: bool,
):
    """Convert raw mzML MS2 spectra to .dat without peptide IDs or scores."""
    if workers < 1:
        raise click.ClickException("--workers must be >= 1")
    charge_values = parse_charges(charges)

    mzml_files = find_mzml_files(mzml_dir)
    if not mzml_files:
        raise click.ClickException(f"No mzML files found in {mzml_dir}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if sdrf:
        manifest_path = str(Path(output_dir) / "mzml_metadata.tsv")
        write_mzml_metadata_manifest(mzml_files, sdrf, manifest_path)
        _warn_missing_sdrf_runs(mzml_files, sdrf)
        click.echo(f"Metadata manifest: {manifest_path}")

    if partitioned:
        if not dataset_name:
            raise click.ClickException("--dataset_name is required with --partitioned")
        partition_manifest = convert_mzml_partitions_to_dat(
            mzml_dir=mzml_dir,
            output_dir=output_dir,
            dataset_name=dataset_name,
            sdrf_path=sdrf,
            default_species=default_species,
            default_instrument=default_instrument,
            skip_instrument=skip_instrument,
            charges=charge_values,
            workers=workers,
        )
        partition_df = pd.read_csv(partition_manifest, sep="\t")
        total_written = 0
        total_skipped = 0
        for row in partition_df.itertuples(index=False):
            spectra_path = Path(output_dir) / row.spectra_path
            n_spectra = len(pd.read_parquet(spectra_path)) if spectra_path.exists() else 0
            total_written += n_spectra
            click.echo(
                f"  {row.partition_id}/{row.charge}: "
                f"{n_spectra} spectra written -> {Path(output_dir) / row.dat_path}"
            )
        click.echo(f"Partition manifest: {partition_manifest}")
    else:
        results, total_written, total_skipped = convert_mzml_dataset_to_dat(
            mzml_dir=mzml_dir,
            output_dir=output_dir,
            file_idx_start=file_idx,
            charges=charge_values,
            workers=workers,
        )

        for charge_result in results:
            click.echo(
                f"  charge{charge_result.charge}: "
                f"{charge_result.written} spectra written, "
                f"{charge_result.skipped} skipped -> {charge_result.dat_path}"
            )

    click.echo(f"\nTotal: {total_written} spectra written, {total_skipped} skipped")
    click.echo(f"Output: {output_dir}")


@click.command(
    "filter-noid-clusters",
    short_help="Filter MaRaCluster clusters using ID-free quality ratio",
)
@click.option("--cluster_tsv", required=True, type=click.Path(exists=True),
              help="MaRaCluster cluster TSV")
@click.option("--dat", "dat_files", multiple=True, type=click.Path(exists=True),
              help="Original non-windowed .dat file; repeatable")
@click.option("--dat_dir", multiple=True,
              type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help="Directory containing original non-windowed .dat files")
@click.option("--output_tsv", required=True, type=click.Path(),
              help="Filtered MaRaCluster TSV output")
@click.option("--metrics_tsv", default=None, type=click.Path(),
              help="Cluster quality metrics TSV output")
@click.option("--min_quality_ratio", default=0.5, show_default=True, type=float,
              help="Minimum fraction of good pairwise comparisons per cluster")
@click.option("--min_pair_similarity", default=0.7, show_default=True, type=float,
              help="Binary-bin cosine threshold for a good spectrum pair")
@click.option("--min_cluster_size", default=1, show_default=True, type=int,
              help="Minimum members required to keep a cluster")
@click.option("--max_pairs_per_cluster", default=10_000, show_default=True, type=int,
              help="Maximum sampled pairwise comparisons per large cluster")
def filter_noid_clusters_cmd(
    cluster_tsv: str,
    dat_files,
    dat_dir,
    output_tsv: str,
    metrics_tsv: str,
    min_quality_ratio: float,
    min_pair_similarity: float,
    min_cluster_size: int,
    max_pairs_per_cluster: int,
):
    """Filter clusters without peptide IDs, PEPs, q-values, or search results."""
    dat_paths = list(dat_files)
    for directory in dat_dir:
        dat_paths.extend(
            str(p) for p in Path(directory).glob("*.dat")
            if not p.name.endswith(".scan_info.dat")
        )
    dat_paths = sorted(set(dat_paths))
    if not dat_paths:
        raise click.ClickException("Provide at least one --dat or --dat_dir")

    if metrics_tsv is None:
        out = Path(output_tsv)
        metrics_tsv = str(out.with_suffix(".quality.tsv"))

    _, _, kept = filter_cluster_tsv_by_quality(
        cluster_tsv=cluster_tsv,
        dat_paths=dat_paths,
        output_tsv=output_tsv,
        metrics_tsv=metrics_tsv,
        min_quality_ratio=min_quality_ratio,
        min_pair_similarity=min_pair_similarity,
        min_cluster_size=min_cluster_size,
        max_pairs_per_cluster=max_pairs_per_cluster,
    )
    click.echo(f"Kept {kept} clusters")
    click.echo(f"Filtered clusters: {output_tsv}")
    click.echo(f"Quality metrics: {metrics_tsv}")


@click.command(
    "build-noid-cluster-db",
    short_help="Build no-ID cluster metadata and raw-spectrum membership",
)
@click.option("--cluster_tsv", required=True, type=click.Path(exists=True))
@click.option("--metrics_tsv", required=True, type=click.Path(exists=True))
@click.option("--spectra_parquet", required=True, type=click.Path(exists=True))
@click.option("--scan_titles", multiple=True, type=click.Path(exists=True))
@click.option("--species", required=True)
@click.option("--instrument", required=True)
@click.option("--charge", required=True)
@click.option("--output_dir", required=True, type=click.Path())
@click.option("--method_type", default="most",
              type=click.Choice(["most", "bin", "average"]))
@click.option("--existing_metadata", default=None, type=click.Path(exists=True))
@click.option("--existing_membership", default=None, type=click.Path(exists=True))
def build_noid_cluster_db_cmd(
    cluster_tsv: str,
    metrics_tsv: str,
    spectra_parquet: str,
    scan_titles,
    species: str,
    instrument: str,
    charge: str,
    output_dir: str,
    method_type: str,
    existing_metadata: str,
    existing_membership: str,
):
    """Build fresh or incremental no-ID cluster DB outputs."""
    if bool(existing_metadata) != bool(existing_membership):
        raise click.ClickException(
            "--existing_metadata and --existing_membership must be provided together"
        )
    if existing_metadata:
        if not scan_titles:
            raise click.ClickException(
                "--scan_titles is required when merging into an existing no-ID DB"
            )
        meta_path, membership_path = merge_into_existing_noid_db(
            cluster_tsv=cluster_tsv,
            metrics_tsv=metrics_tsv,
            scan_titles_files=list(scan_titles),
            spectra_path=spectra_parquet,
            existing_metadata_path=existing_metadata,
            existing_membership_path=existing_membership,
            species=species,
            instrument=instrument,
            charge_str=charge,
            output_dir=output_dir,
            method_type=method_type,
        )
    else:
        meta_path, membership_path = build_noid_cluster_db(
            cluster_tsv=cluster_tsv,
            metrics_tsv=metrics_tsv,
            spectra_path=spectra_parquet,
            species=species,
            instrument=instrument,
            charge_str=charge,
            output_dir=output_dir,
            method_type=method_type,
        )
    click.echo(f"Cluster metadata: {meta_path}")
    click.echo(f"Spectrum membership: {membership_path}")


@click.command(
    "msp-noid",
    short_help="Write no-ID MSP library from no-ID cluster metadata",
)
@click.option("--cluster_metadata", required=True, type=click.Path(exists=True))
@click.option("--output_dir", required=True, type=click.Path())
@click.option("--dataset_name", required=True)
def msp_noid_cmd(cluster_metadata: str, output_dir: str, dataset_name: str):
    """Write peptide-independent MSP output."""
    output = write_noid_msp(
        cluster_metadata_path=cluster_metadata,
        output_dir=output_dir,
        dataset_name=dataset_name,
    )
    click.echo(f"MSP file: {output}")


def _warn_missing_sdrf_runs(mzml_files, sdrf_path: str) -> None:
    local_stems = {Path(p).stem.lower() for p in mzml_files}
    sdrf_df = read_sdrf_mzml_metadata(sdrf_path)
    missing = [
        row.sdrf_data_file for row in sdrf_df.itertuples(index=False)
        if str(row.run_stem).lower() not in local_stems
    ]
    if missing:
        click.echo(
            f"Warning: SDRF lists {len(missing)} run(s) with no local mzML; "
            "continuing with local files only.",
            err=True,
        )
