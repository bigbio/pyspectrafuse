"""CLI command for converting parquet files to MaRaCluster .dat binary format."""
import click
import logging
from pathlib import Path

from pyspectrafuse.maracluster_dat import parquet_to_dat

logger = logging.getLogger(__name__)

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command("convert-dat", short_help="Convert parquet to MaRaCluster .dat binary format")
@click.option('--parquet_dir', '-p', required=True,
              type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='Directory containing .psm.parquet files')
@click.option('--output_dir', '-o', required=True,
              type=click.Path(file_okay=False, dir_okay=True),
              help='Output directory for .dat files')
@click.option('--charge', '-c', type=int, default=None,
              help='Filter to specific charge state (default: all charges)')
@click.option('--file_idx', type=int, default=0,
              help='Starting file index for multi-file batches (default: 0)')
@click.option('--combine', is_flag=True, default=False,
              help='Write all .psm.parquet files of the directory (e.g. project parts) into one '
                   '.dat set named after the directory, with scannr unique across parts')
def parquet2dat(parquet_dir: str, output_dir: str, charge: int, file_idx: int, combine: bool):
    """Convert PSM parquet files to MaRaCluster .dat binary format.

    Bypasses MGF text entirely. Output is ~15x smaller than MGF.
    MaRaCluster can read .dat files directly with the -D flag.

    Outputs per parquet file:
      - {stem}[_charge{N}].dat — 100-byte Spectrum structs
      - {stem}[_charge{N}].scan_info.dat — 16-byte ScanInfo structs
      - {stem}[_charge{N}].scan_titles.txt — maps scannr → original spectrum identity
    """
    from pyspectrafuse.commands.spectrum2msp import find_target_ext_files

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    parquet_files = find_target_ext_files(parquet_dir, '.parquet')
    if not parquet_files:
        raise click.ClickException(f"No parquet files found in {parquet_dir}")

    total_written = 0
    total_skipped = 0

    if combine:
        written, skipped, dat_path = parquet_to_dat(
            parquet_files, output_dir, file_idx=file_idx, charge_filter=charge,
            name=Path(parquet_dir).resolve().name)
        click.echo(f"  {len(parquet_files)} files: {written} spectra written, {skipped} skipped → {dat_path}")
        parquet_files = []
        total_written, total_skipped = written, skipped

    for i, pf in enumerate(parquet_files):
        written, skipped, dat_path = parquet_to_dat(
            pf, output_dir,
            file_idx=file_idx + i,
            charge_filter=charge,
        )
        total_written += written
        total_skipped += skipped
        click.echo(f"  {Path(pf).name}: {written} spectra written, {skipped} skipped → {dat_path}")

    click.echo(f"\nTotal: {total_written} spectra written, {total_skipped} skipped")
    click.echo(f"Output: {output_dir}")
