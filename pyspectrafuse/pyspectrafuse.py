import click
from pyspectrafuse.commands.spectrum2msp import spectrum2msp
from pyspectrafuse.commands.incremental import incremental
from pyspectrafuse.commands.convert_msnet_to_qpx import convert_to_qpx_cmd
from pyspectrafuse.commands.parquet2dat import parquet2dat
from pyspectrafuse.commands.build_cluster_db import build_cluster_db_cmd
from pyspectrafuse.commands.mzml_noid import (
    build_noid_cluster_db_cmd,
    convert_mzml_dat_cmd,
    filter_noid_clusters_cmd,
    msp_noid_cmd,
)
from pyspectrafuse import __version__

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


# Cli returns command line requests
@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version=__version__)
def cli():
    """
    pyspectrafuse - Command-line utilities for spectrum clustering and conversion.

    This tool provides commands for converting parquet files to dat format and
    generating MSP format files using various consensus spectrum strategies.
    """


cli.add_command(spectrum2msp)
cli.add_command(incremental)
cli.add_command(convert_to_qpx_cmd)
cli.add_command(parquet2dat)
cli.add_command(build_cluster_db_cmd)
cli.add_command(convert_mzml_dat_cmd)
cli.add_command(filter_noid_clusters_cmd)
cli.add_command(build_noid_cluster_db_cmd)
cli.add_command(msp_noid_cmd)


def main():
    cli()


