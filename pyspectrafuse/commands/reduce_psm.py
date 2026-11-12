"""CLI: reduce a project's PSMs to one qpx psm file using its spectral clusters, and verify the result."""
import json
import logging

import click

from pyspectrafuse.reduce_psm import DEFAULTS, find_psm_files, reduce_psms, verify_reduced


@click.command("reduce-psm", short_help="One reduced qpx psm file per project from its spectral clusters")
@click.option("--psm-dir", required=True, type=click.Path(exists=True, file_okay=False),
              help="Project directory with the .psm.parquet file(s) that were clustered")
@click.option("--clusters", "clusters", multiple=True, required=True, type=click.Path(exists=True),
              help="Merged MaRaCluster TSV (one per charge; repeat the option)")
@click.option("--titles", "titles", multiple=True, required=True, type=click.Path(exists=True),
              help="scan_titles.txt written by convert-dat (one per charge; repeat the option)")
@click.option("--output", "-o", required=True, type=click.Path(dir_okay=False), help="Output .psm.parquet")
@click.option("--max-rows-per-pfz", default=DEFAULTS["max_rows_per_pfz"], show_default=True, type=int)
@click.option("--majority", default=DEFAULTS["majority"], show_default=True, type=float,
              help="Cluster majority fraction above which a different sequence is removed")
@click.option("--weak-max-runs", default=DEFAULTS["weak_max_runs"], show_default=True, type=int)
@click.option("--weak-min-series", default=DEFAULTS["weak_min_series"], show_default=True, type=int)
@click.option("--weak-filter/--no-weak-filter", default=False, show_default=True,
              help="Remove weak-evidence PSMs; calibrated on replicate-run projects only (fractionated projects see most true peptides in <= 2 runs)")
@click.option("--method", default="MaRaCluster via spectrafuse", show_default=True,
              help="Clustering method text written to the footer")
@click.option("--memory", default="8GB", show_default=True, help="DuckDB memory limit")
@click.option("--threads", default=4, show_default=True, type=int)
@click.option("--verify/--no-verify", default=True, show_default=True,
              help="Re-read the output and check every kept row against its source row")
def reduce_psm_cmd(psm_dir, clusters, titles, output, max_rows_per_pfz, majority, weak_max_runs,
                   weak_min_series, weak_filter, method, memory, threads, verify):
    """Reduce one project's PSMs to a single qpx psm file (see pyspectrafuse.reduce_psm for the rules)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    files = find_psm_files(psm_dir)
    stats = reduce_psms(files, list(clusters), list(titles), output, max_rows_per_pfz=max_rows_per_pfz,
                        majority=majority, weak_max_runs=weak_max_runs, weak_min_series=weak_min_series,
                        weak_filter=weak_filter, method=method, memory_limit=memory, threads=threads)
    click.echo("BUILT " + json.dumps(stats))
    if verify:
        click.echo("VERIFIED " + json.dumps(verify_reduced(files, output)))


@click.command("verify-reduced-psm", short_help="Check a reduced psm file against its source PSMs")
@click.option("--psm-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", required=True, type=click.Path(exists=True, dir_okay=False))
def verify_reduced_psm_cmd(psm_dir, output):
    """Every kept row must equal its source row; cluster_label_psms must add up to the build plan."""
    click.echo("VERIFIED " + json.dumps(verify_reduced(find_psm_files(psm_dir), output)))
