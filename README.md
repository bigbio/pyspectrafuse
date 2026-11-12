# pyspectrafuse

[![Tests](https://github.com/bigbio/pyspectrafuse/actions/workflows/tests.yml/badge.svg)](https://github.com/bigbio/pyspectrafuse/actions/workflows/tests.yml)
[![Containers](https://github.com/bigbio/pyspectrafuse/actions/workflows/pyspectrafuse-containers.yml/badge.svg)](https://github.com/bigbio/pyspectrafuse/actions/workflows/pyspectrafuse-containers.yml)
[![codecov](https://codecov.io/gh/bigbio/pyspectrafuse/branch/master/graph/badge.svg)](https://codecov.io/gh/bigbio/pyspectrafuse)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Python library and CLI for the [spectrafuse](https://github.com/bigbio/spectrafuse) pipeline -- spectral clustering and consensus library generation for mass spectrometry proteomics data.

## Features

- **Parquet to Dat Conversion**: Convert QPX parquet files directly to MaRaCluster's binary `.dat` format (~100 bytes/spectrum)
- **Cluster DB Builder**: Generate `cluster_metadata.parquet` and `psm_cluster_membership.parquet` from MaRaCluster output via DuckDB-powered joins
- **Consensus Spectrum Generation**: Multiple strategies (best, bin, most, average)
- **Incremental Clustering**: Add new datasets to existing cluster DBs without re-clustering
- **QPX Metadata Handling**: Reads species/instrument from `.run.parquet` + `.sample.parquet` (QPX format)
- **MSP Format Export**: Export consensus spectra to MSP (MassBank) format

## Benchmarks

| Dataset | PSMs | Clusters | Reduction | Purity | Time |
|---------|------|----------|-----------|--------|------|
| PXD014877 | 3,608 | 3,389 | 6.1% | 0.9996 | ~15s |
| PXD004452 | 5,490,831 | 2,996,391 | 45.4% | 0.9984 | ~40 min |
| 7 datasets | 11,797,213 | 6,912,696 | 41.4% | 0.9979 | ~87 min |

## Installation

### Using pip

```bash
pip install pyspectrafuse
```

### Development Installation

```bash
git clone https://github.com/bigbio/pyspectrafuse.git
cd pyspectrafuse
pip install -e ".[test]"
```

### Docker

```bash
docker pull ghcr.io/bigbio/pyspectrafuse:0.0.4
docker run --rm -v /data:/data ghcr.io/bigbio/pyspectrafuse:0.0.4 \
    pyspectrafuse convert-dat -p /data/project -o /data/output -c 2
```

### Building the container locally

Two helper scripts are provided under `scripts/`:

```bash
# Build the Docker image (tagged pyspectrafuse:local and ghcr.io/bigbio/pyspectrafuse:<version>)
./scripts/build_docker.sh

# Build a Singularity/Apptainer SIF
# - local:  docker build → docker-daemon:// → SIF     (needs Docker on the same host)
# - remote: pull ghcr.io/bigbio/pyspectrafuse:<tag> directly into a SIF
MODE=local  ./scripts/build_singularity.sh
MODE=remote ./scripts/build_singularity.sh
```

On HPC login nodes (e.g. EBI Codon) use `MODE=remote` — it only needs
`singularity`/`apptainer` on PATH and produces a SIF file whose name matches
the Nextflow `codon_slurm` profile's Singularity cache convention
(`ghcr.io-bigbio-pyspectrafuse-<version>.sif`).

## Requirements

- Python >= 3.10
- Click >= 8.0.0
- DuckDB >= 1.0.0
- numpy >= 1.20.0
- pandas >= 1.3.0
- pyarrow >= 5.0.0
- pyteomics >= 4.5.0
- spectrum_utils >= 0.4.0

## Usage

### Command Line Interface

#### Convert Parquet to Dat

Convert QPX parquet files directly to MaRaCluster's binary `.dat` format:

```bash
pyspectrafuse convert-dat \
    -p /path/to/project/ \
    -o dat_output/ \
    -c 2  # optional: filter by charge state
```

MaRaCluster reads `.dat` files directly with the `-D` flag, skipping its own file conversion step.

#### Build Cluster DB

Generate cluster metadata and PSM membership parquets from MaRaCluster output. Uses DuckDB for efficient joins and chunked array loading to handle millions of PSMs within ~5 GB memory:

```bash
pyspectrafuse build-cluster-db \
    --cluster_tsv clusters_p30.tsv \
    --scan_titles dat_output/project.psm_charge2.scan_titles.txt \
    --parquet_dir /path/to/project/ \
    --dataset_name PXD014877 \
    --species "Homo sapiens" \
    --instrument "Q Exactive HF" \
    --charge charge2 \
    --output_dir cluster_db/
```

#### Generate MSP Consensus Spectra

```bash
pyspectrafuse msp \
    --parquet_dir /path/to/project \
    --method_type best \
    --cluster_tsv_file /path/to/cluster.tsv \
    --species "Homo sapiens" \
    --instrument "Q Exactive HF" \
    --charge "charge2"
```

**Consensus methods:** `best` (default), `bin`, `most`, `average`

#### Incremental Clustering

Add new datasets to an existing cluster DB:

```bash
# Step 1: Extract representative spectra from existing clusters to .dat format
pyspectrafuse incremental extract-reps-dat \
    --cluster_metadata cluster_db/cluster_metadata.parquet \
    --output_dir reps_output/

# Step 2: After MaRaCluster runs on [representatives + new data], merge results
pyspectrafuse incremental merge-clusters \
    --cluster_tsv new_clusters.tsv \
    --scan_titles_dir scan_titles/ \
    --existing_metadata cluster_db/cluster_metadata.parquet \
    --existing_membership cluster_db/psm_cluster_membership.parquet \
    --new_parquet_dir /path/to/new_project/ \
    --species "Homo sapiens" \
    --instrument "Q Exactive HF" \
    --charge charge2 \
    --method_type bin
```

#### Convert MSNet to QPX

One-time data preparation to convert MSNet parquet files to QPX format:

```bash
pyspectrafuse convert-to-qpx \
    --input data.parquet \
    --sdrf data.sdrf.tsv \
    --output data.psm.parquet
```

#### Reduce a project to one ID-based psm file

After a project's PSMs are clustered (convert-dat → MaRaCluster → merged TSV per charge), write one reduced
qpx psm file with the same schema:

```bash
pyspectrafuse reduce-psm \
    --psm-dir project/ \
    --clusters charge2/clusters_p30.tsv --clusters charge3/clusters_p30.tsv \
    --titles project_charge2.scan_titles.txt --titles project_charge3.scan_titles.txt \
    -o project.psm.parquet --max-rows-per-pfz 30
```

Rules: identifications contradicted by a >= 0.8 cluster majority are removed; with `--weak-filter` (replicate-run projects only), PSMs whose
peptidoform is seen in <= 2 runs and whose longest b/y ion series is < 4 are removed (needs `ion_type_array`); the best-PEP PSM per (cluster, peptidoform, charge) is kept; at most `--max-rows-per-pfz` rows
per peptidoform+charge, spread over RT. Kept rows are unchanged; cluster scores are added to
`additional_scores`, `cluster_id` to `cv_params`, `qvalue` = running mean PEP, and the rule and parameters to
the parquet footer. The output is verified row by row (`verify-reduced-psm` repeats the check).

### Python API

```python
from pyspectrafuse.maracluster_dat import parquet_to_dat
from pyspectrafuse.common.qpx_metadata import get_metadata_dict
from pyspectrafuse.common.parquet_utils import ParquetPathHandler

# Convert parquet to dat
written, skipped, dat_path = parquet_to_dat(
    'data/PXD014877/PXD014877.psm.parquet',
    output_dir='dat_files/',
    file_idx=0,
    charge_filter=2,
)

# Get metadata from QPX parquet files
metadata = get_metadata_dict('/path/to/project/')

# List PSM parquet files
parquet_files = list(ParquetPathHandler.iter_parquet_dir('/path/to/project/'))
```

## Project Structure

```
pyspectrafuse/
├── pyspectrafuse.py                # CLI entrypoint
├── maracluster_dat.py              # Parquet to .dat binary conversion + m/z windowing
├── commands/                       # CLI command implementations
│   ├── build_cluster_db.py         # build-cluster-db command (DuckDB + chunked arrays)
│   ├── parquet2dat.py              # convert-dat command
│   ├── cluster2parquet.py          # cluster-parquet command
│   ├── spectrum2msp.py             # msp command
│   ├── incremental.py              # incremental sub-commands (extract-reps-dat, merge-clusters)
│   └── convert_msnet_to_qpx.py     # convert-to-qpx command
├── common/                         # Shared utilities
│   ├── duckdb_ops.py               # DuckDB operations (purity, stats, array scanning)
│   ├── schemas.py                  # Parquet schema definitions (CLUSTER_META, PSM_MEMBERSHIP)
│   ├── constant.py                 # ParquetSchemaAdapter (MSNet/QPX column normalization)
│   ├── qpx_metadata.py             # QPX metadata reader (species/instrument from run+sample)
│   ├── parquet_utils.py            # Parquet file handling (iter_parquet_dir, find_target_ext_files)
│   └── msp_utils.py                # MSP format utilities
├── consensus_strategy/             # Consensus spectrum strategies
│   ├── best_spetrum_strategy.py    # Best PEP/q-value selection
│   ├── binning_strategy.py         # Binning-based consensus
│   ├── most_similar_strategy.py    # Most-similar spectrum selection
│   └── average_spectrum_strategy.py # Averaged spectrum
├── incremental/                    # Incremental clustering modules
│   ├── representative_dat.py       # Extract cluster reps to .dat format
│   ├── resolve_clusters_dat.py     # Resolve cluster IDs via scan_titles
│   └── merge_results.py            # Merge new PSMs into existing cluster DB
└── cluster_parquet_combine/        # Cluster DB builder (legacy path)
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pyspectrafuse --cov-report=html

# Run specific test file
pytest tests/test_parquet_utils.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use pyspectrafuse in your research, please cite:

```bibtex
@software{pyspectrafuse,
  title = {pyspectrafuse: Python tools for spectrafuse pipeline},
  author = {BigBio Team},
  url = {https://github.com/bigbio/pyspectrafuse},
  version = {0.0.4},
  year = {2024}
}
```

## Links

- **Homepage**: https://github.com/bigbio/pyspectrafuse
- **Repository**: https://github.com/bigbio/pyspectrafuse
- **Issues**: https://github.com/bigbio/pyspectrafuse/issues

## Authors

- **BigBio Team** - ypriverol@gmail.com
