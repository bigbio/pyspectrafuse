"""MSP spectral-library output for no-ID cluster metadata."""
from __future__ import annotations

from pathlib import Path
import uuid

import pandas as pd

from pyspectrafuse.common.msp_utils import MspUtil


def write_noid_msp(
    cluster_metadata_path: str,
    output_dir: str,
    dataset_name: str,
) -> str:
    """Write a gzipped MSP library from no-ID cluster metadata."""
    meta = pd.read_parquet(cluster_metadata_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{dataset_name}_{uuid.uuid4()}.msp.gz"
    blocks = [_format_noid_msp_row(row) for _, row in meta.iterrows()]
    if blocks:
        MspUtil.write2msp(str(output), "\n\n".join(blocks) + "\n\n")
    return str(output)


def _format_noid_msp_row(row: pd.Series) -> str:
    mz_values = row["consensus_mz_array"]
    intensity_values = row["consensus_intensity_array"]
    peaks = "\n".join(
        f"{mz} {intensity}" for mz, intensity in zip(mz_values, intensity_values)
    )
    return (
        f"Name: {row['cluster_id']}\n"
        f"MW: {row['precursor_mz']}\n"
        f"Comment: clusterID={row['cluster_id']} "
        f"Nreps={row['member_count']} "
        f"qualityRatio={row['cluster_quality_ratio']}\n"
        f"Num peaks: {len(mz_values)}\n"
        f"{peaks}"
    )
