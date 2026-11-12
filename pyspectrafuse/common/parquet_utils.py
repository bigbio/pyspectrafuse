import glob
from pathlib import Path
from typing import List
import re


class ParquetPathHandler:
    def __init__(self, path: str):
        self.parquet_path: str = path
        self.path_obj: Path = Path(path)

    def get_mgf_filename(self, mgf_file_index: int = 1) -> str:
        """Generate MGF filename from parquet path and index.

        Args:
            mgf_file_index: Index number for the MGF file

        Returns:
            Generated MGF filename string
        """
        filename = f"{self.path_obj.parts[-1].split('.')[0]}_{str(mgf_file_index)}.mgf"
        return filename

    def get_item_info(self) -> str:
        """Extract item identifier (e.g., PXD008467) from parquet path.

        Returns:
            Item identifier (letters + digits prefix from the last path part)
        """
        # 获取路径的最后一个部分（文件名或目录名）
        last_path_part = self.path_obj.parts[-1]

        # 正则匹配：开头的「一个或多个字母 + 一个或多个数字」组合
        # 自动忽略后续的连字符、UUID、扩展名等所有内容
        match = re.search(r'^([A-Za-z]+\d+)', last_path_part)

        if not match and last_path_part.split('.')[0]:
            return last_path_part.split('.')[0]
        if not match:
            raise ValueError(f"Could not extract project ID from path segment: {last_path_part}"
                             f" (Please ensure the path contains a project identifier starting with letters followed by digits.)")

        return match.group(1)

    @staticmethod
    def iter_parquet_dir(dir_path: str) -> List[Path]:
        """Find PSM parquet files in a directory, excluding QPX metadata and output files.

        Skips .run.parquet, .sample.parquet, and pipeline output files
        (cluster_metadata, psm_cluster_membership).
        """
        # Suffixes that are QPX metadata, not PSM data
        _metadata_suffixes = {'.run.parquet', '.sample.parquet',
                              '.dataset.parquet', '.ontology.parquet'}
        _exclude_names = {'psm_cluster_membership', 'cluster_metadata'}
        _exclude_dirs = {'cluster_db', 'msp', 'mgf_output'}

        parquet_path_lst = []

        # Use glob.glob (follows symlinks) instead of Path.rglob (does not)
        for path_str in glob.glob(str(Path(dir_path) / '**/*.parquet'), recursive=True):
            parquet_file = Path(path_str)
            name = str(parquet_file)
            if any(name.endswith(s) for s in _metadata_suffixes):
                continue
            if any(ex in parquet_file.stem for ex in _exclude_names):
                continue
            if any(part in _exclude_dirs for part in parquet_file.parts):
                continue
            parquet_path_lst.append(parquet_file)

        return parquet_path_lst
