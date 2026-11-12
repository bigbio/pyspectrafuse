"""m/z windowing: fixed overlapping windows, every spectrum kept and close pairs share a window."""
from pathlib import Path

import numpy as np
import pytest

from pyspectrafuse.maracluster_dat import SCANINFO_STRUCT, SPECTRUM_STRUCT, mz_windows, split_dat_by_mz_window

def _dat(tmp_path, mzs):
    with open(tmp_path / "p_charge2.dat", "wb") as fd, open(tmp_path / "p_charge2.scan_info.dat", "wb") as fs, \
         open(tmp_path / "p_charge2.scan_titles.txt", "w") as ft:
        for i, mz in enumerate(mzs):
            fd.write(SPECTRUM_STRUCT.pack(0, i, 2, float(mz), 0.0, *([i % 300] * 40)))
            fs.write(SCANINFO_STRUCT.pack(0, i, float(mz), float(mz)))
            ft.write(f"0\t{i}\tid=mzspec::r:scan:{i}:PEP/2\n")
    return [str(tmp_path / f"p_charge2{s}") for s in (".dat", ".scan_info.dat", ".scan_titles.txt")]


def _mzs(n=20000, seed=1):
    r = np.random.default_rng(seed)
    return np.concatenate([r.normal(650, 40, n), r.uniform(200, 2000, n // 10)]).astype(np.float32)


def test_split_windows(tmp_path):
    mzs = _mzs()
    paths = _dat(tmp_path, mzs)
    res = split_dat_by_mz_window(*paths, str(tmp_path / "w"))
    assert [(r["mz_min"], r["mz_max"]) for r in res] == mz_windows()
    member = {}
    for r in res:
        scans = np.frombuffer(Path(r["dat_path"]).read_bytes(), np.uint8).reshape(-1, 100)[:, 4:8].copy().view("<u4").ravel()
        assert len(Path(r["titles_path"]).read_text().splitlines()) == len(scans) == r["n_spectra"]
        for s in scans:
            member.setdefault(int(s), set()).add(r["window_idx"])
    inside = np.nonzero((mzs >= 200) & (mzs < 2000))[0]
    assert set(member) == set(int(i) for i in inside)
    # every pair within 20 ppm shares a window
    o = inside[np.argsort(mzs[inside])]
    for a, b in zip(o[:-1], o[1:]):
        if mzs[b] - mzs[a] <= mzs[a] * 20e-6:
            assert member[int(a)] & member[int(b)], (mzs[a], mzs[b])


def test_mz_windows_fixed():
    w = mz_windows()
    assert w[0] == (200.0, 500.0) and w[1] == (499.0, 799.0) and w[-1][1] == 2000.0
    with pytest.raises(ValueError):
        mz_windows(window_size=1.0, window_overlap=1.0)
