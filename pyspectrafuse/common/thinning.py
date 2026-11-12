"""Deterministic thinning of over-represented peptidoform+charge groups before clustering."""
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc


def first_scan(scan) -> pa.ChunkedArray:
    """Scan number as int64 (first element of a list scan); null scans are rejected."""
    if isinstance(scan, pa.Array):
        scan = pa.chunked_array([scan])
    if pa.types.is_list(scan.type) or pa.types.is_large_list(scan.type):
        scan = pc.list_element(scan, 0)
    if scan.null_count:
        raise ValueError(f"{scan.null_count} PSMs without a scan number")
    return pc.cast(scan, pa.int64())


def _codes(x) -> np.ndarray:
    """Integer codes whose order is the lexicographic order of the values."""
    if isinstance(x, (pa.Array, pa.ChunkedArray)):
        codes, uniq = pd.factorize(pd.array(x, dtype=pd.ArrowDtype(x.type)))
        uniq = uniq.to_numpy(dtype=object)
    else:
        codes, uniq = pd.factorize(np.asarray(x))
        uniq = np.asarray(uniq)
    if (codes < 0).any():
        raise ValueError("null values in a thinning key")
    rank = np.empty(len(uniq), np.int64)
    rank[np.argsort(uniq, kind="stable")] = np.arange(len(uniq))
    return rank[codes]


def _num(x, dtype) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def _first_of_groups(key: np.ndarray) -> np.ndarray:
    return np.r_[True, key[1:] != key[:-1]] if len(key) else np.zeros(0, bool)


def thin_mask(pf, z, run, rt, cap: Optional[int]) -> np.ndarray:
    """True for rows sent to clustering: every row of a peptidoform+charge with <= cap PSMs, else cap rows
    taken at equal-count steps over the group ordered by (run, rt, row position)."""
    n = len(pf)
    keep = np.ones(n, bool)
    if cap is not None and cap < 1:
        raise ValueError("cap must be >= 1")
    if not cap or not n:
        return keep
    zz = _num(z, np.int64)
    _, g = np.unique(_codes(pf) * (int(zz.max() - zz.min()) + 1) + (zz - zz.min()), return_inverse=True)
    size = np.bincount(g)[g]
    big = np.nonzero(size > cap)[0]
    if not len(big):
        return keep
    o = big[np.lexsort((big, _num(rt, np.float64)[big], _codes(run)[big], g[big]))]
    start = np.nonzero(_first_of_groups(g[o]))[0]
    gr = np.arange(len(o)) - np.repeat(start, np.diff(np.r_[start, len(o)]))
    strat = (gr * cap) // size[o]
    first = np.r_[True, (strat[1:] != strat[:-1]) | (gr[1:] == 0)]
    keep[o] = False
    keep[o[first]] = True
    return keep


def cluster_mask(pf, z, run, scan, rt, pep, cap: Optional[int]) -> np.ndarray:
    """True for rows sent to clustering: the best-PEP row per (run, scan, charge), then thin_mask on those."""
    n = len(pf)
    rn, sc, zz = _codes(run), _num(scan, np.int64), _num(z, np.int64)
    o = np.lexsort((np.arange(n), _num(pep, np.float64), zz, sc, rn))
    first = _first_of_groups(rn[o]) | _first_of_groups(sc[o]) | _first_of_groups(zz[o])
    best = np.sort(o[first])
    keep = np.zeros(n, bool)
    keep[best[thin_mask(_codes(pf)[best], zz[best], rn[best], _num(rt, np.float64)[best], cap)]] = True
    return keep
