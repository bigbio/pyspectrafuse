"""Deterministic thinning of over-represented peptidoform+charge groups before clustering."""
from typing import Optional

import numpy as np
import pandas as pd


def thin_mask(pf, z, run, rt, cap: Optional[int]) -> np.ndarray:
    """True for rows sent to clustering: every row of a peptidoform+charge with <= cap PSMs, else cap rows
    taken at equal-count steps over the group ordered by (run, rt, row position)."""
    n = len(pf)
    keep = np.ones(n, bool)
    if not cap:
        return keep
    if cap < 1:
        raise ValueError("cap must be >= 1")
    df = pd.DataFrame({"pf": np.asarray(pf, dtype=object), "z": np.asarray(z, dtype=np.int64),
                       "run": np.asarray(run, dtype=object), "rt": np.asarray(rt, dtype=np.float64),
                       "pos": np.arange(n, dtype=np.int64)})
    size = df.groupby(["pf", "z"], sort=False)["pos"].transform("size").to_numpy()
    big = df[size > cap]
    if big.empty:
        return keep
    big = big.sort_values(["pf", "z", "run", "rt", "pos"], na_position="last", kind="mergesort")
    g = big.groupby(["pf", "z"], sort=False)
    gr = g.cumcount().to_numpy()
    gn = g["pos"].transform("size").to_numpy()
    strat = (gr * cap) // gn
    first = np.r_[True, (strat[1:] != strat[:-1]) | (gr[1:] == 0)]
    pos = big["pos"].to_numpy()
    keep[pos] = False
    keep[pos[first]] = True
    return keep
