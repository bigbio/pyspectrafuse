"""ID-based reduction of one project's PSMs to a single qpx psm file, guided by spectral clusters.

Rules, in order (all PSMs of the project are clustered first, per charge, with MaRaCluster):

1. Cluster contradiction: within a cluster, an identification whose I/L-normalised sequence differs from a
   majority that holds >= ``majority`` of the cluster's PSMs is removed.
2. Weak evidence (opt-in, for replicate-run projects): a PSM whose peptidoform+charge is seen in <= ``weak_max_runs`` runs AND whose longest
   consecutive b or y ion series is < ``weak_min_series`` is removed (needs ``ion_type_array``).
3. One row per (cluster, peptidoform, charge): the best-PEP PSM.
4. At most ``max_rows_per_pfz`` rows per peptidoform+charge: rows sorted by RT, split into that many
   equal-count slices, one hash-random row per slice.

Kept rows are copied unchanged; cluster scores go in ``additional_scores``, the cluster id in ``cv_params``,
``qvalue`` is the running mean PEP over all source PSMs, and the rule and parameters go in the footer.
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pyspectrafuse import __version__

logger = logging.getLogger(__name__)

DEFAULTS = dict(max_rows_per_pfz=30, majority=0.8, weak_max_runs=2, weak_min_series=4)
FLOAT_LISTS = ["mz_array", "intensity_array"]
SCORES = [("cluster_size", None), ("cluster_label_psms", None), ("cluster_label_fraction", True),
          ("cluster_label_runs", None), ("cluster_label_median_pep", False), ("cluster_conflict", False),
          ("pfz_psms", None), ("pfz_runs", None)]
SCORE_T = pa.struct([("score_name", pa.string()), ("score_value", pa.float64()), ("higher_better", pa.bool_())])
CVP_T = pa.struct([("cv_name", pa.string()), ("cv_value", pa.string())])
TITLE_RX = r"^id=mzspec::(.+?):scan:(\d+):.*/(\d+)$"
BATCH = 200_000


class PsmParts:
    """The PSM parquet files of one project, read in a fixed order as one table."""

    def __init__(self, files: Iterable[str]):
        self.files = sorted(str(f) for f in files)
        if not self.files:
            raise ValueError("no PSM files given")
        pfs = [pq.ParquetFile(f) for f in self.files]
        self.schema = pfs[0].schema_arrow
        for f, p in zip(self.files, pfs):
            if not p.schema_arrow.remove_metadata().equals(self.schema.remove_metadata()):
                raise ValueError(f"schema of {f} differs from {self.files[0]}")
        self.num_rows = sum(p.metadata.num_rows for p in pfs)

    def read(self, columns: List[str]) -> pa.Table:
        return pa.concat_tables([pq.read_table(f, columns=columns) for f in self.files])

    def iter_batches(self, columns: Optional[List[str]] = None, batch_size: int = BATCH):
        for f in self.files:
            yield from pq.ParquetFile(f).iter_batches(batch_size=batch_size, columns=columns)


def find_psm_files(psm_dir: str) -> List[str]:
    from pyspectrafuse.commands.spectrum2msp import find_target_ext_files
    files = find_target_ext_files(psm_dir, '.parquet')
    if not files:
        raise FileNotFoundError(f"no .psm.parquet files in {psm_dir}")
    return files


def _arrow(cur) -> pa.Table:
    f = getattr(cur, "to_arrow_table", None) or getattr(cur, "fetch_arrow_table")
    return f()


def longest_ion_series(ion_types: pa.Array, plen: np.ndarray) -> np.ndarray:
    """Longest run of consecutive b or y main-ion positions (no neutral losses) per spectrum."""
    ion_types = ion_types.combine_chunks() if isinstance(ion_types, pa.ChunkedArray) else ion_types
    n = len(ion_types)
    off = np.asarray(ion_types.offsets, np.int64)
    off = off - off[0]
    flat = ion_types.flatten()
    rid = np.repeat(np.arange(n), np.diff(off))
    ex = pc.extract_regex(flat, r"^(?P<t>[by])(?P<k>\d+)$")
    ok = np.asarray(pc.fill_null(ex.is_valid(), False))
    if not ok.any():
        return np.zeros(n, np.int32)
    nul = pa.scalar(None, pa.string())
    t = np.asarray(pc.fill_null(pc.if_else(ex.is_valid(), ex.field("t"), nul), ""), dtype=object)
    k = np.asarray(pc.fill_null(pc.cast(pc.if_else(ex.is_valid(), ex.field("k"), nul), pa.int64()), 0))
    ok &= (k > 0) & (k < plen[rid])
    key = rid[ok] * 2 + (t[ok] == "y")
    k = k[ok]
    o = np.lexsort((k, key))
    key, k = key[o], k[o]
    uniq = np.ones(len(k), bool)
    uniq[1:] = (key[1:] != key[:-1]) | (k[1:] != k[:-1])
    key, k = key[uniq], k[uniq]
    start = np.ones(len(k), bool)
    start[1:] = (key[1:] != key[:-1]) | (k[1:] != k[:-1] + 1)
    run_len = np.bincount(np.cumsum(start) - 1)
    best = np.zeros(2 * n, np.int64)
    np.maximum.at(best, key[start], run_len)
    return best.reshape(n, 2).max(axis=1).astype(np.int32)


def _longest_for_rows(parts: PsmParts, rows: np.ndarray) -> np.ndarray:
    want = np.zeros(parts.num_rows, bool)
    want[rows] = True
    res = np.full(parts.num_rows, -1, np.int32)
    base = 0
    for b in parts.iter_batches(["ion_type_array", "sequence"]):
        sel = np.nonzero(want[base:base + b.num_rows])[0]
        if len(sel):
            t = pa.Table.from_batches([b]).take(pa.array(sel))
            plen = np.asarray(pc.utf8_length(t["sequence"].combine_chunks()), np.int64)
            res[base + sel] = longest_ion_series(t["ion_type_array"].combine_chunks(), plen)
        base += b.num_rows
    assert (res[rows] >= 0).all()
    return res[rows]


def _rule_text(p: dict, weak: bool) -> str:
    s = ("best-PEP PSM per (cluster, peptidoform, charge); identifications whose I/L-normalised sequence differs "
         f"from a >={p['majority']} cluster majority removed; cluster_conflict 0 pure, 1 isomers only (I/L, mod "
         "placement), 2 other peptide with clear majority, 3 no clear majority; ")
    if weak:
        s += (f"PSMs whose peptidoform is seen in <= {p['weak_max_runs']} runs and whose longest b/y ion series is "
              f"< {p['weak_min_series']} removed; ")
    s += (f"then at most {p['max_rows_per_pfz']} rows per peptidoform+charge, one per equal-count RT slice "
          "(hash-random within slice); pfz_psms/pfz_runs = totals before the cap; qvalue = running mean PEP over "
          "all source PSMs")
    return s


def reduce_psms(psm_files: List[str], cluster_tsvs: List[str], titles: List[str], out_file: str,
                max_rows_per_pfz: int = DEFAULTS["max_rows_per_pfz"], majority: float = DEFAULTS["majority"],
                weak_max_runs: int = DEFAULTS["weak_max_runs"], weak_min_series: int = DEFAULTS["weak_min_series"],
                weak_filter: bool = False, method: str = "MaRaCluster via spectrafuse",
                memory_limit: str = "8GB", threads: int = 4) -> dict:
    """Write the reduced PSM file plus ``<out>.build.json`` and ``<out>.keep.npy``; return the stats."""
    t_start = time.time()
    parts = PsmParts(psm_files)
    n = parts.num_rows
    names = set(parts.schema.names)
    for c in ("run_file_name", "scan", "peptidoform", "charge", "posterior_error_probability", "rt", "sequence"):
        if c not in names:
            raise ValueError(f"PSM files lack required column {c}")
    if weak_filter and "ion_type_array" not in names:
        logger.warning("no ion_type_array column: weak-evidence rule disabled")
        weak_filter = False
    params = dict(max_rows_per_pfz=max_rows_per_pfz, majority=majority, weak_max_runs=weak_max_runs,
                  weak_min_series=weak_min_series, weak_filter=weak_filter)

    con = duckdb.connect()
    con.execute(f"set threads={int(threads)}; set memory_limit='{memory_limit}'")
    tmp = os.environ.get("SPECTRAFUSE_DUCKDB_TEMP_DIR")
    if tmp:
        con.execute(f"set temp_directory='{tmp}'")
    q1 = lambda s: con.execute(s).fetchone()  # noqa: E731

    t = parts.read(["run_file_name", "scan", "peptidoform", "charge", "posterior_error_probability", "rt",
                    "is_decoy"] if "is_decoy" in names else
                   ["run_file_name", "scan", "peptidoform", "charge", "posterior_error_probability", "rt"])
    if "is_decoy" in t.column_names:
        n_decoy = int(pc.sum(pc.cast(t["is_decoy"], pa.int64())).as_py() or 0)
        if n_decoy:
            raise ValueError(f"{n_decoy} decoy PSMs: PEP-derived q-values assume a target-only table")
    scan = t["scan"]
    if pa.types.is_list(scan.type):
        scan = pc.list_element(scan, 0)
    con.register("meta_arrow", pa.table({
        "rowpos": pa.array(np.arange(n, dtype=np.int64)), "run": t["run_file_name"],
        "scan": pc.cast(scan, pa.int64()), "pf": t["peptidoform"], "z": pc.cast(t["charge"], pa.int32()),
        "pep": pc.cast(t["posterior_error_probability"], pa.float64()), "rt": pc.cast(t["rt"], pa.float64())}))
    del t, scan
    con.execute("create table m as select * from meta_arrow")
    con.unregister("meta_arrow")
    if q1("select count(*) from m where pep is null")[0]:
        raise ValueError("PSMs without posterior_error_probability")
    dup = q1("select count(*) - count(distinct (run, scan, z)) from m")[0]
    if dup:
        raise ValueError(f"{dup} PSMs share (run, scan, charge): cluster membership would be ambiguous")

    con.execute(f"""create table ti as select regexp_extract(title, '{TITLE_RX}', 3)::int z, scannr,
      regexp_extract(title, '{TITLE_RX}', 1) run, regexp_extract(title, '{TITLE_RX}', 2)::bigint scan
      from read_csv({list(titles)!r}, delim='\t', header=false, quote='', escape='',
                    columns={{'fidx': 'INT', 'scannr': 'BIGINT', 'title': 'VARCHAR'}})""")
    con.execute(f"""create table tsv as select regexp_extract(mgf, '_charge(\\d+)_w\\d+\\.mgf', 1)::int z,
      scannr::double::bigint scannr, regexp_replace(cid, '\\.0$', '') cid
      from read_csv({list(cluster_tsvs)!r}, delim='\t', header=false, quote='', escape='', null_padding=true,
                    columns={{'mgf': 'VARCHAR', 'scannr': 'VARCHAR', 'cid': 'VARCHAR'}})
      where cid is not null and mgf is not null""")
    if q1("select count(*) from tsv where z is null")[0]:
        raise ValueError("cluster TSV rows without a _charge<N>_w<M>.mgf source")
    con.execute("""create table mem as select ti.run, ti.scan, ti.z, any_value(ti.z || ':' || tsv.cid) cid,
                     count(*) ncopies from tsv join ti using (z, scannr) group by ti.run, ti.scan, ti.z""")
    con.execute("""create table a as select m.*, coalesce(mem.cid, 'unclustered:' || m.rowpos::varchar) cid,
                     replace(regexp_replace(m.pf, '\\[[^\\]]*\\]|\\([^)]*\\)|[.\\-]', '', 'g'), 'I', 'L') seq
                   from m left join mem on m.run = mem.run and m.scan = mem.scan and m.z = mem.z""")
    con.execute("""create table q as select rowpos, sum(pep) over w / count(*) over w qvalue from a
                   window w as (order by pep range between unbounded preceding and current row)""")
    con.execute("""create table lab as select cid, pf, z, any_value(seq) seq, count(*) lpsm, count(distinct run) lruns,
                     median(pep) lmed from a group by cid, pf, z""")
    con.execute("""create table cl as select cid, sum(lpsm) csz, arg_max(seq, (lpsm, seq)) majseq,
                     max(lpsm)::double / sum(lpsm) topfrac, count(*) nlab, count(distinct seq) nseq
                   from lab group by cid""")
    con.execute(f"""create table k_all as select lab.cid, lab.pf, lab.z, lab.lpsm,
                     case when cl.nlab = 1 then 0 when cl.nseq = 1 then 1 when cl.topfrac >= {majority} then 2
                          else 3 end cclass,
                     (lab.seq <> cl.majseq and cl.topfrac >= {majority}) contradicted
                   from lab join cl using (cid)""")
    stats = {"psms_in": n, "clusters": q1("select count(*) from cl")[0],
             "psms_unclustered": q1("select count(*) from a where cid like 'unclustered:%'")[0],
             "spectra_in_2plus_windows": q1("select count(*) from mem where ncopies > 1")[0],
             "pfz_in": q1("select count(distinct (pf, z)) from a")[0]}
    stats["rows_contradicted_removed"], stats["psms_contradicted_removed"] = q1(
        "select count(*), coalesce(sum(lpsm), 0) from k_all where contradicted")
    stats["pfz_lost_contradicted"] = q1(
        "select count(*) from (select pf, z from k_all group by pf, z having bool_and(contradicted))")[0]
    con.execute("""create table a2 as select a.* from a
                   semi join (select cid, pf, z from k_all where not contradicted) g using (cid, pf, z)""")
    if weak_filter:
        con.execute("create table r2 as select pf, z, count(distinct run) runs from a2 group by pf, z")
        cand = np.sort(_arrow(con.execute(
            f"select a2.rowpos from a2 join r2 using (pf, z) where r2.runs <= {weak_max_runs}"))["rowpos"].to_numpy())
        lg = _longest_for_rows(parts, cand) if len(cand) else np.zeros(0, np.int32)
        con.register("lg_arrow", pa.table({"rowpos": pa.array(cand, pa.int64()), "longest": pa.array(lg, pa.int32())}))
        con.execute(f"create table weak as select rowpos from lg_arrow where longest < {weak_min_series}")
        stats["psms_low_replication_checked"] = int(len(cand))
    else:
        con.execute("create table weak (rowpos bigint)")
        stats["psms_low_replication_checked"] = 0
    con.execute("create table a3 as select a2.* from a2 anti join weak using (rowpos)")
    stats["psms_weak_removed"] = q1("select count(*) from weak")[0]
    stats["pfz_lost_weak"] = q1("""select count(*) from (select distinct pf, z from a2) x
                                   anti join (select distinct pf, z from a3) y using (pf, z)""")[0]
    con.execute("""create table k as select lab3.keep rowpos, lab3.cid, cl.csz, lab3.lpsm,
                     lab3.lpsm::double / cl.csz lfrac, lab3.lruns, lab3.lmed, kk.cclass, lab3.pf, lab3.z
                   from (select cid, pf, z, count(*) lpsm, count(distinct run) lruns, median(pep) lmed,
                           arg_min(rowpos, (pep, rowpos)) keep from a3 group by cid, pf, z) lab3
                   join cl using (cid) join k_all kk using (cid, pf, z)""")
    con.execute("create table tot as select pf, z, count(*) pfz_psms, count(distinct run) pfz_runs from a3 group by pf, z")
    con.execute("""create table s as select k.rowpos, k.pf, k.z, count(*) over (partition by k.pf, k.z) gn,
                     row_number() over (partition by k.pf, k.z order by m.rt, k.rowpos) gr
                   from k join m using (rowpos)""")
    con.execute(f"""create table capk as select arg_min(rowpos, hash(rowpos + 7919)) rowpos from (
                     select *, case when gn <= {max_rows_per_pfz} then gr
                                    else ((gr - 1) * {max_rows_per_pfz}) // gn end strat from s)
                   group by pf, z, strat""")
    stats["rows_before_cap"] = q1("select count(*) from s")[0]
    stats["psms_in_pfz_totals"] = q1("select coalesce(sum(pfz_psms), 0) from tot")[0]
    assert stats["psms_in_pfz_totals"] + stats["psms_contradicted_removed"] + stats["psms_weak_removed"] == n, stats
    kt = _arrow(con.execute("""select k.*, q.qvalue, tot.pfz_psms, tot.pfz_runs from k join capk using (rowpos)
                               join q using (rowpos) join tot using (pf, z) order by rowpos"""))
    con.close()
    keep = kt["rowpos"].to_numpy()
    np.save(out_file + ".keep.npy", keep)
    stats.update({"rows_out": len(keep), "psms_represented": int(kt["lpsm"].to_numpy().sum()),
                  "pfz_out": len(set(zip(kt["pf"].to_pylist(), kt["z"].to_pylist())))})
    stats["rows_by_conflict"] = {int(c): int(x) for c, x in zip(*np.unique(kt["cclass"].to_numpy(),
                                                                          return_counts=True))}
    assert stats["pfz_out"] + stats["pfz_lost_contradicted"] + stats["pfz_lost_weak"] == stats["pfz_in"], stats
    logger.info(f"plan {stats}")

    vals = np.stack([kt[c].to_numpy().astype(float) for c in
                     ("csz", "lpsm", "lfrac", "lruns", "lmed", "cclass", "pfz_psms", "pfz_runs")], axis=1)
    qv = kt["qvalue"].to_numpy().astype(np.float64)
    cids = kt["cid"].combine_chunks()
    del kt
    footer = {"clustering_method": method, "clustering_rule": _rule_text(params, weak_filter),
              "clustering_source_rows": str(n),
              "clustering_params": json.dumps({"pyspectrafuse": __version__, **params,
                                               "sources": [Path(f).name for f in parts.files]})}
    _write_rows(parts, keep, vals, qv, cids, out_file, footer)
    stats.update({"params": params, "pyspectrafuse": __version__, "seconds": round(time.time() - t_start),
                  "bytes_in": sum(os.path.getsize(f) for f in parts.files), "bytes_out": os.path.getsize(out_file)})
    stats["ratio"] = stats["bytes_in"] / stats["bytes_out"]
    with open(out_file + ".build.json", "w") as fh:
        json.dump(stats, fh, indent=1)
    return stats


def _write_rows(parts: PsmParts, keep, vals, qv, cids, out_file: str, footer: dict) -> None:
    n = parts.num_rows
    pos = np.full(n, -1, np.int64)
    pos[keep] = np.arange(len(keep))
    md = dict(parts.schema.metadata or {})
    md.update({k.encode(): v.encode() for k, v in footer.items()})
    names = [s for s, _ in SCORES]
    hb = [h for _, h in SCORES]
    w, base = None, 0
    for b in parts.iter_batches():
        sel = np.nonzero(pos[base:base + b.num_rows] >= 0)[0]
        p = pos[base + sel]
        base += b.num_rows
        if not len(sel):
            continue
        tb = pa.Table.from_batches([b]).take(pa.array(sel))
        flat = pa.StructArray.from_arrays([pa.array(names * len(p)), pa.array(vals[p].ravel()),
                                           pa.array(hb * len(p), pa.bool_())], fields=list(SCORE_T))
        scores = pa.ListArray.from_arrays(pa.array(np.arange(len(p) + 1) * len(SCORES), pa.int32()), flat)
        if "additional_scores" in tb.schema.names:
            col = tb.schema.get_field_index("additional_scores")
            old = tb["additional_scores"].combine_chunks().to_pylist()
            scores = pa.array([(o or []) + x for o, x in zip(old, scores.to_pylist())], tb.schema.field(col).type)
            tb = tb.set_column(col, "additional_scores", scores)
        else:
            tb = tb.append_column(pa.field("additional_scores", pa.list_(SCORE_T)), scores)
        cv = pa.ListArray.from_arrays(pa.array(np.arange(len(p) + 1), pa.int32()), pa.StructArray.from_arrays(
            [pa.array(["cluster_id"] * len(p)), pc.take(cids, pa.array(p))], fields=list(CVP_T)))
        if "cv_params" in tb.schema.names:
            col = tb.schema.get_field_index("cv_params")
            old = tb["cv_params"].combine_chunks().to_pylist()
            cv = pa.array([(o or []) + x for o, x in zip(old, cv.to_pylist())], tb.schema.field(col).type)
            tb = tb.set_column(col, "cv_params", cv)
        else:
            tb = tb.append_column(pa.field("cv_params", pa.list_(CVP_T)), cv)
        qcol = pa.array(qv[p], pa.float64())
        if "qvalue" in tb.schema.names:
            col = tb.schema.get_field_index("qvalue")
            tb = tb.set_column(col, "qvalue", qcol.cast(tb.schema.field(col).type))
        else:
            tb = tb.append_column(pa.field("qvalue", pa.float64()), qcol)
        if w is None:
            schema = tb.schema.with_metadata(md)
            enc = {f"{c}.list.element": "BYTE_STREAM_SPLIT" for c in FLOAT_LISTS if c in schema.names}
            w = pq.ParquetWriter(out_file, schema, compression="zstd", compression_level=9,
                                 use_dictionary=[c for c in schema.names if c not in FLOAT_LISTS],
                                 column_encoding=enc or None)
        w.write_table(tb.cast(schema))
    if w is None:
        raise ValueError("no rows kept")
    w.close()


def verify_reduced(psm_files: List[str], out_file: str) -> dict:
    """Check every kept row equals its source row on every source column, and the score totals add up."""
    parts = PsmParts(psm_files)
    keep = np.load(out_file + ".keep.npy")
    out = pq.ParquetFile(out_file)
    n = parts.num_rows
    if out.metadata.num_rows != len(keep) or not np.all(np.diff(keep) > 0):
        raise AssertionError("row count or order differs from the keep list")
    src_cols = parts.schema.names
    out_names = out.schema_arrow.names
    if not set(src_cols) <= set(out_names):
        raise AssertionError(f"columns missing in output: {set(src_cols) - set(out_names)}")
    mask = np.zeros(n, bool)
    mask[keep] = True
    out_it = out.iter_batches(batch_size=50_000)
    buf, base, checked, represented = None, 0, 0, 0.0
    for b in parts.iter_batches():
        sel = np.nonzero(mask[base:base + b.num_rows])[0]
        base += b.num_rows
        if not len(sel):
            continue
        want = pa.Table.from_batches([b]).take(pa.array(sel))
        got, need = [], len(sel)
        while need:
            if buf is None or buf.num_rows == 0:
                buf = pa.Table.from_batches([next(out_it)])
            k = min(need, buf.num_rows)
            got.append(buf.slice(0, k))
            buf = buf.slice(k)
            need -= k
        got = pa.concat_tables(got)
        for c in src_cols:
            if c in ("cv_params", "additional_scores"):
                n_new = 1 if c == "cv_params" else len(SCORES)
                g = got[c].to_pylist()
                if [x[:-n_new] for x in g] != [x or [] for x in want[c].to_pylist()]:
                    raise AssertionError(f"{c} differs near row {base}")
            elif c == "qvalue":
                continue
            elif not want[c].combine_chunks().equals(got[c].combine_chunks()):
                raise AssertionError(f"{c} differs near row {base}")
        cv = got["cv_params"].to_pylist()
        if not all(x and x[-1]["cv_name"] == "cluster_id" for x in cv):
            raise AssertionError("cluster_id missing from cv_params")
        for sc in got["additional_scores"].to_pylist():
            represented += {x["score_name"]: x["score_value"] for x in sc}["cluster_label_psms"]
        checked += len(sel)
    with open(out_file + ".build.json") as fh:
        expect = json.load(fh)["psms_represented"]
    qv = out.read(columns=["qvalue"])["qvalue"].to_numpy()
    res = {"psms_in": n, "rows_out": len(keep), "rows_checked_identical": checked,
           "psms_represented": int(represented), "qvalue_range": [float(qv.min()), float(qv.max())],
           "footer": {k.decode(): v.decode() for k, v in (out.schema_arrow.metadata or {}).items()
                      if k.startswith(b"clustering")}}
    if checked != len(keep) or int(represented) != expect or qv.min() < 0 or qv.max() > 1:
        raise AssertionError(f"verification failed: {res}, expected psms_represented {expect}")
    with open(out_file + ".verify.json", "w") as fh:
        json.dump(res, fh, indent=1)
    return res
