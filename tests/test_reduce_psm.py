"""Tests for reduce-psm: cluster contradiction, weak evidence, RT-stratified cap, multi-part input, verify."""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from pyspectrafuse.pyspectrafuse import cli
from pyspectrafuse.reduce_psm import longest_ion_series, reduce_psms, verify_reduced

CVP = pa.list_(pa.struct([("cv_name", pa.string()), ("cv_value", pa.string())]))


def _ions(seq, kind):
    n = len(seq)
    if kind == "strong":
        return [f"y{k}" for k in range(1, n)] + [None, "b2-H2O"]
    if kind == "weak":
        return ["b2", "y4", "y6", None, None]
    return ["b2", "b3", "b4", "b5", "y1", "y2"]


def _project(tmp_path):
    """PSM rows + clusters. Returns (psm_dir, tsv, titles, rows) with rows as dicts in file order."""
    rows, clusters = [], []

    def add(seq, pf, run, rt, pep, cid, kind="ok"):
        rows.append(dict(sequence=seq, peptidoform=pf, charge=2, posterior_error_probability=pep, is_decoy=False,
                         observed_mz=500.0, calculated_mz=500.0, run_file_name=run, scan=[len(rows) + 1],
                         cv_params=[{"cv_name": "source", "cv_value": "msnet"}],
                         mz_array=[100.0 + i for i in range(20)], intensity_array=[1.0 + i for i in range(20)],
                         ion_type_array=_ions(seq, kind), rt=rt))
        clusters.append(cid)

    for i in range(10):
        add("PEPTIDEK", "PEPTIDEK", f"r{i % 5}", 100.0 + i, 1e-4, 1)
    add("WRONGSEQK", "WRONGSEQK", "r0", 50.0, 0.04, 1)            # contradicted by a 10/11 majority
    for i in range(3):
        add("PEPTIDEK", "PEPTIDEK", f"r{i}", 300.0 + i, 1e-3, 2)
    add("PEPTLDEK", "PEPTLDEK", "r4", 310.0, 1e-3, 2)              # I/L isomer: kept, cluster_conflict 1
    add("WEAKPEPTK", "WEAKPEPTK", "r1", 400.0, 0.01, 3, "weak")    # one run, longest series 1: removed
    add("STRONGPEPK", "STRONGPEPK", "r1", 410.0, 0.01, 4, "strong")  # one run, full y series: kept
    for i in range(50):                                            # 50 clusters of one peptidoform: capped
        add("MANYPEPK", "MANYPEPK", f"r{i % 7}", float(i * 10), 1e-5 * (i + 1), 100 + i)
    add("LONEPEPTK", "LONEPEPTK", "r2", 600.0, 1e-3, None)                 # never clustered (too few peaks)
    add("LONEPEPTK", "LONEPEPTK", "r3", 601.0, 1e-3, None)

    psm_dir = tmp_path / "proj"
    psm_dir.mkdir()
    schema = pa.schema([("sequence", pa.string()), ("peptidoform", pa.string()), ("charge", pa.int16()),
                        ("posterior_error_probability", pa.float64()), ("is_decoy", pa.bool_()),
                        ("observed_mz", pa.float64()), ("calculated_mz", pa.float64()),
                        ("run_file_name", pa.string()), ("scan", pa.list_(pa.int32())), ("cv_params", CVP),
                        ("mz_array", pa.list_(pa.float32())), ("intensity_array", pa.list_(pa.float32())),
                        ("ion_type_array", pa.list_(pa.string())), ("rt", pa.float32())])
    half = 37
    pq.write_table(pa.Table.from_pylist(rows[:half], schema), psm_dir / "proj-P1.psm.parquet")
    pq.write_table(pa.Table.from_pylist(rows[half:], schema), psm_dir / "proj-P2.psm.parquet")
    (psm_dir / "proj.run.parquet").write_bytes(b"not a psm file")
    titles = tmp_path / "proj_charge2.scan_titles.txt"
    tsv = tmp_path / "clusters_p30.tsv"
    with open(titles, "w") as ft, open(tsv, "w") as fc:
        for scannr, (r, c) in enumerate(zip(rows, clusters)):
            if c is None:
                continue
            ft.write(f"0\t{scannr}\tid=mzspec::{r['run_file_name']}:scan:{r['scan'][0]}:{r['peptidoform']}/2\n")
            fc.write(f"./dummy_mgf/proj_charge2_w0.mgf\t{scannr}\tw0_{c}\n")
    return psm_dir, str(tsv), str(titles), rows


def _files(psm_dir):
    return sorted(str(p) for p in Path(psm_dir).glob("*.psm.parquet"))


def test_longest_ion_series():
    ions = pa.array([["b2", "b3", "b4", "y1", "y2"], ["b2", "y4", "y6", None], [], None, ["y1-H2O", "b9"],
                     ["y1", "y2", "y3", "y4", "y4", "y5"]], pa.list_(pa.string()))
    plen = np.array([8, 8, 8, 8, 8, 6])
    assert longest_ion_series(ions, plen).tolist() == [3, 1, 0, 0, 0, 5]


def test_reduce_rules(tmp_path):
    psm_dir, tsv, titles, rows = _project(tmp_path)
    out = str(tmp_path / "out.psm.parquet")
    st = reduce_psms(_files(psm_dir), [tsv], [titles], out, max_rows_per_pfz=5, weak_filter=True)
    assert st["psms_in"] == len(rows)
    assert st["psms_contradicted_removed"] == 1 and st["pfz_lost_contradicted"] == 1
    assert st["psms_weak_removed"] == 1 and st["pfz_lost_weak"] == 1
    assert st["psms_unclustered"] == 2
    t = pq.read_table(out)
    pfs = t["peptidoform"].to_pylist()
    assert "WRONGSEQK" not in pfs and "WEAKPEPTK" not in pfs
    assert "STRONGPEPK" in pfs and "PEPTLDEK" in pfs
    assert pfs.count("PEPTIDEK") == 2          # best-PEP row of each of its two clusters
    assert pfs.count("MANYPEPK") == 5 and pfs.count("LONEPEPTK") == 2
    many_rt = sorted(r for r, p in zip(t["rt"].to_pylist(), pfs) if p == "MANYPEPK")
    assert [int(r // 100) for r in many_rt] == [0, 1, 2, 3, 4]   # one row per RT fifth
    sc = {p: {x["score_name"]: x["score_value"] for x in s} for p, s in zip(pfs, t["additional_scores"].to_pylist())}
    assert sc["PEPTLDEK"]["cluster_conflict"] == 1
    assert sc["MANYPEPK"]["pfz_psms"] == 50 and sc["MANYPEPK"]["pfz_runs"] == 7
    assert all(cv[0] == {"cv_name": "source", "cv_value": "msnet"} and cv[-1]["cv_name"] == "cluster_id"
               for cv in t["cv_params"].to_pylist())
    md = pq.ParquetFile(out).schema_arrow.metadata
    params = json.loads(md[b"clustering_params"])
    assert params["max_rows_per_pfz"] == 5 and params["sources"] == ["proj-P1.psm.parquet", "proj-P2.psm.parquet"]
    assert b"longest b/y ion series" in md[b"clustering_rule"]
    v = verify_reduced(_files(psm_dir), out)
    assert v["rows_checked_identical"] == t.num_rows


def test_reduce_deterministic_and_no_weak(tmp_path):
    psm_dir, tsv, titles, _ = _project(tmp_path)
    a, b, c = (str(tmp_path / f"{x}.psm.parquet") for x in "abc")
    reduce_psms(_files(psm_dir), [tsv], [titles], a, max_rows_per_pfz=5, weak_filter=True)
    reduce_psms(_files(psm_dir), [tsv], [titles], b, max_rows_per_pfz=5, weak_filter=True)
    assert np.array_equal(np.load(a + ".keep.npy"), np.load(b + ".keep.npy"))
    st = reduce_psms(_files(psm_dir), [tsv], [titles], c, max_rows_per_pfz=5, weak_filter=False)
    assert st["psms_weak_removed"] == 0
    assert "WEAKPEPTK" in pq.read_table(c, columns=["peptidoform"])["peptidoform"].to_pylist()


def test_reduce_rejects_decoys(tmp_path):
    psm_dir, tsv, titles, _ = _project(tmp_path)
    f = _files(psm_dir)[0]
    t = pq.read_table(f)
    t = t.set_column(t.schema.get_field_index("is_decoy"), "is_decoy", pa.array([True] + [False] * (t.num_rows - 1)))
    pq.write_table(t, f)
    with pytest.raises(ValueError, match="decoy"):
        reduce_psms(_files(psm_dir), [tsv], [titles], str(tmp_path / "o.psm.parquet"))


def test_verify_catches_changed_row(tmp_path):
    psm_dir, tsv, titles, _ = _project(tmp_path)
    out = str(tmp_path / "out.psm.parquet")
    reduce_psms(_files(psm_dir), [tsv], [titles], out, max_rows_per_pfz=5)
    t = pq.read_table(out)
    rt = t["rt"].to_pylist()
    rt[0] += 1
    pq.write_table(t.set_column(t.schema.get_field_index("rt"), "rt", pa.array(rt, pa.float32())), out)
    with pytest.raises(AssertionError, match="rt differs"):
        verify_reduced(_files(psm_dir), out)


def test_cli(tmp_path):
    psm_dir, tsv, titles, _ = _project(tmp_path)
    out = str(tmp_path / "cli.psm.parquet")
    r = CliRunner().invoke(cli, ["reduce-psm", "--psm-dir", str(psm_dir), "--clusters", tsv, "--titles", titles,
                                 "-o", out, "--max-rows-per-pfz", "5"])
    assert r.exit_code == 0, r.output
    assert "VERIFIED" in r.output
    assert "WEAKPEPTK" in pq.read_table(out, columns=["peptidoform"])["peptidoform"].to_pylist()


def test_convert_dat_combine(tmp_path):
    psm_dir, _, _, rows = _project(tmp_path)
    for p in _files(psm_dir):     # convert-dat needs >= 15 peaks above the bin cut-off
        t = pq.read_table(p)
        mz = pa.array([[150.0 + 7.3 * i for i in range(40)]] * t.num_rows, pa.list_(pa.float32()))
        it = pa.array([[10.0 + i for i in range(40)]] * t.num_rows, pa.list_(pa.float32()))
        t = t.set_column(t.schema.get_field_index("mz_array"), "mz_array", mz)
        t = t.set_column(t.schema.get_field_index("intensity_array"), "intensity_array", it)
        pq.write_table(t, p)
    out = tmp_path / "dat"
    r = CliRunner().invoke(cli, ["convert-dat", "-p", str(psm_dir), "-o", str(out), "-c", "2", "--combine"])
    assert r.exit_code == 0, r.output
    lines = (out / "proj_charge2.scan_titles.txt").read_text().splitlines()
    assert len(lines) == len(rows)
    assert [int(x.split("\t")[1]) for x in lines] == list(range(len(rows)))
    assert (out / "proj_charge2.dat").stat().st_size == 100 * len(rows)


def test_psm_allow_list_and_project_name(tmp_path):
    from pyspectrafuse.commands.spectrum2msp import find_target_ext_files
    from pyspectrafuse.common.parquet_utils import ParquetPathHandler
    for n in ("HuiYan-E480-P1.psm.parquet", "HuiYan-E480.provenance.parquet", "x-MSNet.parquet", "x.run.parquet"):
        (tmp_path / n).write_bytes(b"")
    assert [Path(f).name for f in find_target_ext_files(str(tmp_path), ".parquet")] == ["HuiYan-E480-P1.psm.parquet"]
    assert ParquetPathHandler("a/HuiYan-E480.psm.parquet").get_item_info() == "HuiYan-E480"
    assert ParquetPathHandler("a/PXD008467-abc.psm.parquet").get_item_info() == "PXD008467"


def test_duckdb_memory_from_env(monkeypatch):
    from pyspectrafuse.common.duckdb_ops import _create_connection
    monkeypatch.setenv("SPECTRAFUSE_DUCKDB_MEMORY", "1234MB")
    con = _create_connection()
    lim = con.execute("select current_setting('memory_limit')").fetchone()[0]
    assert lim.startswith("1.1") or "1234" in lim or lim.startswith("1.2"), lim


def _thinned_project(tmp_path, cap):
    """The fixture with clusters only for the PSMs convert-dat --max-psms-per-pfz would write."""
    from pyspectrafuse.common.thinning import thin_mask
    psm_dir, tsv, titles, rows = _project(tmp_path)
    keep = thin_mask([r["peptidoform"] for r in rows], [2] * len(rows), [r["run_file_name"] for r in rows],
                     [r["rt"] for r in rows], cap)
    for path in (tsv, titles):
        lines = Path(path).read_text().splitlines()
        idx = [int(x.split("\t")[1]) for x in lines]
        Path(path).write_text("".join(x + "\n" for x, i in zip(lines, idx) if keep[i]))
    return psm_dir, tsv, titles, rows, keep


def test_thin_mask_spreads_over_runs():
    from pyspectrafuse.common.thinning import thin_mask
    n = 1000
    pf = ["A"] * n + ["B"] * 5
    run = [f"r{i % 10}" for i in range(n)] + ["r0"] * 5
    rt = list(np.arange(n, dtype=float)) + [1.0] * 5
    keep = thin_mask(pf, [3] * (n + 5), run, rt, 100)
    assert keep[:n].sum() == 100 and keep[n:].all()
    assert {run[i] for i in np.nonzero(keep[:n])[0]} == {f"r{i}" for i in range(10)}
    assert (keep == thin_mask(pf, [3] * (n + 5), run, rt, 100)).all()
    assert thin_mask(pf, [3] * (n + 5), run, rt, None).all()


def test_reduce_with_thinning(tmp_path):
    psm_dir, tsv, titles, rows, keep = _thinned_project(tmp_path, 12)
    out = str(tmp_path / "out.psm.parquet")
    st = reduce_psms(_files(psm_dir), [tsv], [titles], out, max_rows_per_pfz=5, max_psms_per_pfz_clustered=12)
    assert st["psms_thinned"] == int((~keep).sum()) == 39 and st["pfz_thinned"] == 2
    assert st["psms_unclustered"] == 2 and st["psms_thinned_dropped"] == 0
    t = pq.read_table(out)
    pfs = t["peptidoform"].to_pylist()
    assert pfs.count("MANYPEPK") == 5
    sc = {p: {x["score_name"]: x["score_value"] for x in s} for p, s in zip(pfs, t["additional_scores"].to_pylist())}
    assert sc["MANYPEPK"]["pfz_psms"] == 50 and sc["MANYPEPK"]["pfz_runs"] == 7
    assert sc["PEPTIDEK"]["pfz_psms"] == 13
    kept = set(np.load(out + ".keep.npy").tolist())
    assert kept <= set(np.nonzero(keep)[0].tolist())
    assert json.loads(pq.ParquetFile(out).schema_arrow.metadata[b"clustering_params"])["max_psms_per_pfz_clustered"] == 12
    assert verify_reduced(_files(psm_dir), out)["rows_checked_identical"] == t.num_rows


def test_reduce_thinning_mismatch(tmp_path):
    psm_dir, tsv, titles, _ = _project(tmp_path)
    with pytest.raises(ValueError, match="different"):
        reduce_psms(_files(psm_dir), [tsv], [titles], str(tmp_path / "o.psm.parquet"), max_psms_per_pfz_clustered=12)


def test_convert_dat_thinning(tmp_path):
    from pyspectrafuse.common.thinning import thin_mask
    psm_dir, _, _, rows = _project(tmp_path)
    for p in _files(psm_dir):
        t = pq.read_table(p)
        mz = pa.array([[150.0 + 7.3 * i for i in range(40)]] * t.num_rows, pa.list_(pa.float32()))
        it = pa.array([[10.0 + i for i in range(40)]] * t.num_rows, pa.list_(pa.float32()))
        t = t.set_column(t.schema.get_field_index("mz_array"), "mz_array", mz)
        t = t.set_column(t.schema.get_field_index("intensity_array"), "intensity_array", it)
        pq.write_table(t, p)
    out = tmp_path / "dat"
    r = CliRunner().invoke(cli, ["convert-dat", "-p", str(psm_dir), "-o", str(out), "-c", "2", "--combine",
                                 "--max-psms-per-pfz", "12"])
    assert r.exit_code == 0, r.output
    keep = thin_mask([x["peptidoform"] for x in rows], [2] * len(rows), [x["run_file_name"] for x in rows],
                     [x["rt"] for x in rows], 12)
    got = [(x.split("\t")[2]) for x in (out / "proj_charge2.scan_titles.txt").read_text().splitlines()]
    want = [f"id=mzspec::{x['run_file_name']}:scan:{x['scan'][0]}:{x['peptidoform']}/2"
            for x, k in zip(rows, keep) if k]
    assert got == want
    r = CliRunner().invoke(cli, ["convert-dat", "-p", str(psm_dir), "-o", str(out), "--max-psms-per-pfz", "12"])
    assert r.exit_code != 0


def test_reduce_duplicate_spectrum(tmp_path):
    psm_dir, tsv, titles, rows = _project(tmp_path)
    f = _files(psm_dir)[0]
    t = pq.read_table(f)
    dup = t.slice(0, 1)
    dup = dup.set_column(dup.schema.get_field_index("posterior_error_probability"), "posterior_error_probability",
                         pa.array([0.5], pa.float64()))
    pq.write_table(pa.concat_tables([t, dup]), f)
    out = str(tmp_path / "out.psm.parquet")
    st = reduce_psms(_files(psm_dir), [tsv], [titles], out, max_rows_per_pfz=5)
    assert st["psms_in"] == len(rows) + 1 and st["psms_duplicate_spectrum_removed"] == 1
    assert st["pfz_lost_duplicate_spectrum"] == 0
    assert 0.5 not in pq.read_table(out)["posterior_error_probability"].to_pylist()
