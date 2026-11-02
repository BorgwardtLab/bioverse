import math
from pathlib import Path

from bioverse.adapters.weng_docking import (
    R_KCAL,
    T_REF,
    _affinity_by_pdb_code,
    lookup_affinity,
    parse_affinity_table,
    parse_kd,
)

CSV = """Complex PDB,Class,Kd(M),∆G (kcal/ mol)
1AHW_AB:C,A,3.40E-09,11.6
1EZU_C:AB,EI,1.00E-06,8.2
2B42_A:B,OX,2.50E-08,10.4
1IQD_AB:C,A,<1.4E-11,>15
"""


def test_parse_kd_skips_bounds():
    assert parse_kd("3.40E-09") == 3.4e-9
    assert parse_kd("<1.4E-11") is None
    assert parse_kd(">15") is None
    assert parse_kd("") is None


def test_parse_affinity_table(tmp_path: Path):
    path = tmp_path / "PPAB_V2.csv"
    path.write_text(CSV)
    table = parse_affinity_table(path)
    assert set(table) == {"1AHW_AB:C", "1EZU_C:AB", "2B42_A:B"}
    rec = table["1AHW_AB:C"]
    assert math.isclose(rec["kd"], 3.4e-9)
    assert math.isclose(rec["pkd"], 8.468521, rel_tol=1e-5)
    assert math.isclose(rec["dg"], R_KCAL * T_REF * math.log(3.4e-9), rel_tol=1e-10)
    assert rec["dg"] < 0


def test_lookup_starred_id_and_unique_pdb_code(tmp_path: Path):
    path = tmp_path / "PPAB_V2.csv"
    path.write_text(CSV)
    table = parse_affinity_table(path)
    by_code = _affinity_by_pdb_code(table)
    assert lookup_affinity("1AHW_AB:C", table, by_code)["kd"] == 3.4e-9
    assert lookup_affinity("1EZU_C:AB   *", table, by_code)["kd"] == 1e-6
    assert lookup_affinity("2B42_B:A", table, by_code)["kd"] == 2.5e-8
    assert lookup_affinity("9ZZZ_A:B", table, by_code) is None
