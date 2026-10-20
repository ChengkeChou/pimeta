"""Differential test against R on the Senn2013 network (netmeta 3.7-0).

``tools/r_reference/netmeta_senn2013.R`` produces
``tests/data/_r_senn2013.tsv``: one ``key<TAB>value`` line per quantity, at
17 significant digits, straight out of R.  This module recomputes every one of
them with pimeta and compares.

Covered: the fitted model itself (k, m, n, Q, tau^2, I^2), the common- and
random-effects contrast matrices, the full design-by-treatment decomposition
(``decomp.design``) including ``Q.inc.random`` and the element-wise
``Q.inc.design``, P-score ranking, ``netmeasures`` and ``netsplit``.

Regenerate the fixture (needs R, netmeta and metafor)::

    Rscript tools/r_reference/netmeta_senn2013.R
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

import pimeta as ma

DATA = Path(__file__).resolve().parent / "data"
FIXTURE = DATA / "_r_senn2013.tsv"
COMPARISONS = DATA / "_senn2013_comparison.csv"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists() or not COMPARISONS.exists(),
    reason="the R reference fixture is missing")


def _read_fixture() -> dict[str, object]:
    values: dict[str, object] = {}
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, _, raw = line.partition("\t")
        raw = raw.strip()
        if key == "trts":
            values[key] = raw.split(",")
            continue
        try:
            values[key] = float(raw)
        except ValueError:
            values[key] = None if raw in ("NA", "NaN", "") else raw
    return values


REF = _read_fixture()


@pytest.fixture(scope="module")
def net():
    with COMPARISONS.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return ma.netmeta(
        [float(r["TE"]) for r in rows], [float(r["seTE"]) for r in rows],
        [r["treat1"] for r in rows], [r["treat2"] for r in rows],
        [r["studlab"] for r in rows],
        sm="MD", reference_group="plac", method_tau="REML",
    )


#: Fixed-effect quantities are algebra on the data alone; anything above a
#: rounding wobble would be a real defect.
EXACT = 1e-12
#: Anything that goes through the fitted tau^2 cannot be tighter than R's REML
#: stopping rule.  R stops with tau^2 off by ~1e-8 relative, which propagates
#: to ~3e-9 in the standard errors; 1e-7 leaves room without hiding a bug.
RANDOM = 1e-7


def _same(got, want, tol):
    """Compare a number, or ``None``/NaN, against an R reference value."""
    if want is None:
        return got is None or (isinstance(got, float) and math.isnan(got))
    if got is None:
        return False
    if math.isnan(want):
        return math.isnan(got)
    return abs(got - want) <= tol * max(abs(want), 1.0)


def _pairs(prefix: str) -> dict[str, float | None]:
    return {key.split("|", 1)[1]: value for key, value in REF.items()
            if key.startswith(prefix + "|")}


# --------------------------------------------------------------------------- #
# The fitted model
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field,key,tol", [
    ("k", "core.k", EXACT), ("m", "core.m", EXACT),
    ("n", "core.n", EXACT),
    ("Q", "core.Q", EXACT), ("df_Q", "core.df.Q", EXACT),
    ("pval_Q", "core.pval.Q", EXACT),
    ("I2", "core.I2", EXACT),
    ("tau2", "core.tau2", RANDOM),
])
def test_core_statistics_match_r(net, field, key, tol):
    assert _same(float(getattr(net, field)), REF[key], tol), (
        f"{field}: pimeta {getattr(net, field)!r} vs R {REF[key]!r}")


def test_treatment_order_matches_r(net):
    assert list(net.trts) == REF["trts"]


@pytest.mark.parametrize("field,prefix,tol", [
    ("TE_random", "TE.random", RANDOM), ("seTE_random", "seTE.random", RANDOM),
    ("TE_common", "TE.common", EXACT), ("seTE_common", "seTE.common", EXACT),
])
def test_contrast_matrices_match_r(net, field, prefix, tol):
    trts = list(net.trts)
    matrix = getattr(net, field)
    want = _pairs(prefix)
    for i, row_treat in enumerate(trts):
        for j, col_treat in enumerate(trts):
            key = f"{row_treat}:{col_treat}"
            got = float(matrix[i][j])
            assert _same(got, want[key], tol), (
                f"{field}[{key}]: pimeta {got!r} vs R {want[key]!r}")


# --------------------------------------------------------------------------- #
# Design-by-treatment decomposition
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("row", ["Total", "Within designs", "Between designs"])
def test_design_decomposition_matches_r(net, row):
    decomp = ma.decomp_design(net).Q_decomp[row]
    for index, name in enumerate(("Q", "df", "pval")):
        key = f"decomp.Q.{row}.{name}"
        assert _same(float(decomp[index]), REF[key], EXACT), (
            f"{key}: pimeta {decomp[index]!r} vs R {REF[key]!r}")


def test_within_design_heterogeneity_matches_r(net):
    rows = {r["design"]: r for r in ma.decomp_design(net).Q_het_design}
    for name, want_q in _pairs("decomp.Q.het.design.Q").items():
        assert name in rows, f"{name} missing from Q_het_design"
        assert _same(rows[name]["Q"], want_q, EXACT), (
            f"Q_het_design[{name}]: {rows[name]['Q']!r} vs {want_q!r}")


def test_detached_inconsistency_matches_r(net):
    rows = {r["design"]: r for r in ma.decomp_design(net).Q_inc_detach}
    for name, want_q in _pairs("decomp.Q.inc.detach.Q").items():
        assert name in rows, f"{name} missing from Q_inc_detach"
        assert _same(rows[name]["Q"], want_q, EXACT), (
            f"Q_inc_detach[{name}]: {rows[name]['Q']!r} vs {want_q!r}")


def test_inconsistency_under_random_effects_matches_r(net):
    """``Q.inc.random`` — the between-designs Q with tau.within plugged in."""
    inc = ma.decomp_design(net).Q_inc_random
    for field, key in (("Q", "Q"), ("df", "df"), ("pval", "pval"),
                       ("tau.within", "tau.within")):
        want = REF[f"decomp.Q.inc.random.{key}"]
        assert _same(float(inc[field]), want, EXACT), (
            f"Q_inc_random[{field}]: {inc[field]!r} vs {want!r}")


def test_elementwise_inconsistency_matches_r(net):
    """``Q.inc.design`` is per *comparison*, so multi-arm designs repeat.

    R keys the vector by design name, so a three-arm design contributes two
    entries under the same label; the comparison here is positional.
    """
    dd = ma.decomp_design(net)
    got_names = list(dd.Q_inc_design_names)
    got_values = [float(v) for v in dd.Q_inc_design]
    want = _pairs("decomp.Q.inc.design")
    assert len(got_values) == len(want), (
        f"pimeta returned {len(got_values)} contributions, R {len(want)}")
    for key, value in want.items():
        position, _, design = key.partition(":")
        index = int(position) - 1
        assert got_names[index] == design, (
            f"contribution {position}: pimeta {got_names[index]!r}, R {design!r}")
        assert _same(got_values[index], value, EXACT), (
            f"Q.inc.design[{position}] {design}: "
            f"{got_values[index]!r} vs R {value!r}")


# --------------------------------------------------------------------------- #
# Ranking, direct/indirect evidence
# --------------------------------------------------------------------------- #
def test_p_scores_match_r(net):
    rank = ma.netrank(net)
    names = list(rank.trts)
    for field, prefix in (("pscore_random", "netrank.random"),
                          ("pscore_common", "netrank.common")):
        want = _pairs(prefix)
        for name, value in zip(names, getattr(rank, field)):
            assert _same(float(value), want[name], RANDOM), (
                f"{prefix}[{name}]: {value!r} vs R {want[name]!r}")


@pytest.mark.parametrize("field,prefix", [
    ("proportion", "netmeasures.proportion"),
    ("meanpath", "netmeasures.meanpath"),
    ("minpar", "netmeasures.minpar"),
])
def test_netmeasures_match_r(net, field, prefix):
    got = getattr(ma.netmeasures(net), field)
    want = _pairs(prefix)
    assert set(got) == set(want)
    for comparison, value in want.items():
        assert _same(float(got[comparison]), value, RANDOM), (
            f"netmeasures.{field}[{comparison}]: {got[comparison]!r} vs R {value!r}")


def test_netsplit_direct_and_indirect_match_r(net):
    split = ma.netsplit(net)
    comparisons = list(split.comparison)
    assert comparisons == list(_pairs("netsplit.prop.random"))
    for prefix, column in (("netsplit.TE.direct.random", ("direct_random", "TE")),
                           ("netsplit.seTE.direct.random", ("direct_random", "seTE")),
                           ("netsplit.TE.indirect.random", ("indirect_random", "TE")),
                           ("netsplit.seTE.indirect.random", ("indirect_random", "seTE"))):
        block = getattr(split, column[0])[column[1]]
        want = _pairs(prefix)
        for comparison, value in zip(comparisons, block):
            got = None if value is None else float(value)
            assert _same(got, want[comparison], RANDOM), (
                f"{prefix}[{comparison}]: {got!r} vs R {want[comparison]!r}")


def test_netsplit_proportions_match_r(net):
    split = ma.netsplit(net)
    want = _pairs("netsplit.prop.random")
    for comparison, value in zip(split.comparison, split.prop_random):
        assert _same(float(value), want[comparison], RANDOM), (
            f"prop.random[{comparison}]: {value!r} vs R {want[comparison]!r}")
