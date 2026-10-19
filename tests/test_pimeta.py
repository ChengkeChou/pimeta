"""Regression tests for ``pimeta`` against hard-coded R reference results.

Every expected value in this file was produced by running the equivalent call
in R 4.6.1 and pasting the printed scalar, so a failure means ``pimeta`` has
drifted from the reference implementation:

  * ``meta`` 8.5-0     — ``metabin`` / ``metaprop`` / ``metacor`` / ``metainc``
                         / ``metamean`` / ``metacum`` / ``metainf`` / ``setting``
  * ``metafor`` 5.2-1  — ``rma`` / ``rma.mv`` / ``gosh`` / ``selmodel`` /
                         ``transf.*`` / ``escalc`` / ``fitstats`` / ``blup``
  * ``netmeta`` 3.7-0  — ``netmeta`` / ``netrank`` / ``netmeasures`` /
                         ``netcontrib`` / ``netmetareg`` / ``netpath``

The four CSVs under ``tests/data/`` are the reference inputs and R outputs used
by the network meta-analysis tests.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

import numpy as np

import pimeta as ma
from pimeta import (  # noqa: F401
    BaujatResult,
    BeggResult,
    CumulResult,
    DecompDesignResult,
    EggerResult,
    EscalcResult,
    FsnResult,
    GoshResult,
    InfluenceResult,
    LeaveOneOutResult,
    MetabResult,
    MetacontResult,
    MetacorResult,
    MetagenResult,
    MetacumResult,
    MetainfResult,
    MetaincResult,
    MetameanResult,
    MetapropResult,
    MetarateResult,
    MetaregResult,
    NetcombResult,
    NetconnectionResult,
    NetdistanceResult,
    NetleagueResult,
    NetmetaResult,
    NetmetabinResult,
    NetpairwiseResult,
    NetrankResult,
    NetsplitResult,
    PairwiseData,
    PooledResult,
    PredictResult,
    RegtestResult,
    RmaMvResult,
    RmaResult,
    RmaGlmmResult,
    SelModelResult,
    StudyEffect,
    Tau2Confint,
    TrimfillResult,
    RmaMhResult,
    RmaPetoResult,
    baujat,
    begg_test,
    confint_tau2,
    hc_rma,
    predict_rma,
    cooks_distance,
    cumul,
    decomp_design,
    discomb,
    egger_test,
    escalc,
    estimate_tau2,
    fsn,
    gosh,
    hatvalues,
    influence,
    leave1out,
    metabin,
    metacont,
    metacor,
    metacum,
    metagen,
    metainf,
    metainc,
    metamean,
    metaprop,
    metarate,
    anova_rma,
    metareg,
    netcomb,
    netconnection,
    netdistance,
    netleague,
    netmeta,
    netmetabin,
    netpairwise,
    netcontrib,
    netrank,
    netsplit,
    pairwise,
    regtest,
    rma_mh,
    rma_mv,
    rma_peto,
    rma,
    rma_glmm,
    ranef,
    robust,
    rstudent,
    selmodel,
    trimfill,
    backtransf,
    transf,
    deltamethod,
    dfround,
    fmtp,
    fmtp2,
    fmtt,
    fmtx,
    permutest,
    tes,
    vif,
    asin2ir,
    asin2p,
    conv_delta,
    conv_fivenum,
    conv_2x2,
    conv_wald,
    cor2z,
    bubble,
    drapery,
    hasse,
    hettest,
    labbe,
    llplot,
    logit2p,
    logVR2VE,
    matreg,
    metabind,
    metaadd,
    bldiag, vec2mat, cmicalc, replmiss, blsplit, contrmat, rcalc, pairmat,
    fitstats, aic_rma, bic_rma, se_rma, df_residual_rma, model_matrix_rma, blup,
    estimates, forest_args, netmatrix, netbind, netcomparison,
    read_rm5, read_cdir, read_mtv, gemtc2netmeta, crossnma2netmeta,
    heatplot, regplot,
    aggregate_escalc, cbind_escalc, rbind_escalc, cidprop, longarm,
    cilayout, format_ci, forest_dims, reporter, emmprep, addpoly, rob,
    coef, vcov, nobs, df_residual, formula, logLik, deviance, AIC, BIC,
    weights as s3_weights, fitted, residuals, predict, confint, simulate,
    hatvalues_rma, rstandard_rma, ranef_rma, summary, summary_text, plot,
    metacr,
    MetacrResult,
    gs,
    setvals,
    metamerge,
    netheat,
    netcomplex,
    netcomplexity,
    netmetareg,
    netmeasures,
    netimpact,
    netpath,
    NetmetaregResult,
    NetmeasuresResult,
    NetimpactResult,
    nnt,
    or2smd,
    p2asin,
    p2logit,
    profile_rma,
    qqnorm_rma,
    radial,
    rankogram,
    rd,
    simulate_rma,
    smd2or,
    to_long,
    to_table,
    to_wide,
    traffic_light,
    vcalc,
    VE2logVR,
    z2cor,
    transf_abt,
    transf_ahw,
    transf_arcsin,
    transf_clestod,
    transf_dtobesd,
    transf_dtocles,
    transf_dtocliffd,
    transf_dtolnor_logis,
    transf_dtolnor_norm,
    transf_dtomd,
    transf_dtoovl,
    transf_dtorbis,
    transf_dtorpb,
    transf_dtou1,
    transf_dtou2,
    transf_dtou3,
    transf_exp_int,
    transf_exp_mode,
    transf_iabt,
    transf_iahw,
    transf_iarcsin,
    transf_ilogit,
    transf_ilogit_int,
    transf_ipft,
    transf_iprobit,
    transf_irft,
    transf_isqrt,
    transf_logit,
    transf_pft,
    transf_probit,
    transf_r2toz,
    transf_rpbtod,
    transf_rpbtorbis,
    transf_rtod,
    transf_rtorpb,
    transf_rtoz,
    transf_ztor,
    transf_ztor2,
    transf_ztor_int,
)
# dat.bcg (BCG vaccine vs tuberculosis, 13 trials)
T_POS = [4, 6, 3, 62, 33, 180, 8, 505, 29, 17, 186, 5, 27]
T_NEG = [119, 300, 228, 13536, 5036, 1361, 2537, 87886, 7470, 1699, 50448, 2493, 16886]
C_POS = [11, 29, 11, 248, 47, 372, 10, 499, 45, 65, 141, 3, 29]
C_NEG = [128, 274, 209, 12619, 5761, 1079, 619, 87892, 7232, 1600, 27197, 2338, 17825]
N_E = [t + n for t, n in zip(T_POS, T_NEG)]
N_C = [c + n for c, n in zip(C_POS, C_NEG)]


# ---------------------------------------------------------------------------
# Public API smoke test
# ---------------------------------------------------------------------------

def test_all_public_symbols_exported():
    for name in ma.__all__:
        assert hasattr(ma, name), f"missing export: {name}"
    assert callable(ma.gosh)
    assert callable(ma.netleague)
    assert callable(ma.decomp_design)
    assert callable(ma.baujat)
    assert callable(ma.rma_mv)
    assert callable(ma.robust)


# ---------------------------------------------------------------------------
# metabin (meta::metabin)
# ---------------------------------------------------------------------------

def test_metabin_rr_inverse_dl():
    r = metabin(T_POS, N_E, C_POS, N_C, sm="RR", method="Inverse", method_tau="DL")
    assert r.fixed.pooled_est == pytest.approx(-0.4302852, abs=1e-6)
    assert r.fixed.pooled_se == pytest.approx(0.04049875, abs=1e-7)
    assert r.random.pooled_est == pytest.approx(-0.7141172, abs=1e-6)
    assert r.random.pooled_se == pytest.approx(0.1787421, abs=1e-6)
    assert r.random.tau2 == pytest.approx(0.3087603, abs=1e-6)
    assert r.random.q == pytest.approx(152.233, abs=1e-2)
    assert r.random.i2 == pytest.approx(92.11735, abs=1e-4)


def test_metabin_rr_mh_dl():
    # Q.Cochrane is TRUE by default for MH + DL: Q and DL tau2 center on the
    # Mantel-Haenszel common estimate (-0.453709604683754), not the IV mean.
    r = metabin(T_POS, N_E, C_POS, N_C, sm="RR", method="MH", method_tau="DL")
    assert r.fixed.pooled_est == pytest.approx(-0.453709604683754, abs=1e-9)
    assert r.fixed.pooled_se == pytest.approx(0.039337366871294, abs=1e-9)
    assert r.random.pooled_est == pytest.approx(-0.714186119568731, abs=1e-9)
    assert r.random.pooled_se == pytest.approx(0.178913379672946, abs=1e-9)
    assert r.random.tau2 == pytest.approx(0.309496853763867, abs=1e-9)
    assert r.random.q == pytest.approx(152.567553586657, abs=1e-9)
    assert r.random.i2 == pytest.approx(92.1346316973064, abs=1e-9)
    assert r.random.h2 == pytest.approx(12.713962798896, abs=1e-9)


def test_metabin_or_mh_dl():
    r = metabin(T_POS, N_E, C_POS, N_C, sm="OR", method="MH", method_tau="DL")
    assert r.fixed.pooled_est == pytest.approx(-0.473411, abs=1e-6)
    assert r.fixed.pooled_se == pytest.approx(0.04100778, abs=1e-7)
    assert r.random.pooled_est == pytest.approx(-0.747528079266850, abs=1e-9)
    assert r.random.pooled_se == pytest.approx(0.192665648017594, abs=1e-9)
    assert r.random.tau2 == pytest.approx(0.368228051856333, abs=1e-9)
    assert r.random.q == pytest.approx(163.942579569193, abs=1e-9)
    assert r.random.i2 == pytest.approx(92.6803640448177, abs=1e-9)
    assert r.random.h2 == pytest.approx(13.661881630766098, abs=1e-9)


def test_metabin_rr_mh_reml():
    # Q.Cochrane is FALSE when method.tau != "DL": Q and tau2 revert to the
    # inverse-variance center (identical to the Inverse method).
    r = metabin(T_POS, N_E, C_POS, N_C, sm="RR", method="MH", method_tau="REML")
    assert r.fixed.pooled_est == pytest.approx(-0.453709604683754, abs=1e-9)
    assert r.random.q == pytest.approx(152.233008082373, abs=1e-9)
    assert r.random.tau2 == pytest.approx(0.313243325980895, abs=1e-9)
    assert r.random.pooled_est == pytest.approx(-0.714532348365101, abs=1e-9)


# ---------------------------------------------------------------------------
# metaprop / metacor / metainc / metamean
# ---------------------------------------------------------------------------

def test_metaprop_pft_homogeneous():
    p = metaprop([10, 15, 20], [50, 60, 70], sm="PFT", random=True, method_tau="DL")
    assert p.fixed.pooled_est == pytest.approx(0.5274349610156516, abs=1e-9)
    assert p.fixed.pooled_se == pytest.approx(0.03711348095126028, abs=1e-9)
    assert p.fixed.tau2 == pytest.approx(0.0, abs=1e-12)
    assert p.fixed.q == pytest.approx(1.09388819110537, abs=1e-9)
    assert p.fixed_prop == pytest.approx(0.2491863181924724, abs=1e-9)


def test_metaprop_pft_heterogeneous():
    p = metaprop([2, 20, 50, 5], [20, 30, 60, 40], sm="PFT", random=True, method_tau="DL")
    assert p.fixed.pooled_est == pytest.approx(0.7926613660589767, abs=1e-9)
    assert p.random.pooled_est == pytest.approx(0.7080485625759835, abs=1e-9)
    assert p.random.tau2 == pytest.approx(0.1710342123434483, abs=1e-9)
    assert p.random.q == pytest.approx(77.05331304438826, abs=1e-9)


# ---------------------------------------------------------------------------
# metarate (meta::metarate)
# ---------------------------------------------------------------------------

# Reference values from meta 8.5-0 (REML), event/time as below.
RATE_EVENT = [10, 24, 12, 1, 22, 11]
RATE_TIME = [300, 500, 250, 120, 400, 300]


def test_metarate_ir():
    m = metarate(RATE_EVENT, RATE_TIME, sm="IR", method_tau="REML")
    # analysis-scale pooled + CI (raw rate scale)
    assert m.fixed.pooled_est == pytest.approx(0.03432481, abs=1e-7)
    assert m.fixed.pooled_se == pytest.approx(0.004284333, abs=1e-8)
    assert m.random.pooled_est == pytest.approx(0.03701658, abs=1e-7)
    assert m.random.pooled_se == pytest.approx(0.007417347, abs=1e-8)
    assert m.random.tau2 == pytest.approx(0.0002128426, abs=1e-9)
    assert m.q == pytest.approx(15.81261, abs=1e-4)
    assert m.i2 == pytest.approx(0.6837966, abs=1e-6)
    # rate scale = analysis scale for IR
    assert m.fixed_rate == pytest.approx(0.03432481, abs=1e-7)
    assert m.fixed_lower == pytest.approx(0.02592767, abs=1e-7)
    assert m.fixed_upper == pytest.approx(0.04272195, abs=1e-7)
    assert m.random_rate == pytest.approx(0.03701658, abs=1e-7)


def test_metarate_irln():
    m = metarate(RATE_EVENT, RATE_TIME, sm="IRLN", method_tau="REML")
    assert m.fixed.pooled_est == pytest.approx(-3.103618, abs=1e-6)
    assert m.fixed.pooled_se == pytest.approx(0.1118034, abs=1e-7)
    assert m.random.pooled_est == pytest.approx(-3.10362, abs=1e-6)
    assert m.random.tau2 == pytest.approx(1.595139e-06, abs=1e-9)
    assert m.q == pytest.approx(5.241267, abs=1e-5)
    assert m.i2 == pytest.approx(0.0460322, abs=1e-6)
    # back-transformed = exp(analysis scale)
    assert m.fixed_rate == pytest.approx(math.exp(-3.103618), abs=1e-6)
    assert m.fixed_lower == pytest.approx(math.exp(-3.322749), abs=1e-6)
    assert m.fixed_upper == pytest.approx(math.exp(-2.884488), abs=1e-6)


def test_metarate_irs():
    m = metarate(RATE_EVENT, RATE_TIME, sm="IRS", method_tau="REML")
    assert m.fixed.pooled_est == pytest.approx(0.2039024, abs=1e-7)
    assert m.fixed.pooled_se == pytest.approx(0.01156243, abs=1e-8)
    assert m.random.pooled_est == pytest.approx(0.2009921, abs=1e-7)
    assert m.random.tau2 == pytest.approx(0.0003628906, abs=1e-9)
    assert m.i2 == pytest.approx(0.445075, abs=1e-6)
    # back-transform = square
    assert m.fixed_rate == pytest.approx(0.2039024 ** 2, abs=1e-6)
    assert m.random_rate == pytest.approx(0.2009921 ** 2, abs=1e-6)


def test_metarate_irft():
    m = metarate(RATE_EVENT, RATE_TIME, sm="IRFT", method_tau="REML")
    assert m.fixed.pooled_est == pytest.approx(0.2082791, abs=1e-7)
    assert m.random.pooled_est == pytest.approx(0.2080429, abs=1e-7)
    assert m.random.tau2 == pytest.approx(2.578493e-05, abs=1e-9)
    assert m.i2 == pytest.approx(0.3110864, abs=1e-6)
    # asin2ir back-transform at harmonic-mean time (R-verified)
    assert m.fixed_rate == pytest.approx(0.04144395, abs=1e-7)
    assert m.random_rate == pytest.approx(0.04134567, abs=1e-7)
    assert m.fixed_lower == pytest.approx(0.03252305, abs=1e-6)
    assert m.fixed_upper == pytest.approx(0.05139362, abs=1e-6)


def test_metarate_zero_event_increment():
    # zero-event study (study 4: event=0, time=120) gets incr=0.5 for IRLN
    m = metarate([10, 24, 12, 0, 22, 11], RATE_TIME, sm="IRLN", method_tau="REML")
    assert m.incr_event[3] == pytest.approx(0.5)
    assert m.incr_event[0] == pytest.approx(0.0)
    assert m.yi[3] == pytest.approx(math.log(0.5 / 120), abs=1e-9)



def test_metacor_zcor():
    c = metacor([0.3, 0.5, 0.4], [30, 40, 50], sm="ZCOR", random=True, method_tau="DL")
    assert c.fixed.pooled_est == pytest.approx(0.4377734808373274, abs=1e-9)
    assert c.fixed.pooled_se == pytest.approx(0.0949157995752499, abs=1e-9)
    assert c.fixed.tau2 == pytest.approx(0.0, abs=1e-12)
    assert c.fixed.q == pytest.approx(0.913763969274668, abs=1e-9)
    assert c.fixed_cor == pytest.approx(0.4117971860460617, abs=1e-9)


def test_metainc_irr():
    i = metainc([10, 15], [100, 200], [20, 25], [110, 190], sm="IRR", random=True, method_tau="DL")
    assert i.fixed_effect == pytest.approx(-0.577187299465077, abs=1e-9)
    assert i.fixed.pooled_se == pytest.approx(0.2496856944906246, abs=1e-9)
    assert i.random_effect == pytest.approx(-0.5769627966375221, abs=1e-9)
    assert i.random.pooled_se == pytest.approx(0.2496751135729437, abs=1e-9)
    assert i.random.tau2 == pytest.approx(0.0, abs=1e-12)
    assert i.random.q == pytest.approx(0.00497057697961933, abs=1e-9)


def test_metamean_mraw():
    m = metamean([30, 40, 50], [10, 12, 11], [2, 2.5, 3], sm="MRAW", random=True, method_tau="DL")
    assert m.fixed.pooled_est == pytest.approx(10.94346087949743, abs=1e-9)
    assert m.fixed.pooled_se == pytest.approx(0.2267139176514794, abs=1e-9)
    assert m.random.pooled_est == pytest.approx(10.99274560228829, abs=1e-9)
    assert m.random.pooled_se == pytest.approx(0.5988322301241722, abs=1e-9)
    assert m.random.tau2 == pytest.approx(0.9196095829636201, abs=1e-9)
    assert m.random.q == pytest.approx(13.8378069674472, abs=1e-9)


# ---------------------------------------------------------------------------
# metagen (meta::metagen)
# ---------------------------------------------------------------------------

def _bcg_lnrr():
    """dat.bcg lnRR treatment estimates and standard errors (Inverse method)."""
    te = [
        math.log((tp / ne) / (cp / nc))
        for tp, ne, cp, nc in zip(T_POS, N_E, C_POS, N_C)
    ]
    se = [
        math.sqrt(1 / tp - 1 / ne + 1 / cp - 1 / nc)
        for tp, ne, cp, nc in zip(T_POS, N_E, C_POS, N_C)
    ]
    return te, se


def test_metagen_rr_dl_bcg():
    te, se = _bcg_lnrr()
    m = metagen(te, se, sm="RR", method_tau="DL")
    # common (fixed) effect
    assert m.TE_common == pytest.approx(-0.4302851636540906, abs=1e-12)
    assert m.seTE_common == pytest.approx(0.04049875171086381, abs=1e-12)
    assert m.lower_common == pytest.approx(-0.5096612584262136, abs=1e-12)
    assert m.upper_common == pytest.approx(-0.3509090688819677, abs=1e-12)
    # random effects
    assert m.TE_random == pytest.approx(-0.714117222073227, abs=1e-12)
    assert m.seTE_random == pytest.approx(0.1787420895354032, abs=1e-12)
    assert m.lower_random == pytest.approx(-1.064445280084051, abs=1e-12)
    assert m.upper_random == pytest.approx(-0.3637891640624032, abs=1e-12)
    # heterogeneity (Q-based meta convention)
    assert m.Q == pytest.approx(152.2330080823734, abs=1e-12)
    assert m.df_Q == 12
    assert m.tau2 == pytest.approx(0.3087602628623765, abs=1e-12)
    assert m.tau == pytest.approx(0.5556620041557426, abs=1e-12)
    assert m.I2 == pytest.approx(0.9211734685456205, abs=1e-12)
    assert m.H == pytest.approx(3.561752940177694, abs=1e-12)
    assert m.lower_I2 == pytest.approx(0.8831637947405894, abs=1e-12)
    assert m.upper_I2 == pytest.approx(0.9468176662590829, abs=1e-12)
    # prediction interval (always computed when random effects fitted)
    assert m.lower_predict == pytest.approx(-1.985896443665729, abs=1e-12)
    assert m.upper_predict == pytest.approx(0.5576619995192749, abs=1e-12)
    assert m.seTE_predict == pytest.approx(0.5837028331555867, abs=1e-12)
    assert m.df_predict == 12
    # counts
    assert m.k == 13 and m.k_all == 13 and m.k_TE == 13
    # study weights (R-verified)
    assert m.w_common[0] == pytest.approx(3.0713968, abs=1e-6)
    assert m.w_common[7] == pytest.approx(252.4245824, abs=1e-6)
    assert m.w_random[0] == pytest.approx(1.5764292, abs=1e-6)
    assert m.w_random[7] == pytest.approx(3.1977299, abs=1e-6)


def test_metagen_backtransf_and_report():
    te, se = _bcg_lnrr()
    m = metagen(te, se, sm="RR", method_tau="DL")
    r = m.report()
    # back-transformation on the report scale = exp(analysis scale) for RR
    assert r["common"]["est"] == pytest.approx(math.exp(-0.4302851636540906), abs=1e-12)
    assert r["random"]["est"] == pytest.approx(math.exp(-0.714117222073227), abs=1e-12)
    assert r["prediction"]["lower"] == pytest.approx(math.exp(-1.985896443665729), abs=1e-12)
    assert r["prediction"]["upper"] == pytest.approx(math.exp(0.5576619995192749), abs=1e-12)
    # scalar transforms
    assert float(transf(0.5, "RR")) == pytest.approx(math.log(0.5), abs=1e-15)
    assert float(backtransf(0.0, "RR")) == pytest.approx(1.0, abs=1e-15)
    assert float(backtransf(0.0, "OR")) == pytest.approx(1.0, abs=1e-15)
    assert float(backtransf(math.log(0.3), "RR")) == pytest.approx(0.3, abs=1e-15)


# ---------------------------------------------------------------------------
# gosh (metafor::gosh)
# ---------------------------------------------------------------------------

def test_gosh_reml_exhaustive():
    esc = escalc(measure="RR", ai=T_POS, bi=T_NEG, ci=C_POS, di=C_NEG)
    g = gosh(esc.yi, esc.vi, method="REML")

    # 13 studies -> sum(choose(13, i)) = 2^13 - 1 = 8191 subsets
    assert len(g.k) == 8191

    # k counts (sorted ascending): choose(13, k)
    from collections import Counter

    counts = Counter(g.k)
    assert counts[1] == 13
    assert counts[13] == 1
    assert counts[6] == counts[7] == 1716

    # rows are sorted by k
    assert g.k == sorted(g.k)

    # full data set (k=13) equals the REML model on all studies
    assert g.QE[-1] == pytest.approx(152.233008082373, abs=1e-8)
    assert g.I2[-1] == pytest.approx(92.2213860749656, abs=1e-8)
    assert g.H2[-1] == pytest.approx(12.8557608031121, abs=1e-8)
    assert g.tau2[-1] == pytest.approx(0.313243325980895, abs=1e-9)
    assert g.estimate[-1] == pytest.approx(-0.714532348365102, abs=1e-9)

    # first (k=1) subset: study 1 only -> estimate == y1
    assert g.estimate[0] == pytest.approx(esc.yi[0], abs=1e-12)

    # two non-converging subsets (studies {1,2,5,9,11,12,13} and
    # {3,5,7,9,11,12,13}) must be NA, matching R's Fisher-scoring failure.
    na = [i for i in range(len(g.k)) if math.isnan(g.QE[i])]
    assert na == [5675, 5699]


# ---------------------------------------------------------------------------
# netmeta (netmeta::netmeta, Senn 2013) — skipped if benchmark CSV is absent
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"

SENN_CSV = DATA_DIR / "_senn2013_comparison.csv"


@pytest.mark.skipif(not SENN_CSV.exists(), reason="Senn 2013 benchmark CSV not present")
def test_netmeta_senn():
    import csv

    te, sete, t1, t2, stud = [], [], [], [], []
    with SENN_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            te.append(float(row["TE"]))
            sete.append(float(row["seTE"]))
            t1.append(row["treat1"])
            t2.append(row["treat2"])
            stud.append(row["studlab"])

    r = netmeta(te, sete, t1, t2, stud, sm="MD", reference_group="plac")
    assert r.Q == pytest.approx(96.98555, abs=1e-4)
    assert r.df_Q == pytest.approx(18.0, abs=1e-9)
    assert r.I2 == pytest.approx(0.8144053, abs=1e-5)
    assert r.tau2 == pytest.approx(0.1087169, abs=1e-6)

    pos = {t: i for i, t in enumerate(r.trts)}
    assert r.TE_common[pos["plac"], pos["metf"]] == pytest.approx(1.11411018904, abs=1e-6)
    assert r.TE_random[pos["plac"], pos["metf"]] == pytest.approx(1.12677535121, abs=1e-6)


@pytest.mark.skipif(not SENN_CSV.exists(), reason="Senn 2013 benchmark CSV not present")
def test_netpairwise_senn():
    import csv

    te, sete, t1, t2, stud = [], [], [], [], []
    with SENN_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            te.append(float(row["TE"]))
            sete.append(float(row["seTE"]))
            t1.append(row["treat1"])
            t2.append(row["treat2"])
            stud.append(row["studlab"])

    r = netmeta(te, sete, t1, t2, stud, sm="MD", reference_group="plac")
    np_ = netpairwise(r)

    # Overall pooled (metagen, DL) — R netmeta::netpairwise(Senn2013, separate.trts=TRUE).
    assert np_.k == 28
    assert np_.TE_common == pytest.approx(-0.82210972982358899, abs=1e-12)
    assert np_.seTE_common == pytest.approx(0.028088633026469061, abs=1e-12)
    assert np_.TE_random == pytest.approx(-0.78841578045926952, abs=1e-12)
    assert np_.seTE_random == pytest.approx(0.099067078208071549, abs=1e-12)
    assert np_.tau2 == pytest.approx(0.23259798447047908, abs=1e-12)
    assert np_.I2 == pytest.approx(0.91166716625734656, abs=1e-12)
    assert np_.Q == pytest.approx(305.66210610497438, abs=1e-9)
    assert np_.df_Q == pytest.approx(27.0, abs=1e-12)

    # Per-pair direct evidence (random-effects DL tau^2 per pair).
    pairs = {(p["treat1"], p["treat2"]): p for p in np_.pairs}
    b = pairs[("benf", "plac")]
    assert b["k"] == 2
    assert b["TE_common"] == pytest.approx(-0.90518644941738258, abs=1e-12)
    assert b["TE_random"] == pytest.approx(-0.68509059615384615, abs=1e-12)
    assert b["seTE_random"] == pytest.approx(0.38452986137923928, abs=1e-12)
    assert b["Q"] == pytest.approx(4.3813771307812992, abs=1e-12)
    assert b["I2"] == pytest.approx(0.77176125904010517, abs=1e-12)
    assert b["tau2"] == pytest.approx(0.23476977500000015, abs=1e-12)

    m = pairs[("metf", "plac")]
    assert m["k"] == 4
    assert m["TE_common"] == pytest.approx(-1.1523071750195584, abs=1e-12)
    assert m["TE_random"] == pytest.approx(-1.1215844029777127, abs=1e-12)
    assert m["tau2"] == pytest.approx(0.45942007423349152, abs=1e-12)

    rp = pairs[("rosi", "plac")]
    assert rp["k"] == 6
    assert rp["TE_common"] == pytest.approx(-1.148290535347845, abs=1e-12)
    assert rp["TE_random"] == pytest.approx(-1.1764389660070329, abs=1e-12)
    assert rp["tau2"] == pytest.approx(0.055627344131377737, abs=1e-12)

    # A single-study pair has NaN I2/tau2 but random == common.
    ac = pairs[("acar", "metf")]
    assert ac["k"] == 1
    assert math.isnan(ac["I2"]) and math.isnan(ac["tau2"])
    assert ac["TE_random"] == pytest.approx(0.2, abs=1e-12)

    # to_dict round-trips the overall pooled fields.
    d = np_.to_dict()
    assert d["TE_random"] == pytest.approx(-0.78841578045926952, abs=1e-12)
    assert len(d["pairs"]) == 15


def test_pairwise_binary_or():
    # netmeta::pairwise(sm="OR") — no zero cells, so no continuity correction.
    pw = pairwise(
        treat=["A", "B", "A", "B", "C"], event=[3, 5, 1, 4, 2],
        n=[10, 10, 12, 12, 12], studlab=["s1", "s1", "s2", "s2", "s2"],
        sm="OR",
    )
    assert pw.sm == "OR"
    assert pw.treat1 == ["A", "A", "A", "B"]
    assert pw.treat2 == ["B", "B", "C", "C"]
    assert pw.TE == pytest.approx([
        -0.84729786038720367, -1.70474809223842527, -0.78845736036427017,
        0.91629073187415511], abs=1e-12)
    assert pw.seTE == pytest.approx([
        0.93605046668995162, 1.21074732744247293, 1.30034960334099803,
        0.98742088290657493], abs=1e-12)


def test_pairwise_binary_rr_addincr():
    # netmeta::pairwise(sm="RR", addincr=TRUE) — add 0.5 to every event.
    pw = pairwise(
        treat=["A", "B", "A", "B", "C"], event=[3, 5, 1, 4, 2],
        n=[10, 10, 12, 12, 12], studlab=["s1", "s1", "s2", "s2", "s2"],
        sm="RR", addincr=True,
    )
    assert pw.TE == pytest.approx([
        -0.45198512374305744, -1.09861228866810978, -0.51082562376599072,
        0.58778666490211895], abs=1e-12)
    assert pw.seTE == pytest.approx([
        0.52636135596781519, 0.85374989832437986, 0.95219045713904671,
        0.67986926847903795], abs=1e-12)


def test_pairwise_continuous_md():
    pw = pairwise(
        treat=["A", "B", "A", "B", "C"], n=[20, 20, 25, 25, 25],
        mean=[1.5, 2.5, 2.0, 3.0, 2.8], sd=[0.5, 0.6, 0.7, 0.8, 0.6],
        studlab=["s1", "s1", "s2", "s2", "s2"], sm="MD",
    )
    assert pw.TE == pytest.approx([-1.0, -1.0, -0.8, 0.2], abs=1e-12)
    assert pw.seTE == pytest.approx([
        0.17464249196572981, 0.21260291625469299, 0.18439088914585774,
        0.20000000000000001], abs=1e-12)


def test_pairwise_continuous_smd():
    # Cohen's d; multi-arm SMD uses the study-level pooled SD and netmeta's
    # seTE = sqrt(1/n1 + 1/n2 + TE^2/(2*n_i)).
    pw = pairwise(
        treat=["A", "B", "A", "B", "C"], n=[20, 20, 25, 25, 25],
        mean=[1.5, 2.5, 2.0, 3.0, 2.8], sd=[0.5, 0.6, 0.7, 0.8, 0.6],
        studlab=["s1", "s1", "s2", "s2", "s2"], sm="SMD",
    )
    assert pw.TE == pytest.approx([
        -1.8107149208503706, -1.4189513095212063, -1.1351610476169647,
        0.2837902619042415], abs=1e-12)
    assert pw.seTE == pytest.approx([
        0.37547783763809156, 0.30565146620284078, 0.29764173770969288,
        0.28379026190424128], abs=1e-12)


def test_netcomb_additive_cnma():
    # netmeta::netcomb on a 3-treatment combination network (R 4.6.1,
    # netmeta 3.7-0). Treatments A, A+B, B -> components A and B.
    treat1 = ["A", "A", "A", "A", "B", "B"]
    treat2 = ["B", "B", "A+B", "A+B", "A+B", "A+B"]
    te = [0.1, 0.2, -0.3, -0.4, -0.5, -0.6]
    se = [0.2, 0.3, 0.25, 0.2, 0.3, 0.25]
    stud = ["s1", "s2", "s3", "s4", "s5", "s6"]

    r = netmeta(te, se, treat1, treat2, stud, sm="MD", common=True, random=True)
    nc = netcomb(r)

    assert nc.trts == ["A", "A+B", "B"]
    assert nc.comps == ["A", "B"]

    # Component (incremental) effects.
    assert nc.Comp_common == pytest.approx(
        [0.53112621473340704, 0.37941789594864161], abs=1e-12)
    assert nc.seComp_common == pytest.approx(
        [0.14694541790286386, 0.13305538203858094], abs=1e-12)

    # Combination (per-treatment) effects.
    assert nc.Comb_common == pytest.approx(
        [0.53112621473340704, 0.91054411068204866, 0.37941789594864161],
        abs=1e-12)
    assert nc.seComb_common == pytest.approx(
        [0.14694541790286386, 0.24396852364351390, 0.13305538203858094],
        abs=1e-12)

    # All-pairs contrast matrices (A, A+B, B order).
    assert nc.TE_common[0, 1] == pytest.approx(-0.37941789594864161, abs=1e-12)
    assert nc.TE_common[0, 2] == pytest.approx(0.15170831878476543, abs=1e-12)
    assert nc.TE_common[1, 2] == pytest.approx(0.53112621473340704, abs=1e-12)
    assert nc.seTE_common[0, 1] == pytest.approx(0.13305538203858094, abs=1e-12)
    assert nc.seTE_common[0, 2] == pytest.approx(0.13810409311589666, abs=1e-12)
    assert nc.seTE_common[1, 2] == pytest.approx(0.14694541790286386, abs=1e-12)

    # Additive-model Q / heterogeneity / inconsistency.
    assert nc.Q_additive == pytest.approx(0.29092411351685848, abs=1e-12)
    assert nc.df_Q_additive == pytest.approx(4.0, abs=1e-12)
    assert nc.pval_Q_additive == pytest.approx(0.99039249172529853, abs=1e-12)
    assert nc.Q_standard == pytest.approx(0.29092411351685837, abs=1e-12)
    assert nc.Q_diff == pytest.approx(0.0, abs=1e-12)
    assert nc.df_Q_diff == pytest.approx(0.0, abs=1e-12)
    assert nc.tau == pytest.approx(0.0, abs=1e-12)
    assert nc.I2 == pytest.approx(0.0, abs=1e-12)

    # Zero heterogeneity -> random == common.
    assert nc.TE_random == pytest.approx(nc.TE_common, abs=1e-12)

    # Component composition and design matrices.
    import numpy as np
    assert nc.C_matrix.tolist() == [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    assert nc.X_matrix.tolist() == [
        [1.0, -1.0], [1.0, -1.0], [0.0, -1.0], [0.0, -1.0], [1.0, 0.0],
        [1.0, 0.0]]

    # to_dict round-trips the key quantities.
    d = nc.to_dict()
    assert d["Comp_common"] == pytest.approx(
        [0.53112621473340704, 0.37941789594864161], abs=1e-12)
    assert d["comps"] == ["A", "B"]


def test_discomb_additive_cnma():
    # netmeta::discomb on a connected 3-treatment combination network
    # (R 4.6.1, netmeta 3.7-0). Same data as netcomb -> identical results.
    import numpy as np
    treat1 = ["A", "A", "A", "A", "B", "B"]
    treat2 = ["B", "B", "A+B", "A+B", "A+B", "A+B"]
    te = [0.1, 0.2, -0.3, -0.4, -0.5, -0.6]
    se = [0.2, 0.3, 0.25, 0.2, 0.3, 0.25]
    stud = ["s1", "s2", "s3", "s4", "s5", "s6"]

    dc = discomb(te, se, treat1, treat2, stud, sm="MD")

    assert dc.trts == ["A", "A+B", "B"]
    assert dc.comps == ["A", "B"]
    assert dc.k == 6 and dc.m == 6 and dc.n == 3 and dc.d == 3 and dc.c == 2

    # Component / combination effects match netcomb (and R discomb).
    assert dc.Comp_common == pytest.approx(
        [0.53112621473340704, 0.37941789594864161], abs=1e-12)
    assert dc.seComp_common == pytest.approx(
        [0.14694541790286386, 0.13305538203858094], abs=1e-12)
    assert dc.Comb_common == pytest.approx(
        [0.53112621473340704, 0.91054411068204866, 0.37941789594864161],
        abs=1e-12)
    assert dc.seComb_common == pytest.approx(
        [0.14694541790286386, 0.24396852364351390, 0.13305538203858094],
        abs=1e-12)

    # All-pairs contrasts (A, A+B, B order).
    assert dc.TE_common[0, 1] == pytest.approx(-0.37941789594864161, abs=1e-12)
    assert dc.TE_common[0, 2] == pytest.approx(0.15170831878476543, abs=1e-12)
    assert dc.TE_common[1, 2] == pytest.approx(0.53112621473340704, abs=1e-12)

    # Additive-model Q (df = n.a - k - rank(X) = 12 - 6 - 2 = 4).
    assert dc.Q_additive == pytest.approx(0.29092411351685848, abs=1e-12)
    assert dc.df_Q_additive == pytest.approx(4.0, abs=1e-12)
    assert dc.pval_Q_additive == pytest.approx(0.99039249172529853, abs=1e-12)
    assert dc.Q_standard == pytest.approx(0.29092411351685837, abs=1e-12)
    assert dc.df_Q_standard == pytest.approx(4.0, abs=1e-12)
    assert dc.Q_diff == pytest.approx(0.0, abs=1e-12)
    assert dc.df_Q_diff == pytest.approx(0.0, abs=1e-12)
    assert dc.tau == pytest.approx(0.0, abs=1e-12)
    assert dc.I2 == pytest.approx(0.0, abs=1e-12)

    # discomb leaves the standard-NMA estimates undefined.
    assert np.isnan(dc.TE_nma_common).all()


def test_discomb_disconnected_na_unident():
    # netmeta::discomb on a DISCONNECTED network: sub-net {A, A+B, B} plus an
    # isolated {C, D} edge. C and D are not uniquely identifiable, so their
    # component and combination effects are NA (R 4.6.1, netmeta 3.7-0).
    import numpy as np
    t1 = ["A", "A", "B", "C"]
    t2 = ["B", "A+B", "A+B", "D"]
    te = [0.1, -0.3, -0.5, 0.2]
    se = [0.2, 0.25, 0.3, 0.25]
    st = ["s1", "s2", "s3", "s4"]

    dc = discomb(te, se, t1, t2, st, sm="MD")

    assert dc.trts == ["A", "A+B", "B", "C", "D"]
    assert dc.comps == ["A", "B", "C", "D"]
    assert dc.k == 4 and dc.m == 4 and dc.n == 5 and dc.d == 4 and dc.c == 4

    # Identifiable components A, B keep their values; C, D are NA.
    assert dc.Comp_common[:2] == pytest.approx(
        [0.45324675324675323, 0.33246753246753236], abs=1e-12)
    assert np.isnan(dc.Comp_common[2:]).all()
    assert dc.Comb_common[:3] == pytest.approx(
        [0.45324675324675323, 0.78571428571428559, 0.33246753246753236],
        abs=1e-12)
    assert np.isnan(dc.Comb_common[3:]).all()

    # Cross-subnet contrasts keep their minimum-norm additive estimates.
    assert dc.TE_common[0, 3] == pytest.approx(0.35324675324675325, abs=1e-12)
    assert dc.TE_common[0, 4] == pytest.approx(0.55324675324675321, abs=1e-12)
    assert dc.TE_common[3, 4] == pytest.approx(0.19999999999999996, abs=1e-12)

    # df.Q.additive = n.a - k - rank(X) = 8 - 4 - 3 = 1.
    assert dc.Q_additive == pytest.approx(0.051948051948051951, abs=1e-12)
    assert dc.df_Q_additive == pytest.approx(1.0, abs=1e-12)
    assert dc.pval_Q_additive == pytest.approx(0.81970753782978834, abs=1e-12)

    # Standard NMA Q is undefined for a disconnected network.
    assert np.isnan(dc.Q_standard)
    assert np.isnan(dc.Q_diff)
    assert np.isnan(dc.df_Q_diff)
    assert dc.tau == pytest.approx(0.0, abs=1e-12)


def test_netmetabin_mh():
    # netmeta::netmetabin(method="MH", sm="OR") on a 3-treatment binary network
    # with 6 two-arm studies (R 4.6.1, netmeta 3.7-0). Reference values below
    # are log OR (analysis scale). Stijnen et al. (2010) Mantel-Haenszel NMA.
    import numpy as np
    event1 = [10, 8, 20, 15, 12, 10]
    n1 = [100, 90, 200, 150, 120, 100]
    event2 = [15, 12, 30, 25, 30, 25]
    n2 = [100, 95, 180, 160, 180, 160]
    t1 = ["A", "A", "A", "A", "B", "B"]
    t2 = ["B", "B", "C", "C", "C", "C"]
    st = ["s1", "s2", "s3", "s4", "s5", "s6"]

    r = netmetabin(event1, n1, event2, n2, t1, t2, st, sm="OR", method="MH")

    assert r.trts == ["A", "B", "C"]
    assert r.k == 6 and r.m == 6 and r.n == 3 and r.d == 3
    assert r.designs == ["A:B", "A:C", "B:C"]

    # Common effects (log OR).
    assert r.TE_common[0, 1] == pytest.approx(-0.23607776315012882, abs=1e-12)
    assert r.TE_common[0, 2] == pytest.approx(-0.65379429665442546, abs=1e-12)
    assert r.TE_common[1, 2] == pytest.approx(-0.41771653350429661, abs=1e-12)
    assert r.seTE_common[0, 1] == pytest.approx(0.23883563723778642, abs=1e-12)
    assert r.seTE_common[0, 2] == pytest.approx(0.20264647687638793, abs=1e-12)
    assert r.seTE_common[1, 2] == pytest.approx(0.22259419112705031, abs=1e-12)

    assert r.lower_common[0, 1] == pytest.approx(-0.704187010360863508, abs=1e-12)
    assert r.upper_common[0, 1] == pytest.approx(0.23203148406060584, abs=1e-12)
    assert r.statistic_common[0, 1] == pytest.approx(-0.98845283677279772, abs=1e-12)
    assert r.pval_common[0, 1] == pytest.approx(0.322930921694048245, abs=1e-12)
    assert r.pval_common[0, 2] == pytest.approx(0.0012541055752289677, abs=1e-12)
    assert r.pval_common[1, 2] == pytest.approx(0.0605752330769287126, abs=1e-12)

    # Direct per-comparison Mantel-Haenszel estimates.
    assert r.TE_direct_common[0, 1] == pytest.approx(-0.43149805421782161, abs=1e-12)
    assert r.TE_direct_common[0, 2] == pytest.approx(-0.55367204612471399, abs=1e-12)
    assert r.TE_direct_common[1, 2] == pytest.approx(-0.55281806706437220, abs=1e-12)
    assert r.seTE_direct_common[0, 1] == pytest.approx(0.32316011316491822, abs=1e-12)
    assert r.Q_direct[0, 1] == pytest.approx(0.011370456291726968, abs=1e-12)
    assert r.Q_direct[0, 2] == pytest.approx(0.027290281574135523, abs=1e-12)
    assert r.Q_direct[1, 2] == pytest.approx(0.020346189570538697, abs=1e-12)
    assert r.tau2_direct[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert r.I2_direct[0, 1] == pytest.approx(0.0, abs=1e-12)

    # Stijnen inconsistency Q.
    assert r.Q_inconsistency == pytest.approx(0.80584706640640991, abs=1e-12)
    assert r.df_Q_inconsistency == pytest.approx(1.0, abs=1e-12)
    assert r.pval_Q_inconsistency == pytest.approx(0.36935092231716449, abs=1e-12)
    assert np.isnan(r.Q) and np.isnan(r.tau) and np.isnan(r.I2)

    # Structure matrices.
    assert r.A_matrix.tolist() == [[0, 2, 2], [2, 0, 2], [2, 2, 0]]
    assert r.H_matrix_common.tolist() == [[1, 0], [0, 1], [-1, 1]]

    # Raw per-comparison log OR (metabin method="Inverse").
    assert r.TE == pytest.approx([
        -0.46262352194811296, -0.39334374757581958, -0.58778666490211895,
        -0.51082562376599072, -0.58778666490211895, -0.51082562376599072],
        abs=1e-12)


def test_netmetabin_inverse():
    # netmeta::netmetabin(method="Inverse", sm="OR") delegates to netmeta on
    # per-comparison log OR (R 4.6.1, netmeta 3.7-0).
    event1 = [10, 8, 20, 15, 12, 10]
    n1 = [100, 90, 200, 150, 120, 100]
    event2 = [15, 12, 30, 25, 30, 25]
    n2 = [100, 95, 180, 160, 180, 160]
    t1 = ["A", "A", "A", "A", "B", "B"]
    t2 = ["B", "B", "C", "C", "C", "C"]
    st = ["s1", "s2", "s3", "s4", "s5", "s6"]

    r = netmetabin(event1, n1, event2, n2, t1, t2, st, sm="OR", method="Inverse")

    assert r.trts == ["A", "B", "C"]
    assert r.k == 6 and r.m == 6 and r.n == 3
    assert r.TE_common[0, 1] == pytest.approx(-0.23621464052565061, abs=1e-12)
    assert r.TE_common[0, 2] == pytest.approx(-0.65392531402808451, abs=1e-12)
    assert r.TE_common[1, 2] == pytest.approx(-0.41771067350243390, abs=1e-12)
    assert r.seTE_common[0, 1] == pytest.approx(0.23882826275233604, abs=1e-12)
    assert r.tau == pytest.approx(0.0, abs=1e-12)
    assert r.I2 == pytest.approx(0.0, abs=1e-12)
    assert r.Q == pytest.approx(0.86379745958743759, abs=1e-12)
    assert r.df_Q == pytest.approx(4.0, abs=1e-12)


# netmeta::netconnection / netmeta::netdistance (Senn 2013, connected network)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SENN_CSV.exists(), reason="Senn 2013 benchmark CSV not present")
def test_netconnection_senn():
    import csv

    t1, t2, stud = [], [], []
    with SENN_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            t1.append(row["treat1"])
            t2.append(row["treat2"])
            stud.append(row["studlab"])

    nc = netconnection(t1, t2, stud)
    assert nc.k == 26
    assert nc.m == 28
    assert nc.n == 10
    assert nc.n_subnets == 1
    assert nc.d == 15
    assert nc.seq == [
        "acar", "benf", "metf", "migl", "piog",
        "plac", "rosi", "sulf", "sita", "vild",
    ]

    # Distance matrix (row acar): R netconnection(..., )$D.matrix
    assert nc.D_matrix[0].tolist() == pytest.approx(
        [0, 2, 1, 2, 2, 1, 2, 1, 2, 2])
    # Adjacency matrix (row plac): number of direct comparisons per treatment
    assert nc.A_matrix[5].tolist() == [2, 2, 4, 3, 1, 0, 6, 0, 1, 1]
    assert set(nc.subnet) == {1}
    assert len(nc.designs) == 15
    assert "acar:metf:plac" in nc.designs


def test_netconnection_disconnected():
    # Two disconnected components {A,B,C} and {X,Y} — R netmeta::netconnection.
    nc = netconnection(["A", "A", "B", "X"], ["B", "C", "C", "Y"],
                       studlab=["s1", "s2", "s3", "s4"])
    assert nc.k == 4
    assert nc.m == 4
    assert nc.n == 5
    assert nc.n_subnets == 2
    assert nc.d == 4
    assert nc.seq == ["A", "B", "C", "X", "Y"]
    assert nc.subnet == [1, 1, 1, 2]
    assert nc.comparisons == ["A:B", "A:C", "B:C", "X:Y"]
    assert nc.subnet_comparisons == [1, 1, 1, 2]
    # D matrix: within-component distances 1, across-component inf.
    import numpy as np

    assert nc.D_matrix[0, 3] == np.inf
    assert nc.D_matrix[0, 2] == 1.0
    assert nc.D_matrix[3, 4] == 1.0


def test_netdistance_order():
    # R hclust(dist(D2))$order for the {A,B,C}/{X,Y} network is C,A,B,X,Y.
    import numpy as np

    a = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=float)
    nd = netdistance(a)
    assert nd.order == [2, 0, 1, 3, 4]
    assert nd.D[0, 1] == 1.0
    assert nd.D[0, 3] == np.inf


# netmeta::netcontrib (shortestpath) — skipped if reference CSVs are absent
SENN_NC_COMMON = DATA_DIR / "_nc_common_ref.csv"
SENN_NC_RANDOM = DATA_DIR / "_nc_random_ref.csv"


def _load_nc_ref(path):
    """Load an R netcontrib reference CSV into (directs, {comparison: row})."""
    import csv

    with path.open(newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        rows = {}
        for row in rd:
            rows[row[0]] = [float(x) for x in row[1:]]
    return header[1:], rows


@pytest.mark.skipif(
    not (SENN_CSV.exists() and SENN_NC_COMMON.exists() and SENN_NC_RANDOM.exists()),
    reason="Senn 2013 netcontrib reference CSVs not present",
)
def test_netcontrib_senn():
    import csv

    te, sete, t1, t2, stud = [], [], [], [], []
    with SENN_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            te.append(float(row["TE"]))
            sete.append(float(row["seTE"]))
            t1.append(row["treat1"])
            t2.append(row["treat2"])
            stud.append(row["studlab"])

    r = netmeta(te, sete, t1, t2, stud, sm="MD", reference_group="plac")
    nc = netcontrib(r)

    directs_c, ref_common = _load_nc_ref(SENN_NC_COMMON)
    directs_r, ref_random = _load_nc_ref(SENN_NC_RANDOM)

    assert nc.directs == directs_c == directs_r
    assert nc.comparisons == list(ref_common)

    max_c = max_r = 0.0
    for ri, comp in enumerate(nc.comparisons):
        for ci in range(len(nc.directs)):
            max_c = max(max_c, abs(nc.common[ri, ci] - ref_common[comp][ci]))
            max_r = max(max_r, abs(nc.random[ri, ci] - ref_random[comp][ci]))
    assert max_c < 1e-9
    assert max_r < 1e-9
    # each row sums to 1
    import numpy as np

    assert np.allclose(nc.common.sum(axis=1), 1.0)
    assert np.allclose(nc.random.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# confint.tau2 (metafor::confint.rma.uni)
# ---------------------------------------------------------------------------

# dat.bcg on the log-RR analysis scale (escalc measure="RR"), with the REML
# tau^2 estimate fixed at the R value.
BCG_YI = [
    -0.889311333920205, -1.585388657201431, -1.348073148299693,
    -1.441551190021305, -0.217547322211296, -0.786115585818864,
    -1.620898223598392, 0.011952333523841, -0.469417648738149,
    -1.371344803472784, -0.339358828338391, 0.445913400571379,
    -0.017313948216880,
]
BCG_VI = [
    0.325584765003961, 0.194581121398144, 0.415367965367965,
    0.020010031902248, 0.051210172169631, 0.006905618455909,
    0.223017247572315, 0.003961579297818, 0.056434210463249,
    0.073024793613029, 0.012412213971560, 0.532505845200153,
    0.071404659683986,
]
BCG_TAU2 = 0.313243325980895


def test_confint_tau2_qp():
    r = confint_tau2(BCG_YI, BCG_VI, BCG_TAU2, method="REML", level=0.05, type="QP")
    assert r.tau2 == pytest.approx(BCG_TAU2, abs=1e-12)
    assert r.tau2_lb == pytest.approx(0.119695312816638, abs=1e-9)
    assert r.tau2_ub == pytest.approx(1.111486262215440, abs=1e-9)
    assert r.tau_lb == pytest.approx(0.345970103934773, abs=1e-9)
    assert r.tau_ub == pytest.approx(1.054270488164892, abs=1e-9)
    assert r.i2 == pytest.approx(92.221386074965551, abs=1e-9)
    assert r.h2 == pytest.approx(12.855760803112140, abs=1e-9)
    assert r.lb_conv and r.ub_conv


def test_confint_tau2_pl():
    r = confint_tau2(BCG_YI, BCG_VI, BCG_TAU2, method="REML", level=0.05, type="PL")
    assert r.tau2_lb == pytest.approx(0.115239399672938, abs=1e-9)
    assert r.tau2_ub == pytest.approx(0.964666425362115, abs=1e-9)
    assert r.tau_lb == pytest.approx(0.339469291207523, abs=1e-9)
    assert r.tau_ub == pytest.approx(0.982174335524053, abs=1e-9)
    assert r.lb_conv and r.ub_conv


def test_confint_tau2_ht():
    r = confint_tau2(BCG_YI, BCG_VI, BCG_TAU2, method="REML", level=0.05, type="HT")
    assert r.tau2_lb == pytest.approx(0.202742154857789, abs=1e-9)
    assert r.tau2_ub == pytest.approx(0.477027489415974, abs=1e-9)
    assert r.tau_lb == pytest.approx(0.450268980563606, abs=1e-9)
    assert r.tau_ub == pytest.approx(0.690671766772013, abs=1e-9)
    assert r.lb_conv and r.ub_conv


# ---------------------------------------------------------------------------
# predict (metafor::predict.rma.uni)
# ---------------------------------------------------------------------------

BCG_PRED = -0.714532348365102
BCG_PRED_SE = 0.179781531780361


def test_predict_rma_default():
    r = predict_rma(BCG_PRED, BCG_PRED_SE, BCG_TAU2, 13, predtype="default", level=0.05)
    assert r.pred == pytest.approx(BCG_PRED, abs=1e-12)
    assert r.se == pytest.approx(BCG_PRED_SE, abs=1e-12)
    assert r.ci_lb == pytest.approx(-1.066897675740053, abs=1e-9)
    assert r.ci_ub == pytest.approx(-0.362167020990150, abs=1e-9)
    assert r.pi_lb == pytest.approx(-1.866692346607069, abs=1e-9)
    assert r.pi_ub == pytest.approx(0.437627649876866, abs=1e-9)
    assert r.pi_dist == "norm"


def test_predict_rma_predtypes():
    # Riley et al. 2011 (df = k - parms = 11) and t (df = k - p = 12) use a
    # t reference; "simple" drops se^2 from pi.se.
    ref = {
        "riley": (-2.008376050690519, 0.579311353960316, "t"),
        "t": (-1.995342100891281, 0.566277404161078, "t"),
        "simple": (-1.811487834067632, 0.382423137337428, "norm"),
    }
    for pt, (lb, ub, dist) in ref.items():
        r = predict_rma(BCG_PRED, BCG_PRED_SE, BCG_TAU2, 13, predtype=pt, level=0.05)
        assert r.pi_lb == pytest.approx(lb, abs=1e-9), pt
        assert r.pi_ub == pytest.approx(ub, abs=1e-9), pt
        assert r.pi_dist == dist


# ---------------------------------------------------------------------------
# hc (metafor::hc.rma.uni — Hartung-Knapp / Sidik-Jonkman exact)
# ---------------------------------------------------------------------------


def test_hc_rma():
    r = hc_rma(BCG_YI, BCG_VI, level=0.05, tau2_rma=BCG_TAU2)
    assert r.beta == pytest.approx(-0.430285163654091, abs=1e-9)
    assert r.se == pytest.approx(0.283544351355250, abs=1e-9)
    assert r.ci_lb == pytest.approx(-1.155511867500363, abs=1e-9)
    assert r.ci_ub == pytest.approx(0.294941540192181, abs=1e-9)
    assert r.tau2 == pytest.approx(0.308760262862376, abs=1e-9)


# ---------------------------------------------------------------------------
# rma.peto / rma.mh (metafor::rma.peto / rma.mh, dat.bcg)
# ---------------------------------------------------------------------------


def test_rma_peto():
    r = rma_peto(T_POS, T_NEG, C_POS, C_NEG)
    assert r.beta == pytest.approx(-0.474446310940636, abs=1e-9)
    assert r.se == pytest.approx(0.040659080744648, abs=1e-9)
    assert r.zval == pytest.approx(-11.668889267819587, abs=1e-9)
    assert r.ci_lb == pytest.approx(-0.554136644844653, abs=1e-9)
    assert r.ci_ub == pytest.approx(-0.394755977036620, abs=1e-9)
    assert r.tau2 == pytest.approx(0.0, abs=1e-12)
    assert r.I2 == pytest.approx(92.845651931291371, abs=1e-9)
    assert r.H2 == pytest.approx(13.977513959290784, abs=1e-9)
    assert r.QE == pytest.approx(167.730167511489412, abs=1e-9)
    assert r.k_pos == 13


def test_rma_mh_or():
    r = rma_mh(T_POS, T_NEG, C_POS, C_NEG, measure="OR")
    assert r.beta == pytest.approx(-0.473410988388585, abs=1e-9)
    assert r.se == pytest.approx(0.041007778404231, abs=1e-9)
    assert r.zval == pytest.approx(-11.544419298260156, abs=1e-9)
    assert r.ci_lb == pytest.approx(-0.553784757146878, abs=1e-9)
    assert r.ci_ub == pytest.approx(-0.393037219630292, abs=1e-9)
    assert r.I2 == pytest.approx(92.680364044817708, abs=1e-9)
    assert r.H2 == pytest.approx(13.661881630766096, abs=1e-9)
    assert r.QE == pytest.approx(163.942579569193157, abs=1e-9)
    assert r.CO == pytest.approx(135.708714855120490, abs=1e-9)
    assert r.MH == pytest.approx(135.688943723906277, abs=1e-9)
    assert r.BD == pytest.approx(171.765366579396243, abs=1e-9)
    assert r.TA == pytest.approx(171.756672993871462, abs=1e-9)
    assert r.k_pos == 13


def test_rma_mh_rr():
    r = rma_mh(T_POS, T_NEG, C_POS, C_NEG, measure="RR")
    assert r.beta == pytest.approx(-0.453709604683754, abs=1e-9)
    assert r.se == pytest.approx(0.039337366871294, abs=1e-9)
    assert r.zval == pytest.approx(-11.533807185626319, abs=1e-9)
    assert r.ci_lb == pytest.approx(-0.530809426998130, abs=1e-9)
    assert r.ci_ub == pytest.approx(-0.376609782369378, abs=1e-9)
    assert r.I2 == pytest.approx(92.134631697306418, abs=1e-9)
    assert r.H2 == pytest.approx(12.713962798888106, abs=1e-9)
    assert r.QE == pytest.approx(152.567553586657283, abs=1e-9)
    assert r.CO is None and r.BD is None


def test_rma_mh_rd():
    r = rma_mh(T_POS, T_NEG, C_POS, C_NEG, measure="RD")
    assert r.beta == pytest.approx(-0.003288181832613, abs=1e-9)
    assert r.se == pytest.approx(0.000286656748527, abs=1e-9)
    assert r.zval == pytest.approx(-11.470798610202621, abs=1e-9)
    assert r.ci_lb == pytest.approx(-0.003850018735652, abs=1e-9)
    assert r.ci_ub == pytest.approx(-0.002726344929574, abs=1e-9)
    assert r.QE == pytest.approx(386.775941836559809, abs=1e-9)
    assert r.I2 == pytest.approx(96.897428536268464, abs=1e-9)
    assert r.H2 == pytest.approx(32.231328486379986, abs=1e-9)


# ---------------------------------------------------------------------------
# rma.mv + robust (metafor::rma.mv / robust, dat.konstantopoulos2011)
# ---------------------------------------------------------------------------

KONST_CSV = DATA_DIR / "_konstantopoulos2011.csv"


def _konstantopoulos2011():
    import csv

    district, school, yi, vi = [], [], [], []
    with KONST_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            district.append(int(row["district"]))
            school.append(int(row["school"]))
            yi.append(float(row["yi"]))
            vi.append(float(row["vi"]))
    return district, school, yi, vi


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_rma_mv_multilevel_reml():
    district, school, yi, vi = _konstantopoulos2011()
    r = rma_mv(yi, vi=vi, data={"district": district, "school": school},
               random="~ 1 | district/school", method="REML")

    assert r.sigma2[0] == pytest.approx(0.065061944310798620, abs=1e-6)
    assert r.sigma2[1] == pytest.approx(0.032736517032037331, abs=1e-6)
    assert r.beta[0] == pytest.approx(0.18471316367285326, abs=1e-6)
    assert r.se[0] == pytest.approx(0.084555918761870158, abs=1e-6)
    assert r.zval[0] == pytest.approx(2.1845089779350637, abs=1e-5)
    assert r.pval[0] == pytest.approx(0.028924868295301054, abs=1e-6)
    assert r.ci_lb[0] == pytest.approx(0.01898660821989312, abs=1e-6)
    assert r.ci_ub[0] == pytest.approx(0.3504397191258134, abs=1e-6)
    assert r.qe == pytest.approx(578.86401795822349, abs=1e-6)
    assert r.qe_p == pytest.approx(1.7968303019346829e-88, abs=1e-90)
    assert r.k == 56 and r.p == 1


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_rma_mv_single_level_reml():
    district, school, yi, vi = _konstantopoulos2011()
    r = rma_mv(yi, vi=vi, data={"school": school}, random="~ 1 | school", method="REML")

    assert r.sigma2[0] == pytest.approx(0.0064628988894651776, abs=1e-6)
    assert r.beta[0] == pytest.approx(0.023057127294058153, abs=1e-6)
    assert r.se[0] == pytest.approx(0.027365924483360785, abs=1e-6)


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_rma_mv_multilevel_ml():
    district, school, yi, vi = _konstantopoulos2011()
    r = rma_mv(yi, vi=vi, data={"district": district, "school": school},
               random="~ 1 | district/school", method="ML")

    assert r.sigma2[0] == pytest.approx(0.057738358914493193, abs=1e-6)
    assert r.sigma2[1] == pytest.approx(0.032864791870188537, abs=1e-6)
    assert r.beta[0] == pytest.approx(0.18445538396558636, abs=1e-6)
    assert r.se[0] == pytest.approx(0.080481682198788193, abs=1e-6)


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_robust_cr1_default():
    district, school, yi, vi = _konstantopoulos2011()
    r = rma_mv(yi, vi=vi, data={"district": district, "school": school},
               random="~ 1 | district/school", method="REML")
    rob = robust(r, cluster=district)

    assert rob["vbest"] == "CR1"
    assert rob["se"][0] == pytest.approx(0.084502217152394019, abs=1e-6)
    assert rob["zval"][0] == pytest.approx(2.1858972450360161, abs=1e-5)
    assert rob["pval"][0] == pytest.approx(0.053707313636821144, abs=1e-6)
    assert rob["ci_lb"][0] == pytest.approx(-0.003569509443376806, abs=1e-6)
    assert rob["ci_ub"][0] == pytest.approx(0.37299583678908332, abs=1e-6)
    assert rob["dfs"] == pytest.approx(10.0, abs=1e-9)


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_robust_cr2_satterthwaite():
    district, school, yi, vi = _konstantopoulos2011()
    r = rma_mv(yi, vi=vi, data={"district": district, "school": school},
               random="~ 1 | district/school", method="REML")
    rob = robust(r, cluster=district, clubSandwich=True, vcov="CR2",
                 coef_test="Satterthwaite")

    assert rob["vbest"] == "CR2"
    assert rob["se"][0] == pytest.approx(0.0844579954347457, abs=1e-6)
    assert rob["zval"][0] == pytest.approx(2.1870417681836543, abs=1e-5)
    assert rob["dfs"][0] == pytest.approx(9.863156, abs=1e-4)
    assert rob["pval"][0] == pytest.approx(0.0539667424703544, abs=1e-6)
    assert rob["ci_lb"][0] == pytest.approx(-0.0038254835164811, abs=1e-6)
    assert rob["ci_ub"][0] == pytest.approx(0.3732518108621876, abs=1e-6)
    assert rob["vb"][0, 0] == pytest.approx(0.007133153, abs=1e-7)


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_robust_cr2_naive_t():
    district, school, yi, vi = _konstantopoulos2011()
    r = rma_mv(yi, vi=vi, data={"district": district, "school": school},
               random="~ 1 | district/school", method="REML")
    rob = robust(r, cluster=district, clubSandwich=True, vcov="CR2",
                 coef_test="naive-t")

    assert rob["dfs"][0] == pytest.approx(10.0, abs=1e-9)
    assert rob["pval"][0] == pytest.approx(0.0536034625691189, abs=1e-6)


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_robust_cr3_satterthwaite():
    district, school, yi, vi = _konstantopoulos2011()
    r = rma_mv(yi, vi=vi, data={"district": district, "school": school},
               random="~ 1 | district/school", method="REML")
    rob = robust(r, cluster=district, clubSandwich=True, vcov="CR3",
                 coef_test="Satterthwaite")

    assert rob["vbest"] == "CR3"
    assert rob["se"][0] == pytest.approx(0.0886668488224245, abs=1e-6)
    assert rob["zval"][0] == pytest.approx(2.0832268894858705, abs=1e-5)
    assert rob["dfs"][0] == pytest.approx(9.84, abs=1e-3)
    assert rob["pval"][0] == pytest.approx(0.0642962126978074, abs=1e-6)


# ---------------------------------------------------------------------------
# rma.mv `~ inner | outer` terms: compound symmetry (CS) and random slopes
# (GEN / unstructured).  Reference values from metafor::rma.mv (5.2-1).
# ---------------------------------------------------------------------------

def _konstantopoulos2011_full():
    """Return district, school, year, yi, vi for dat.konstantopoulos2011."""
    import csv

    district, school, year, yi, vi = [], [], [], [], []
    with KONST_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            district.append(int(row["district"]))
            school.append(int(row["school"]))
            year.append(int(row["year"]))
            yi.append(float(row["yi"]))
            vi.append(float(row["vi"]))
    return district, school, year, yi, vi


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_rma_mv_cs_year_district():
    # metafor: rma.mv(yi, vi, random = ~ year | district, data=dat)
    district, school, year, yi, vi = _konstantopoulos2011_full()
    r = rma_mv(yi, vi=vi, data={"district": district, "year": year},
               random="~ year | district", method="REML")

    assert r.tau2[0] == pytest.approx(0.0828108325305787, abs=1e-7)
    assert r.rho[0] == pytest.approx(-0.9999999942140179, abs=1e-6)
    assert r.beta[0] == pytest.approx(0.1928224491138022, abs=1e-7)
    assert r.se[0] == pytest.approx(0.0774615401123044, abs=1e-7)
    assert r.zval[0] == pytest.approx(2.4892669166433627, abs=1e-5)
    assert r.pval[0] == pytest.approx(0.0128006822672409, abs=1e-6)


# ---------------------------------------------------------------------------
# rma.mv struct="ID"/"DIAG"/"HCS"/"AR"/"HAR"/"CAR"/"UN" for ~ year | district.
# Reference values from metafor::rma.mv (5.2-1), REML.
# ---------------------------------------------------------------------------

# tau2 for the diagonal / (heteroscedastic) structures: the inner years are
# sorted [1976, 1989, 1994, 1995, 1997, 2000]; years 1989 (idx 1) and 1995
# (idx 3) have k=1 so their tau2 is fixed to 0.
_T2_DIAG = [0.0740555842408778, 0, 0.0303383815062765, 0, 0.1982947631071745, 0.0128716193634324]


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_rma_mv_struct_id():
    district, school, year, yi, vi = _konstantopoulos2011_full()
    r = rma_mv(yi, vi=vi, data={"district": district, "year": year},
               random="~ year | district", struct="ID", method="REML")
    assert r.tau2[0] == pytest.approx(0.0812805463707763, abs=1e-6)
    assert r.rho[0] == pytest.approx(0.0, abs=1e-9)
    assert r.beta[0] == pytest.approx(0.196239300040954, abs=1e-6)
    assert r.se[0] == pytest.approx(0.0868631945659352, abs=1e-6)
    assert r.parms == 2


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_rma_mv_struct_diag():
    district, school, year, yi, vi = _konstantopoulos2011_full()
    r = rma_mv(yi, vi=vi, data={"district": district, "year": year},
               random="~ year | district", struct="DIAG", method="REML")
    for got, want in zip(r.tau2, _T2_DIAG):
        assert got == pytest.approx(want, abs=1e-5)
    assert r.rho[0] == pytest.approx(0.0, abs=1e-9)
    assert r.beta[0] == pytest.approx(0.126994363414686, abs=1e-6)
    assert r.se[0] == pytest.approx(0.0452002999495505, abs=1e-6)
    assert r.parms == 5


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_rma_mv_struct_hcs():
    district, school, year, yi, vi = _konstantopoulos2011_full()
    r = rma_mv(yi, vi=vi, data={"district": district, "year": year},
               random="~ year | district", struct="HCS", method="REML")
    for got, want in zip(r.tau2, _T2_DIAG):
        assert got == pytest.approx(want, abs=1e-5)
    assert r.rho[0] == pytest.approx(0.5, abs=1e-6)
    assert r.beta[0] == pytest.approx(0.126994363414686, abs=1e-6)
    assert r.se[0] == pytest.approx(0.0452002999495505, abs=1e-6)
    assert r.parms == 6


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_rma_mv_struct_ar():
    district, school, year, yi, vi = _konstantopoulos2011_full()
    r = rma_mv(yi, vi=vi, data={"district": district, "year": year},
               random="~ year | district", struct="AR", method="REML")
    assert r.tau2[0] == pytest.approx(0.0828108324866266, abs=1e-7)
    assert r.rho[0] == pytest.approx(-0.999999994213991, abs=1e-6)
    assert r.beta[0] == pytest.approx(0.192822449117012, abs=1e-7)
    assert r.se[0] == pytest.approx(0.0774615400989977, abs=1e-7)
    assert r.parms == 3


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_rma_mv_struct_har():
    district, school, year, yi, vi = _konstantopoulos2011_full()
    r = rma_mv(yi, vi=vi, data={"district": district, "year": year},
               random="~ year | district", struct="HAR", method="REML")
    for got, want in zip(r.tau2, _T2_DIAG):
        assert got == pytest.approx(want, abs=1e-5)
    assert r.rho[0] == pytest.approx(0.5, abs=1e-6)
    assert r.beta[0] == pytest.approx(0.126994363414686, abs=1e-6)
    assert r.se[0] == pytest.approx(0.0452002999495505, abs=1e-6)
    assert r.parms == 6


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_rma_mv_struct_car():
    # CAR correlation is exp(-rho*|year_i - year_j|); rho is essentially 0
    # (reduces to ID), so only check it is tiny rather than exact.
    district, school, year, yi, vi = _konstantopoulos2011_full()
    r = rma_mv(yi, vi=vi, data={"district": district, "year": year},
               random="~ year | district", struct="CAR", method="REML")
    assert r.tau2[0] == pytest.approx(0.081280316473474, abs=1e-6)
    assert r.rho[0] == pytest.approx(0.0, abs=1e-4)
    assert r.beta[0] == pytest.approx(0.196239311554681, abs=1e-6)
    assert r.se[0] == pytest.approx(0.0868630823918689, abs=1e-6)
    assert r.parms == 3


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_rma_mv_struct_un():
    # only the (1994, 1976) inner pair co-occurs, so exactly one rho (lower
    # index 9) is free (== 0.5); all other 14 rho are fixed to 0.
    district, school, year, yi, vi = _konstantopoulos2011_full()
    r = rma_mv(yi, vi=vi, data={"district": district, "year": year},
               random="~ year | district", struct="UN", method="REML")
    t2_un = [0.0740554979649331, 0, 0.0303386094726117, 0, 0.1982958462908044, 0.0128715743138283]
    for got, want in zip(r.tau2, t2_un):
        assert got == pytest.approx(want, abs=1e-5)
    assert len(r.rho) == 15
    assert r.rho[9] == pytest.approx(0.500007500271117, abs=1e-4)
    assert sum(r.rho[:9] + r.rho[10:]) == pytest.approx(0.0, abs=1e-9)
    assert r.beta[0] == pytest.approx(0.126994300851989, abs=1e-6)
    assert r.se[0] == pytest.approx(0.04520029675127, abs=1e-6)
    assert r.parms == 6


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_rma_mv_gen_random_slopes_reml():
    # metafor: rma.mv(yi, vi, random = ~ year | district, struct="GEN", data=dat)
    # tau2[0] sits on a very flat ridge of the likelihood, so its absolute
    # tolerance is looser than beta/se (which are the medically relevant bits).
    district, school, year, yi, vi = _konstantopoulos2011_full()
    r = rma_mv(yi, vi=vi, data={"district": district, "year": year},
               random="~ year | district", struct="GEN", method="REML")

    assert r.tau2[0] == pytest.approx(3131.847, abs=0.01)
    assert r.tau2[1] == pytest.approx(0.0007942266, abs=1e-7)
    assert r.rho[0] == pytest.approx(-1.0, abs=1e-9)
    assert r.beta[0] == pytest.approx(0.1491238, abs=1e-6)
    assert r.se[0] == pytest.approx(0.07302746, abs=1e-7)
    assert r.zval[0] == pytest.approx(2.042024, abs=1e-5)
    assert r.pval[0] == pytest.approx(0.04114918, abs=1e-6)
    assert r.ci_lb[0] == pytest.approx(0.005992612, abs=1e-5)
    assert r.ci_ub[0] == pytest.approx(0.292255, abs=1e-5)


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_rma_mv_gen_random_slopes_ml():
    district, school, year, yi, vi = _konstantopoulos2011_full()
    r = rma_mv(yi, vi=vi, data={"district": district, "year": year},
               random="~ year | district", struct="GEN", method="ML")

    assert r.tau2[0] == pytest.approx(3009.838, abs=0.01)
    assert r.tau2[1] == pytest.approx(0.0007629405, abs=1e-7)
    assert r.rho[0] == pytest.approx(-1.0, abs=1e-9)
    assert r.beta[0] == pytest.approx(0.1461727, abs=1e-6)
    assert r.se[0] == pytest.approx(0.06828273, abs=1e-7)


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_rma_mv_gen_no_intercept():
    # metafor: rma.mv(yi, vi, random = ~ 0 + year | district, struct="GEN")
    district, school, year, yi, vi = _konstantopoulos2011_full()
    r = rma_mv(yi, vi=vi, data={"district": district, "year": year},
               random="~ 0 + year | district", struct="GEN", method="REML")

    assert r.tau2[0] == pytest.approx(2.088692e-08, abs=1e-9)
    assert r.beta[0] == pytest.approx(0.1957425, abs=1e-6)
    assert r.se[0] == pytest.approx(0.08995575, abs=1e-6)


# ---------------------------------------------------------------------------
# rma.uni (metafor::rma, dat.bcg on the lnRR scale)
# ---------------------------------------------------------------------------

BCG_ABLAT = [44, 55, 42, 52, 13, 44, 19, 13, 27, 42, 18, 33, 33]
BCG_STUDLAB = [
    "Aronson 1948", "Ferguson & Simes 1949", "Rosenthal et al 1960",
    "Hart & Sutherland 1977", "Frimodt-Moller et al 1973", "Stein & Aronson 1953",
    "Vandiviere et al 1973", "TPT Madras 1980", "Coetzee & Berjak 1968",
    "Rosenthal et al 1961", "Comstock et al 1974", "Comstock & Webster 1969",
    "Comstock et al 1976",
]
BCG_YEAR = [1948, 1949, 1960, 1977, 1973, 1953, 1973, 1980, 1968, 1961, 1974,
            1969, 1976]


def test_rma_uni_intercept_reml():
    r = rma(BCG_YI, vi=BCG_VI, method="REML")
    assert r.b[0] == pytest.approx(-0.714532348365102, abs=1e-9)
    assert r.se[0] == pytest.approx(0.179781531780361, abs=1e-9)
    assert r.zval[0] == pytest.approx(-3.974447994124577, abs=1e-9)
    assert r.pval[0] == pytest.approx(7.05426734852056e-05, abs=1e-10)
    assert r.tau2 == pytest.approx(0.313243325980895, abs=1e-9)
    assert r.QE == pytest.approx(152.233008082373, abs=1e-9)
    assert r.QM == pytest.approx(15.796236858001, abs=1e-9)
    assert r.QMdf == [1, None]
    assert r.I2 == pytest.approx(92.221386074966, abs=1e-9)
    assert r.H2 == pytest.approx(12.855760803112, abs=1e-9)
    assert r.k == 13 and r.p == 1 and r.int_only
    assert r.parms == 2
    assert r.dfs is None and r.ddf is None


def test_rma_uni_intercept_dl():
    r = rma(BCG_YI, vi=BCG_VI, method="DL")
    assert r.b[0] == pytest.approx(-0.714117222073227, abs=1e-9)
    assert r.tau2 == pytest.approx(0.308760262862376, abs=1e-9)
    assert r.I2 == pytest.approx(92.117346854562, abs=1e-9)
    assert r.H2 == pytest.approx(12.686084006864, abs=1e-9)


def test_rma_uni_intercept_fe():
    r = rma(BCG_YI, vi=BCG_VI, method="FE")
    assert r.b[0] == pytest.approx(-0.430285163654091, abs=1e-9)
    assert r.se[0] == pytest.approx(0.040498751710864, abs=1e-9)
    assert r.tau2 == pytest.approx(0.0, abs=1e-12)
    assert r.tau2_fix is True
    assert r.QE == pytest.approx(152.233008082373, abs=1e-9)
    # FE I^2/H^2 use the QE-based Cochran formula (equal to DL values here).
    assert r.I2 == pytest.approx(92.117346854562, abs=1e-9)
    assert r.H2 == pytest.approx(12.686084006864, abs=1e-9)


def test_rma_uni_intercept_knha():
    r = rma(BCG_YI, vi=BCG_VI, method="REML", test="knha")
    assert r.b[0] == pytest.approx(-0.714532348365102, abs=1e-9)
    assert r.se[0] == pytest.approx(0.180791745519100, abs=1e-9)
    assert r.pval[0] == pytest.approx(0.001920015085134, abs=1e-9)
    assert r.dfs == 12 and r.ddf == 12
    assert r.s2w > 1.0


def test_rma_uni_mods_reml():
    r = rma(BCG_YI, vi=BCG_VI, mods=BCG_ABLAT, method="REML")
    assert r.b[0] == pytest.approx(0.251464294374285, abs=1e-9)
    assert r.b[1] == pytest.approx(-0.029101660928015, abs=1e-9)
    assert r.tau2 == pytest.approx(0.076354694705660, abs=1e-9)
    assert r.QE == pytest.approx(30.733090010749, abs=1e-9)
    assert r.QM == pytest.approx(16.357125572159, abs=1e-9)
    assert r.QMdf == [1, None]
    assert r.I2 == pytest.approx(68.393130509048, abs=1e-9)
    assert r.H2 == pytest.approx(3.163869171815, abs=1e-9)
    assert r.R2 == pytest.approx(75.624478361491, abs=1e-9)
    assert r.p == 2 and not r.int_only
    assert r.X.shape == (13, 2)


def test_rma_uni_mods_dl():
    r = rma(BCG_YI, vi=BCG_VI, mods=BCG_ABLAT, method="DL")
    assert r.b[0] == pytest.approx(0.259543712434023, abs=1e-9)
    assert r.b[1] == pytest.approx(-0.029228738755188, abs=1e-9)
    assert r.tau2 == pytest.approx(0.063300502426415, abs=1e-9)
    assert r.QM == pytest.approx(18.845233542635, abs=1e-9)
    assert r.R2 == pytest.approx(79.498494450165, abs=1e-9)


def test_rma_uni_mods_fe():
    r = rma(BCG_YI, vi=BCG_VI, mods=BCG_ABLAT, method="FE")
    assert r.b[0] == pytest.approx(0.343564577446590, abs=1e-9)
    assert r.b[1] == pytest.approx(-0.029236934259542, abs=1e-9)
    assert r.QM == pytest.approx(121.499918071624, abs=1e-9)
    assert r.tau2 == pytest.approx(0.0, abs=1e-12)


def test_rma_uni_mods_knha():
    r = rma(BCG_YI, vi=BCG_VI, mods=BCG_ABLAT, method="REML", test="knha")
    assert r.se[0] == pytest.approx(0.283930037419131, abs=1e-9)
    assert r.se[1] == pytest.approx(0.008201541796008, abs=1e-9)
    assert r.QM == pytest.approx(12.590545526074, abs=1e-9)
    assert r.QMp == pytest.approx(0.004565545588957, abs=1e-9)
    assert r.pval[1] == pytest.approx(0.004565545588957, abs=1e-9)
    assert r.dfs == 11 and r.ddf == 11
    assert r.s2w > 1.0


def test_rma_uni_weights_and_fitted():
    r = rma(BCG_YI, vi=BCG_VI, method="REML")
    # weights.f (raw) vs percentage weights: percentages must sum to 100.
    assert sum(r.weights_pct) == pytest.approx(100.0, abs=1e-9)
    assert r.weights_pct[0] == pytest.approx(5.059483079315513, abs=1e-9)
    assert r.weights_pct[1] == pytest.approx(6.364679632126595, abs=1e-9)
    # fitted = X b (all equal the pooled estimate, intercept-only).
    assert r.fitted[0] == pytest.approx(r.b[0], abs=1e-12)
    assert r.residuals[0] == pytest.approx(BCG_YI[0] - r.b[0], abs=1e-12)


def test_rma_uni_to_dict():
    r = rma(BCG_YI, vi=BCG_VI, mods=BCG_ABLAT, method="REML")
    d = r.to_dict()
    assert d["tau2"] == pytest.approx(0.076354694705660, abs=1e-9)
    assert d["b"][1] == pytest.approx(-0.029101660928015, abs=1e-9)
    assert isinstance(d["vb"], list)


# ---------------------------------------------------------------------------
# rma.mv test="t"/"knha"/"adhoc" + dfs, fit.stats, and anova_rma LRT
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_rma_mv_test_t():
    # metafor: rma.mv(..., test="t")  -> t distribution, dfs="residual" (k-p)
    district, school, year, yi, vi = _konstantopoulos2011_full()
    r = rma_mv(yi, vi=vi, data={"district": district, "school": school},
               random="~ 1 | district/school", method="REML", test="t")

    assert r.beta[0] == pytest.approx(0.184713163672853, abs=1e-6)
    assert r.se[0] == pytest.approx(0.0845559187618702, abs=1e-7)
    assert r.zval[0] == pytest.approx(2.18450897793506, abs=1e-5)
    assert r.pval[0] == pytest.approx(0.0332050188345618, abs=1e-6)
    assert r.ci_lb[0] == pytest.approx(0.0152593157819066, abs=1e-6)
    assert r.ci_ub[0] == pytest.approx(0.3541670115638, abs=1e-6)
    assert r.ddf[0] == pytest.approx(55.0, abs=1e-9)


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_rma_mv_test_knha():
    # Knapp-Hartung: variance scaled by RSS/(k-p), t distribution.
    district, school, year, yi, vi = _konstantopoulos2011_full()
    r = rma_mv(yi, vi=vi, data={"district": district, "school": school},
               random="~ 1 | district/school", method="REML", test="knha")

    assert r.se[0] == pytest.approx(0.086931149115619, abs=1e-7)
    assert r.zval[0] == pytest.approx(2.12482137360434, abs=1e-5)
    assert r.pval[0] == pytest.approx(0.0381100077004185, abs=1e-6)
    assert r.ci_lb[0] == pytest.approx(0.0104992477823661, abs=1e-6)
    assert r.ci_ub[0] == pytest.approx(0.35892707956334, abs=1e-6)
    assert r.ddf[0] == pytest.approx(55.0, abs=1e-9)
    assert r.s2w == pytest.approx(1.0569704602286025, abs=1e-6)


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_rma_mv_test_adhoc():
    # s2w = max(RSS/(k-p), 1); here s2w > 1 so adhoc == knha.
    district, school, year, yi, vi = _konstantopoulos2011_full()
    r = rma_mv(yi, vi=vi, data={"district": district, "school": school},
               random="~ 1 | district/school", method="REML", test="adhoc")

    assert r.se[0] == pytest.approx(0.086931149115619, abs=1e-7)
    assert r.pval[0] == pytest.approx(0.0381100077004185, abs=1e-6)
    assert r.s2w == pytest.approx(1.0569704602286025, abs=1e-6)


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_rma_mv_test_t_contain():
    # dfs="contain" -> ddf = min(containing factor levels) - p = 11 - 1 = 10
    district, school, year, yi, vi = _konstantopoulos2011_full()
    r = rma_mv(yi, vi=vi, data={"district": district, "school": school},
               random="~ 1 | district/school", method="REML", test="t",
               dfs="contain")

    assert r.ddf[0] == pytest.approx(10.0, abs=1e-9)
    assert r.pval[0] == pytest.approx(0.05383354122866, abs=1e-6)
    assert r.ci_lb[0] == pytest.approx(-0.00368916408586478, abs=1e-6)
    assert r.ci_ub[0] == pytest.approx(0.373115491431571, abs=1e-6)


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_rma_mv_fit_stats():
    # metafor rma.mv $fit.stats (ML + REML columns)
    district, school, year, yi, vi = _konstantopoulos2011_full()
    r = rma_mv(yi, vi=vi, data={"district": district, "school": school},
               random="~ 1 | district/school", method="REML")

    fs = r.fit_stats
    assert r.parms == 3
    assert fs["ML"]["ll"] == pytest.approx(-8.419996, abs=1e-6)
    assert fs["ML"]["dev"] == pytest.approx(161.496506, abs=1e-6)
    assert fs["ML"]["AIC"] == pytest.approx(22.839992, abs=1e-6)
    assert fs["ML"]["BIC"] == pytest.approx(28.916047, abs=1e-6)
    assert fs["ML"]["AICc"] == pytest.approx(23.301531, abs=1e-6)
    assert fs["REML"]["ll"] == pytest.approx(-7.95872403371098, abs=1e-8)
    assert fs["REML"]["dev"] == pytest.approx(15.917448067422, abs=1e-8)
    assert fs["REML"]["AIC"] == pytest.approx(21.917448, abs=1e-6)
    assert fs["REML"]["BIC"] == pytest.approx(27.939448, abs=1e-6)
    assert fs["REML"]["AICc"] == pytest.approx(22.388036, abs=1e-6)


@pytest.mark.skipif(not KONST_CSV.exists(), reason="dat.konstantopoulos2011 CSV not present")
def test_anova_rma_lrt():
    # anova(full, reduced) likelihood-ratio test on the REML deviance
    district, school, year, yi, vi = _konstantopoulos2011_full()
    full = rma_mv(yi, vi=vi, data={"district": district, "school": school},
                  random="~ 1 | district/school", method="REML")
    reduced = rma_mv(yi, vi=vi, data={"district": district},
                     random="~ 1 | district", method="REML")

    a = anova_rma(full, reduced)
    assert a["df"] == 1
    assert a["LRT"] == pytest.approx(48.5155043045382, abs=1e-8)
    assert a["pval"] == pytest.approx(3.27687921744243e-12, abs=1e-14)
    assert a["full"]["df"] == 3
    assert a["reduced"]["df"] == 2
    assert a["full"]["logLik"] == pytest.approx(-7.95872403371098, abs=1e-8)
    assert a["reduced"]["logLik"] == pytest.approx(-32.2164761859801, abs=1e-8)


def test_rma_mv_ranef():
    # BLUPs for ~1 | district/school must match metafor::ranef.rma.mv
    district, school, year, yi, vi = _konstantopoulos2011_full()
    r = rma_mv(yi, vi=vi, data={"district": district, "school": school},
               random="~ 1 | district/school", method="REML")
    rf = ranef(r)

    assert list(rf.keys()) == ["district", "district/school"]

    d = rf["district"]
    assert d["labels"] == ["11", "12", "18", "27", "56", "58", "71", "86",
                           "91", "108", "644"]
    assert d["intrcpt"][:6] == pytest.approx(
        [-0.18998596, -0.08467077, 0.14072730, 0.24064814, -0.10729420, -0.23650899],
        abs=1e-6)
    assert d["se"][:6] == pytest.approx(
        [0.1665397, 0.1240789, 0.1372405, 0.1188590, 0.1189523, 0.1011285],
        abs=1e-6)
    assert d["pi_lb"][:2] == pytest.approx([-0.51639769, -0.32786096], abs=1e-6)
    assert d["pi_ub"][:2] == pytest.approx([0.13642578, 0.15851940], abs=1e-6)

    s = rf["district/school"]
    assert s["intrcpt"][:6] == pytest.approx(
        [-0.037946745, -0.046633829, 0.043579063, -0.054591674,
         0.020983756, -0.252191107], abs=1e-6)


def test_rma_mv_ranef_cs_gen():
    # BLUPs for ~ year | district (CS and GEN) must match metafor
    district, school, year, yi, vi = _konstantopoulos2011_full()
    dat = {"district": district, "school": school, "year": year}

    rcs = rma_mv(yi, vi=vi, data=dat, random="~ year | district", method="REML")
    rf = ranef(rcs)["~ year | district"]
    assert rf["labels"][:8] == ["1976 | 11", "1989 | 12", "1994 | 18",
                                "1976 | 27", "1997 | 56", "1976 | 58",
                                "1997 | 71", "1997 | 86"]
    assert rf["intrcpt"][:8] == pytest.approx(
        [-0.2288825, -0.1204897, 0.1457523, 0.3016115,
         -0.1392105, -0.2319697, 0.6616857, -0.2212385], abs=1e-6)
    assert rf["se"][:4] == pytest.approx(
        [0.16248403, 0.09623356, 0.10633537, 0.08494146], abs=1e-6)

    rgen = rma_mv(yi, vi=vi, data=dat, random="~ year | district",
                  method="REML", struct="GEN")
    rfgen = ranef(rgen)["~ year | district"]
    assert rfgen["labels"] == ["11", "12", "18", "27", "56", "58", "71", "86",
                               "91", "108", "644"]
    assert rfgen["var_names"] == ["intrcpt", "year"]
    assert rfgen["pred"][0, :] == pytest.approx([-39.128370, 0.019704422], abs=1e-4)
    assert rfgen["pred"][3, :] == pytest.approx([69.950459, -0.035225935], abs=1e-4)
    assert rfgen["pred"][10, :] == pytest.approx([-4.091332, 0.002060329], abs=1e-4)


# ---------------------------------------------------------------------------
# Plotting layer (forest / funnel / netgraph / baujat / gosh)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _plot_tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("plots")


def test_plot_forest_and_funnel(_plot_tmp):
    yi = [0.2, 0.4, 0.1, 0.5, 0.3]
    vi = [0.04, 0.09, 0.02, 0.16, 0.06]
    studlab = [f"S{i}" for i in range(1, 6)]
    pooled = ma.pool_yv(yi, vi, None, "MD", model="random", method="REML")

    fpath = str(_plot_tmp / "forest.png")
    ax = ma.forest(yi, vi, studlab=studlab, pooled=pooled, effect_type="MD",
                   filename=fpath)
    assert ax is not None
    assert Path(fpath).exists() and Path(fpath).stat().st_size > 0

    fupath = str(_plot_tmp / "funnel.png")
    ax = ma.funnel(yi, vi, effect_type="MD", filename=fupath)
    assert Path(fupath).exists() and Path(fupath).stat().st_size > 0


def test_plot_netgraph():
    # Senn 1993: treatments A/B/C with a star of direct comparisons
    te = [0.1, -0.2, 0.05, -0.15, 0.3, 0.25]
    se = [0.1] * 6
    t1 = ["A", "B", "C", "A", "B", "C"]
    t2 = ["B", "C", "A", "D", "D", "D"]
    nm = netmeta(treat1=t1, treat2=t2, TE=te, seTE=se,
                 studlab=[f"s{i}" for i in range(6)])
    ax = ma.netgraph(nm)
    assert ax is not None


def test_plot_baujat_and_gosh(_plot_tmp):
    yi = [0.2, 0.4, 0.1, 0.5, 0.3]
    vi = [0.04, 0.09, 0.02, 0.16, 0.06]

    bj = ma.baujat(yi, vi, method="REML")
    ax = ma.plot_baujat(bj)
    assert ax is not None

    gs = ma.gosh(yi, vi, method="REML")
    ax = ma.plot_gosh(gs, x="tau2", y="estimate")
    assert ax is not None




# ---------------------------------------------------------------------------
# metainf / metacum (meta package interfaces)
# ---------------------------------------------------------------------------

def test_metainf_random_dl():
    r = metainf(BCG_YI, [math.sqrt(v) for v in BCG_VI], studlab=BCG_STUDLAB,
                method_tau="DL", pooled="random")
    assert r.k == 13
    assert r.TE == pytest.approx([
        -0.70512519841745092, -0.6545158750879716, -0.68484174870459469,
        -0.61869687071527524, -0.76403570174150914, -0.71272733660621257,
        -0.65569394935532199, -0.79008659985370056, -0.73926335653631525,
        -0.65258520458253289, -0.76069417416119312, -0.76001281829968115,
        -0.77892374005707243], abs=1e-9)
    assert r.seTE == pytest.approx([
        0.18425006745165473, 0.18253833132073974, 0.1827639209338621,
        0.16407307222272868, 0.19141235027154849, 0.20537550603998503,
        0.18235266476785419, 0.16308463840032222, 0.19138004027666364,
        0.18325651070163995, 0.20627517574791629, 0.18238617575781949,
        0.18891870412607734], abs=1e-9)
    assert r.I2 == pytest.approx([
        0.92743229118856729, 0.92430373595907278, 0.92676287313310335,
        0.8860842544335209, 0.92730637489582046, 0.91418764169260081,
        0.92456951728113279, 0.83820153415175369, 0.92772907845203534,
        0.92133140071466946, 0.92737620334257131, 0.92704933402525502,
        0.92656308332064719], abs=1e-9)
    assert r.tau2 == pytest.approx([
        0.3121714990406787, 0.29997459082210964, 0.30852706980140093,
        0.2167335579458137, 0.32620888271247467, 0.38291902183230342,
        0.30057127632259173, 0.21088523257486985, 0.32660926031825083,
        0.29464487852775811, 0.38780919896360982, 0.30938044631726375,
        0.31770327992381431], abs=1e-9)
    assert r.statistic[0] == pytest.approx(-3.8270010327267223, abs=1e-9)


def test_metainf_common_dl():
    r = metainf(BCG_YI, [math.sqrt(v) for v in BCG_VI], studlab=BCG_STUDLAB,
                method_tau="DL", pooled="common")
    assert r.TE == pytest.approx([
        -0.42796108964187823, -0.42046588220075515, -0.42664675964922189,
        -0.33999461309983398, -0.43732413017959482, -0.31944702002775893,
        -0.42146409453503442, -0.74273706304094822, -0.42911381223139472,
        -0.40866317576307976, -0.44412955902537282, -0.43299224401856762,
        -0.4399940295652971], abs=1e-9)
    assert r.seTE == pytest.approx([
        0.040601145938919989, 0.04067052293891029, 0.040578947287201329,
        0.042268061676096959, 0.041163300902464117, 0.046379319634979287,
        0.040648499370980098, 0.052905153590745546, 0.041100406498433148,
        0.040961363532149547, 0.043472716573839013, 0.040561265409508843,
        0.040972044659416104], abs=1e-9)


def test_metacum_random_dl_sort_year():
    r = metacum(BCG_YI, [math.sqrt(v) for v in BCG_VI], studlab=BCG_STUDLAB,
                method_tau="DL", pooled="random", sortvar=BCG_YEAR)
    assert r.k == 13
    # first cumulative entry = earliest study (Aronson 1948)
    assert r.studlab[0] == "Adding Aronson 1948 (k=1)"
    assert r.studlab[12] == "Adding TPT Madras 1980 (k=13)"
    assert r.TE == pytest.approx([
        -0.88931133392020545, -1.3250034422448755, -0.96153532186821034,
        -0.95239474261544821, -1.0993222871615695, -0.95399424073437056,
        -0.89773336587553221, -0.7819364493697637, -0.8531334162503027,
        -0.76446126112149082, -0.68812306593791839, -0.79008659985370056,
        -0.71411722207322714], abs=1e-9)
    assert r.seTE == pytest.approx([
        0.57060035489295069, 0.34898877228738301, 0.23350096090599703,
        0.18462647473563587, 0.1978234883586413, 0.16764808806694406,
        0.17831589201897699, 0.17394917566123999, 0.17214420966904975,
        0.15422040755719971, 0.14961550773679833, 0.16308463840032222,
        0.1787420895354031], abs=1e-9)
    assert r.tau2 == pytest.approx([
        math.nan, 0.0, 0.075439865860435856, 0.042551518827814036,
        0.083189072908367023, 0.072319064775077824, 0.096532909179625537,
        0.12197554964457913, 0.13510356075869168, 0.12788015539773795,
        0.1382344134643228, 0.21088523257486985, 0.30876026286237612],
        abs=1e-9, nan_ok=True)
    assert r.I2 == pytest.approx([
        math.nan, 0.0, 0.3726300396294841, 0.22307367690589008,
        0.4765450674532557, 0.50909955591757816, 0.54585381018504642,
        0.64239899101769737, 0.64944229041195312, 0.73622741456029617,
        0.7475522319856196, 0.83820153415175369, 0.92117346854562054],
        abs=1e-9, nan_ok=True)


def test_metainf_to_dict():
    r = metainf(BCG_YI, [math.sqrt(v) for v in BCG_VI], studlab=BCG_STUDLAB,
                method_tau="DL", pooled="random")
    d = r.to_dict()
    assert d["k"] == 13
    assert d["pooled"] == "random"
    assert d["TE"][0] == pytest.approx(-0.70512519841745092, abs=1e-9)


# ---------------------------------------------------------------------------
# rma.glmm (metafor::rma.glmm — exact mixed-effects GLMM via AGQ)
# ---------------------------------------------------------------------------
#
# Reference values from R metafor 5.2-1 (lme4 backend, nAGQ=7).
# Two-arm measures (OR/RR/RD) use dat.bcg; single-arm (PLO/PR/PLN) use
# xi = [10,20,30,40], ni = [100,150,200,250].
#
# NOTE on RD ML: the identity-link binomial GLMM's stage-2 AGQ deviance is NaN
# in lme4 itself (Gauss-Hermite nodes push the linear predictor outside (0,1)),
# so lme4's two-stage fit falls back to the stage-1 Laplace profile (bobyqa over
# theta); metafor marks RD as "experimental".  Our reproduction minimises the
# same Laplace objective (-2*logLik + sqrL + ldL2) over theta and matches
# metafor to ~1e-5 (the residual ~5e-8 in beta comes from the theta argmin
# differing by ~5e-7 under the two optimisers).

_GLMM_XI = [10, 20, 30, 40]
_GLMM_NI = [100, 150, 200, 250]


def test_rma_glmm_or_fe():
    r = rma_glmm("OR", ai=T_POS, bi=T_NEG, ci=C_POS, di=C_NEG, method="FE")
    assert r.beta[0] == pytest.approx(-0.47725542790787762, abs=1e-6)
    assert r.se[0] == pytest.approx(0.04123188732297034, abs=1e-6)
    assert r.tau2 == pytest.approx(0.0, abs=1e-12)
    assert r.I2 == pytest.approx(0.0, abs=1e-12)
    assert r.H2 == pytest.approx(1.0, abs=1e-12)
    assert r.vt == pytest.approx(0.029081621720550975, abs=1e-12)
    assert r.QE_Wld == pytest.approx(163.16491518506905, abs=1e-5)
    assert r.QE_LRT == pytest.approx(176.95444818836168, abs=1e-5)
    assert r.QM == pytest.approx(133.9785656586952, abs=1e-3)
    assert r.parms == 14 and r.k_eff == 26 and r.k_f == 13
    assert r.fit_stats["ll.ML"] == pytest.approx(-157.64816844291505, abs=1e-5)
    assert r.fit_stats["dev.ML"] == pytest.approx(176.95444818836168, abs=1e-5)


def test_rma_glmm_or_ml():
    r = rma_glmm("OR", ai=T_POS, bi=T_NEG, ci=C_POS, di=C_NEG, method="ML")
    assert r.beta[0] == pytest.approx(-0.74501063937721645, abs=3e-4)
    assert r.se[0] == pytest.approx(0.17556516093478039, abs=3e-4)
    assert r.zval[0] == pytest.approx(-4.2434993105151184, abs=1e-3)
    assert r.pval[0] == pytest.approx(2.2006104936550462e-05, abs=1e-7)
    assert r.ci_lb[0] == pytest.approx(-1.0891120317493646, abs=5e-4)
    assert r.ci_ub[0] == pytest.approx(-0.40090924700506847, abs=5e-4)
    assert r.tau2 == pytest.approx(0.29489100485451791, abs=2e-4)
    assert r.I2 == pytest.approx(91.023432433785473, abs=1e-2)
    assert r.H2 == pytest.approx(11.140115557796719, abs=2e-2)
    assert r.QM == pytest.approx(18.007286398342288, abs=3e-3)
    assert r.parms == 15
    assert r.fit_stats["ll.ML"] == pytest.approx(-70.967718597939253, abs=2e-3)
    assert r.fit_stats["dev.ML"] == pytest.approx(3.5935484984100867, abs=2e-3)


def test_rma_glmm_rr_fe():
    r = rma_glmm("RR", ai=T_POS, bi=T_NEG, ci=C_POS, di=C_NEG, method="FE")
    assert r.beta[0] == pytest.approx(-0.47080893264346724, abs=1e-6)
    assert r.se[0] == pytest.approx(0.03926057021410171, abs=1e-6)
    assert r.vt == pytest.approx(0.026421191451388631, abs=1e-12)
    assert r.QE_Wld == pytest.approx(152.23300808693764, abs=1e-5)
    assert r.QE_LRT == pytest.approx(166.32030100063062, abs=1e-5)
    assert r.QM == pytest.approx(143.80572712211151, abs=1e-3)
    assert r.parms == 14
    assert r.fit_stats["ll.ML"] == pytest.approx(-152.33109484904952, abs=1e-5)


def test_rma_glmm_rr_ml():
    r = rma_glmm("RR", ai=T_POS, bi=T_NEG, ci=C_POS, di=C_NEG, method="ML")
    assert r.beta[0] == pytest.approx(-0.71390150275805153, abs=3e-4)
    assert r.se[0] == pytest.approx(0.16907261319161498, abs=3e-4)
    assert r.tau2 == pytest.approx(0.27113606217329195, abs=2e-4)
    assert r.I2 == pytest.approx(91.120636069347995, abs=1e-2)
    assert r.H2 == pytest.approx(11.262067956781779, abs=2e-2)
    assert r.QM == pytest.approx(17.829126318133461, abs=1e-3)
    assert r.parms == 15
    assert r.fit_stats["ll.ML"] == pytest.approx(-71.039540003151302, abs=2e-3)


def test_rma_glmm_rd_fe():
    r = rma_glmm("RD", ai=T_POS, bi=T_NEG, ci=C_POS, di=C_NEG, method="FE")
    assert r.beta[0] == pytest.approx(-0.00099849302343935314, abs=1e-7)
    assert r.se[0] == pytest.approx(0.00022774727056450452, abs=1e-8)
    assert r.vt == pytest.approx(8.5005423374021957e-07, abs=1e-13)
    assert r.QE_Wld == pytest.approx(276.4737127044807, abs=1e-5)
    assert r.QE_LRT == pytest.approx(294.23951653865288, abs=1e-5)
    assert r.QM == pytest.approx(19.22134207566528, abs=1e-3)
    assert r.parms == 14
    assert r.fit_stats["ll.ML"] == pytest.approx(-216.29070261806066, abs=1e-5)


def test_rma_glmm_rd_ml():
    # identity-link ML: lme4's stage-2 AGQ is NaN, so the fit is its stage-1
    # Laplace profile (see module note); the RX-based se/tau2 are stable.
    r = rma_glmm("RD", ai=T_POS, bi=T_NEG, ci=C_POS, di=C_NEG, method="ML")
    assert r.beta[0] == pytest.approx(-0.024940854747536899, abs=1e-6)
    assert r.se[0] == pytest.approx(0.010221278361179259, abs=1e-6)
    assert r.tau2 == pytest.approx(0.0012591667502266676, abs=1e-7)
    assert r.zval[0] == pytest.approx(-2.4400915292810206, abs=1e-4)
    assert r.pval[0] == pytest.approx(0.014683541046640229, abs=1e-5)
    assert r.I2 == pytest.approx(99.932536278029701, abs=1e-4)
    assert r.H2 == pytest.approx(1482.2781352624552, abs=0.1)
    assert r.QM == pytest.approx(5.9540466712689879, abs=1e-3)
    assert r.fit_stats["ll.ML"] == pytest.approx(-70.00580201530498, abs=1e-4)
    assert r.fit_stats["dev.ML"] == pytest.approx(1.6697153331415393, abs=1e-4)
    assert r.parms == 15


def test_rma_glmm_plo_fe():
    r = rma_glmm("PLO", xi=_GLMM_XI, ni=_GLMM_NI, method="FE")
    assert r.beta[0] == pytest.approx(-1.791759469228055, abs=1e-6)
    assert r.se[0] == pytest.approx(0.10801234495969682, abs=1e-6)
    assert r.vt == pytest.approx(0.049881281382585345, abs=1e-12)
    assert r.QE_Wld == pytest.approx(2.263447666128612, abs=1e-5)
    assert r.QE_LRT == pytest.approx(2.4253359341495049, abs=1e-5)
    assert r.QM == pytest.approx(275.17731397601256, abs=1e-4)
    assert r.parms == 1 and r.k_eff == 4
    assert r.fit_stats["ll.ML"] == pytest.approx(-10.807625293417528, abs=1e-6)


def test_rma_glmm_plo_ml():
    r = rma_glmm("PLO", xi=_GLMM_XI, ni=_GLMM_NI, method="ML")
    assert r.beta[0] == pytest.approx(-1.7917594688972798, abs=1e-6)
    assert r.se[0] == pytest.approx(0.10801234496070439, abs=1e-6)
    assert r.tau2 == pytest.approx(0.0, abs=1e-10)
    assert r.I2 == pytest.approx(0.0, abs=1e-8)
    assert r.H2 == pytest.approx(1.0, abs=1e-8)
    assert r.QM == pytest.approx(275.17731386927818, abs=1e-4)
    assert r.parms == 2
    assert r.fit_stats["ll.ML"] == pytest.approx(-10.807625293417509, abs=1e-6)


def test_rma_glmm_pr_fe():
    r = rma_glmm("PR", xi=_GLMM_XI, ni=_GLMM_NI, method="FE")
    assert r.beta[0] == pytest.approx(0.14285714285714285, abs=1e-8)
    assert r.se[0] == pytest.approx(0.013226001425322164, abs=1e-8)
    assert r.vt == pytest.approx(0.00069385579554022501, abs=1e-12)
    assert r.QE_Wld == pytest.approx(2.7363729218008843, abs=1e-5)
    assert r.QE_LRT == pytest.approx(2.4253359341495049, abs=1e-5)
    assert r.QM == pytest.approx(116.66666666666666, abs=1e-4)
    assert r.fit_stats["ll.ML"] == pytest.approx(-10.807625293417528, abs=1e-6)


def test_rma_glmm_pln_fe():
    r = rma_glmm("PLN", xi=_GLMM_XI, ni=_GLMM_NI, method="FE")
    assert r.beta[0] == pytest.approx(-1.9459101490553132, abs=1e-6)
    assert r.se[0] == pytest.approx(0.092582009830152939, abs=1e-6)
    assert r.vt == pytest.approx(0.036810997942903018, abs=1e-12)
    assert r.QE_Wld == pytest.approx(2.1981033615756167, abs=1e-5)
    assert r.QE_LRT == pytest.approx(2.4253359341495049, abs=1e-5)
    assert r.QM == pytest.approx(441.76607069341969, abs=1e-4)
    assert r.fit_stats["ll.ML"] == pytest.approx(-10.807625293417528, abs=1e-6)


# ---------------------------------------------------------------------------
# selmodel (metafor::selmodel, dat.bcg on the lnRR scale, ML refit)
# ---------------------------------------------------------------------------

def _selmodel_ml_fit(mods=None):
    if mods is None:
        return rma(BCG_YI, vi=BCG_VI, method="ML")
    return rma(BCG_YI, vi=BCG_VI, mods=mods, method="ML")


def test_selmodel_halfnorm():
    # reference: metafor::selmodel(res, type="halfnorm", alternative="greater")
    s = selmodel(_selmodel_ml_fit(), type="halfnorm", alternative="greater")

    assert s.type == "halfnorm"
    assert s.beta[0] == pytest.approx(-0.7653220767220561, abs=1e-4)
    assert s.se[0] == pytest.approx(0.2628652855045935, abs=1e-4)
    assert s.tau2 == pytest.approx(0.2579943185547304, abs=1e-4)
    assert s.se_tau2 == pytest.approx(0.1542558333834163, abs=1e-4)
    assert s.delta[0] == pytest.approx(0.3988460907894774, abs=1e-4)
    assert s.se_delta[0] == pytest.approx(1.599638268633516, abs=1e-3)
    assert s.ll == pytest.approx(-12.63464902707828, abs=1e-5)
    assert s.LRT == pytest.approx(0.06085464239710703, abs=1e-4)
    assert s.LRT_tau2 == pytest.approx(115.1749483061165, abs=1e-3)
    assert s.QM == pytest.approx(8.476606255939979, abs=1e-3)


def test_selmodel_power():
    # reference: metafor::selmodel(res, type="power", alternative="greater")
    s = selmodel(_selmodel_ml_fit(), type="power", alternative="greater")

    assert s.beta[0] == pytest.approx(-0.9469545105839964, abs=1e-4)
    assert s.tau2 == pytest.approx(0.2553436018020375, abs=1e-4)
    assert s.delta[0] == pytest.approx(0.1307961511285225, abs=1e-5)
    assert s.se_delta[0] == pytest.approx(0.1619597733191987, abs=2e-3)
    assert s.ll == pytest.approx(-12.32527627073395, abs=1e-4)


def test_selmodel_beta_intercept():
    # reference: metafor::selmodel(res, type="beta", alternative="greater")
    s = selmodel(_selmodel_ml_fit(), type="beta", alternative="greater")

    assert s.beta[0] == pytest.approx(-0.7892365674129026, abs=1e-3)
    assert s.tau2 == pytest.approx(0.4073780876934796, abs=1e-3)
    assert s.delta == pytest.approx([1.492517247906222, 1.108636561941954], abs=1e-3)
    assert s.se_delta == pytest.approx([0.7429285924008098, 0.1617873423708372], abs=1e-3)
    assert s.ll == pytest.approx(-11.95296017608093, abs=1e-4)


def test_selmodel_beta_mods():
    # reference: metafor::selmodel(rma(yi, vi, mods=~ablat, method="ML"), type="beta")
    r = _selmodel_ml_fit(mods=BCG_ABLAT)
    s = selmodel(r, type="beta", alternative="greater")

    assert len(s.beta) == 2
    assert s.beta[0] == pytest.approx(0.44679064548034381, abs=1e-3)
    assert s.beta[1] == pytest.approx(-0.03239933936700958, abs=1e-5)
    assert s.tau2 == pytest.approx(0.03352384983599838, abs=1e-4)
    assert s.delta == pytest.approx([1.298662449467102, 0.9340238094538855], abs=1e-3)
    assert s.se_delta[0] == pytest.approx(0.6748044825288786, abs=1e-3)
    assert s.se_delta[1] == pytest.approx(0.2587835198139862, abs=5e-3)
    assert s.ll == pytest.approx(-7.522387933416114, abs=1e-4)


def test_selmodel_stepfun():
    # reference: selmodel(res, type="stepfun", steps=c(0.5,1), alternative="greater")
    s = selmodel(
        _selmodel_ml_fit(), type="stepfun", alternative="greater", steps=(0.5, 1)
    )

    assert s.type == "stepfun"
    assert s.beta[0] == pytest.approx(-0.7530036605865462, abs=1e-5)
    assert s.se[0] == pytest.approx(0.2039843083927558, abs=1e-5)
    assert s.tau2 == pytest.approx(0.2547772053306595, abs=1e-5)
    # delta[0] is fixed to 1 for stepfun
    assert s.delta[0] == pytest.approx(1.0, abs=0)
    assert s.delta[1] == pytest.approx(0.6894417026948186, abs=1e-5)
    assert math.isnan(s.se_delta[0])
    assert s.se_delta[1] == pytest.approx(0.7802731729158419, abs=1e-5)
    assert s.ll == pytest.approx(-12.61297992430033, abs=1e-6)
    assert s.LRT == pytest.approx(0.1041928479530085, abs=1e-5)
    assert s.LRTdf == 1  # delta[0] is fixed at 1 and excluded
    assert list(s.steps) == [0.5, 1.0]


def test_selmodel_stepfun_decreasing():
    # reference: selmodel(res, type="stepfun", steps=c(0.5,1), decreasing=TRUE)
    s = selmodel(
        _selmodel_ml_fit(), type="stepfun", alternative="greater",
        steps=(0.5, 1), decreasing=True,
    )

    assert s.beta[0] == pytest.approx(-0.7530032120030661, abs=1e-5)
    assert s.tau2 == pytest.approx(0.25477758324072, abs=1e-5)
    assert s.delta[1] == pytest.approx(0.689455690793465, abs=1e-4)
    assert s.se_delta[1] == pytest.approx(0.7803041047462351, abs=1e-4)
    assert s.ll == pytest.approx(-12.61297992454713, abs=1e-6)


def test_selmodel_trunc():
    # reference: selmodel(res, type="trunc", steps=0, alternative="greater")
    s = selmodel(
        _selmodel_ml_fit(), type="trunc", alternative="greater", steps=0.0
    )

    assert s.beta[0] == pytest.approx(-0.7530070877332522, abs=1e-4)
    assert s.tau2 == pytest.approx(0.2547764729114153, abs=1e-4)
    assert s.delta[0] == pytest.approx(0.6894273714158993, abs=1e-4)
    assert s.se_delta[0] == pytest.approx(0.7802479219294256, abs=1e-4)
    assert s.ll == pytest.approx(-12.61297992450104, abs=1e-5)


def test_selmodel_negexppow():
    # reference: selmodel(res, type="negexppow", alternative="greater")
    s = selmodel(_selmodel_ml_fit(), type="negexppow", alternative="greater")

    assert s.beta[0] == pytest.approx(-0.7712821585447376, abs=1e-4)
    assert s.tau2 == pytest.approx(0.2581578898931833, abs=1e-4)
    assert s.delta == pytest.approx([0.4136946779378282, 0.3719660182932117], abs=1e-3)
    assert s.ll == pytest.approx(-12.62738497241576, abs=1e-5)


def test_selmodel_stepcon():
    # reference: selmodel(res, type="stepcon", steps=c(0.5,1)) using
    # Rsolnp::solnp with the monotonicity constraint 0 <= delta[j]-delta[j+1] <= 1
    s = selmodel(
        _selmodel_ml_fit(), type="stepcon", alternative="greater", steps=(0.5, 1)
    )

    assert s.type == "stepcon"
    assert s.beta[0] == pytest.approx(-0.753003758943408, abs=1e-5)
    assert s.se[0] == pytest.approx(0.203984311030195, abs=1e-5)
    assert s.tau2 == pytest.approx(0.25477715447288, abs=1e-5)
    assert s.se_tau2 == pytest.approx(0.146563667343049, abs=1e-5)
    assert s.delta[0] == pytest.approx(1.0, abs=0)
    assert s.delta[1] == pytest.approx(0.689441851639733, abs=1e-5)
    # step weights must be monotone non-increasing
    assert s.delta[0] >= s.delta[1]
    assert s.delta[1] <= 1.0
    assert math.isnan(s.se_delta[0])
    assert s.se_delta[1] == pytest.approx(0.780273797402034, abs=1e-5)
    assert s.ll == pytest.approx(-12.6129799243002, abs=1e-7)
    assert s.LRT == pytest.approx(0.104192847953307, abs=1e-6)
    assert s.LRTdf == 1
    assert s.QM == pytest.approx(13.6270157034449, abs=1e-4)
    assert s.parms == 3


def test_selmodel_stepcon_three_steps():
    # reference: selmodel(res, type="stepcon", steps=c(0.025,0.5,1))
    s = selmodel(
        _selmodel_ml_fit(), type="stepcon", alternative="greater",
        steps=(0.025, 0.5, 1),
    )

    assert s.beta[0] == pytest.approx(-0.752999617887824, abs=1e-4)
    assert s.se[0] == pytest.approx(0.207146351337534, abs=1e-4)
    assert s.tau2 == pytest.approx(0.254779472711916, abs=1e-5)
    assert s.delta == pytest.approx(
        [1.0, 0.999999766862390, 0.689453690383268], abs=1e-4
    )
    assert s.se_delta[1] == pytest.approx(1.40299152102497, abs=1e-4)
    assert s.se_delta[2] == pytest.approx(1.25742986240433, abs=1e-4)
    assert s.ll == pytest.approx(-12.612979997784, abs=1e-6)
    assert s.LRTdf == 2


def test_selmodel_negexp_flat_likelihood():
    # For negexp the profile likelihood is numerically flat in 'delta' below
    # roughly 0.03, so the exact delta estimate is optimizer dependent — R's own
    # optim / nlminb / Nelder-Mead / L-BFGS-B return 1.7e-2, 1.0e-7, 2.7e-5 and
    # 1.2e-5 respectively (with ll = -12.66534 / -12.66508). Compare the
    # optimizer-independent, well-identified quantities instead, using
    # tolerances that bracket the whole flat region.
    s = selmodel(_selmodel_ml_fit(), type="negexp", alternative="greater")

    assert s.delta[0] < 0.05
    assert s.beta[0] == pytest.approx(-0.7122, abs=2e-3)
    assert s.tau2 == pytest.approx(0.2795, abs=1.5e-3)
    assert s.ll == pytest.approx(-12.6651, abs=5e-4)
    assert s.se_delta[0] == pytest.approx(2.045, abs=1e-2)
    # H0: delta = 0 is on the boundary of the parameter space; the LRT is ~0
    assert s.LRT == pytest.approx(0.0, abs=1e-6)


def test_selmodel_rejects_unknown_type():
    with pytest.raises(ValueError):
        selmodel(_selmodel_ml_fit(), type="not-a-model")


def test_selmodel_trunc_two_sided_raises():
    with pytest.raises(ValueError):
        selmodel(_selmodel_ml_fit(), type="trunc", alternative="two.sided")


# ---------------------------------------------------------------------------
# P2: transf.* (metafor effect-size transforms)
# ---------------------------------------------------------------------------

def test_transf_proportion_family():
    assert transf_logit(0.1) == pytest.approx(-2.197224577336219, abs=1e-13)
    assert transf_probit(0.1) == pytest.approx(-1.281551565544601, abs=1e-13)
    assert transf_arcsin(0.5) == pytest.approx(0.785398163397448, abs=1e-13)
    assert transf_ilogit(0.1) == pytest.approx(0.524979187478940, abs=1e-13)
    assert transf_iprobit(0.1) == pytest.approx(0.539827837277029, abs=1e-13)
    assert transf_isqrt(0.3) == pytest.approx(0.09, abs=1e-15)
    assert transf_iarcsin(0.5) == pytest.approx(0.22984884706593, abs=1e-13)
    assert transf_iabt(0.5) == pytest.approx(0.393469340287367, abs=1e-14)
    assert transf_iahw(0.5) == pytest.approx(0.875, abs=1e-15)
    assert transf_abt(1.0) == float("inf")
    assert math.isnan(transf_ahw(3.0))


def test_transf_correlation_family():
    assert transf_rtoz(0.5) == pytest.approx(0.549306144334055, abs=1e-14)
    assert transf_rtoz(1.0) == float("inf")
    assert transf_ztor(1.0) == pytest.approx(0.761594155955765, abs=1e-14)
    assert transf_ztor2(1.0) == pytest.approx(0.580025658385974, abs=1e-14)
    assert transf_r2toz(0.5) == pytest.approx(0.881373587019543, abs=1e-14)
    assert transf_rtod(-1.0) == pytest.approx(-2.647216193577026, abs=1e-13)
    assert transf_rtorpb(-1.0) == pytest.approx(-0.797884560802865, abs=1e-14)
    assert transf_rpbtod(-1.0) == float("-inf")
    assert transf_rpbtorbis(-1.0) == pytest.approx(-1.25331413731550, abs=1e-13)
    assert transf_dtorpb(-2.0) == pytest.approx(-0.707106781186547, abs=1e-14)
    assert transf_dtorbis(0.2) == pytest.approx(0.124709417764487, abs=1e-14)


def test_transf_smd_family():
    assert transf_dtobesd(0.2) == pytest.approx(0.549751859510499, abs=1e-14)
    assert transf_dtocles(0.2) == pytest.approx(0.556231458009142, abs=1e-14)
    assert transf_dtocliffd(0.2) == pytest.approx(0.112462916018285, abs=1e-14)
    assert transf_dtoovl(0.2) == pytest.approx(0.920344325445942, abs=1e-14)
    assert transf_dtou1(0.2) == pytest.approx(0.147557552711348, abs=1e-14)
    assert transf_dtou2(0.2) == pytest.approx(0.539827837277029, abs=1e-14)
    assert transf_dtou3(0.2) == pytest.approx(0.579259709439103, abs=1e-14)
    assert transf_dtolnor_logis(0.5) == pytest.approx(0.906899682117109, abs=1e-14)
    assert transf_dtolnor_norm(0.5) == pytest.approx(0.825, abs=1e-14)
    assert transf_clestod(0.5) == pytest.approx(0.0, abs=1e-14)
    assert transf_dtomd(0.2, {"sd": 2.0}) == pytest.approx(0.4, abs=1e-14)


def test_transf_exp_and_rate():
    assert transf_exp_int(0.5, {"tau2": 0.1}) == pytest.approx(
        math.exp(0.55), abs=1e-13
    )
    assert transf_exp_mode(0.5, {"tau2": 0.1}) == pytest.approx(
        math.exp(0.4), abs=1e-13
    )
    assert transf_pft(0.1, 100) == pytest.approx(
        0.5 * (math.asin(math.sqrt(10 / 101)) + math.asin(math.sqrt(11 / 101))),
        abs=1e-14,
    )
    assert transf_irft(0.5, 1.0) == pytest.approx(
        0.5 * (math.sqrt(0.5) + math.sqrt(1.5)), abs=1e-14
    )


def test_transf_integrated_variants():
    # tau2 = 0 reproduces the plain transform
    assert transf_ilogit_int(0.3, {"tau2": 0.0}) == pytest.approx(
        transf_ilogit(0.3), abs=1e-14
    )
    assert transf_ztor_int(0.3, {"tau2": 0.0}) == pytest.approx(
        transf_ztor(0.3), abs=1e-14
    )
    # with heterogeneity the mean of the back-transformed value is pulled
    # towards the middle of the parameter space (Jensen: logit link is concave)
    v = transf_ilogit_int(0.3, {"tau2": 0.25})
    assert v < transf_ilogit(0.3)
    assert v > 0.5


# ---------------------------------------------------------------------------
# P2: fmt / dfround
# ---------------------------------------------------------------------------

def test_fmt_helpers():
    assert fmtp(0.00001234) == "<.0001"
    assert fmtp(0.5) == "0.5000"
    assert fmtp2(0.0005) == "p < .001"
    assert fmtp2(0.03) == "p = .030"
    assert fmtp2(0.23456) == "p = .23"
    assert fmtt(2.3456, "t", df=12, pval=0.03) == "t(df = 12) = 2.3456, p-val = 0.0300"
    assert fmtt(2.3456, "t", df=12, pval=0.03, format=2) == (
        "t = 2.3456, df = 12, p-val = 0.0300"
    )
    assert fmtx(2.3456, 3) == "2.346"


def test_dfround():
    out = dfround({"a": [2.3456, 1.005], "b": ["x", "y"]}, digits=2)
    assert out["a"] == pytest.approx([2.35, 1.0], abs=1e-12)
    assert out["b"] == ["x", "y"]


# ---------------------------------------------------------------------------
# P0: diagnostics (deltamethod / tes / vif)
# ---------------------------------------------------------------------------

def test_deltamethod_order1():
    r = rma(BCG_YI, vi=BCG_VI, method="REML")
    dm = deltamethod(np.asarray(r.b).ravel(), np.asarray(r.vb), lambda x: np.exp(x))

    assert dm.coef[0] == pytest.approx(0.489420936788213, abs=1e-12)
    assert dm.se[0] == pytest.approx(0.0879888456987118, abs=1e-10)
    assert dm.zval[0] == pytest.approx(5.56230659581636, abs=1e-8)
    assert dm.pval[0] == pytest.approx(2.66232015675564e-08, abs=1e-16)
    assert dm.ci_lb[0] == pytest.approx(0.316965968177486, abs=1e-10)
    assert dm.ci_ub[0] == pytest.approx(0.66187590539894, abs=1e-10)


def test_deltamethod_order2_and_multiarg():
    r = rma(BCG_YI, vi=BCG_VI, method="REML")
    dm2 = deltamethod(
        np.asarray(r.b).ravel(), np.asarray(r.vb), lambda x: np.exp(x), order=2
    )
    assert dm2.se[0] == pytest.approx(0.0886969768425675, abs=1e-9)

    dmt = deltamethod([0.5, 0.2], np.diag([0.01, 0.04]), lambda a, b: a / b)
    assert dmt.coef[0] == pytest.approx(2.5, abs=1e-12)
    assert dmt.se[0] == pytest.approx(2.54950975683505, abs=1e-9)


def test_tes_exact():
    t = tes(BCG_YI, BCG_VI)

    assert t.k == 13 and t.O == 8
    assert t.E == pytest.approx(6.09265909028069, abs=1e-10)
    assert t.OEratio == pytest.approx(1.3130555774509, abs=1e-10)
    assert t.test == "exact"
    assert t.pval == pytest.approx(0.134408585400308, abs=1e-8)
    assert t.theta == pytest.approx(-0.430285163654091, abs=1e-12)
    assert t.theta_lim == pytest.approx(-0.407790930499875, abs=1e-5)
    assert t.power[0] == pytest.approx(0.1172566740573639, abs=1e-12)


def test_tes_with_tau2():
    t = tes(BCG_YI, BCG_VI, tau2=0.313243325980895, test="chi2")

    assert t.power[0] == pytest.approx(0.317611611454395, abs=1e-10)
    assert t.E == pytest.approx(7.8088303607381, abs=1e-9)
    assert t.X2 == pytest.approx(0.0117200640231583, abs=1e-9)
    assert t.pval == pytest.approx(0.456895025358493, abs=1e-10)


def test_vif_ablat_year():
    r = rma(BCG_YI, vi=BCG_VI, mods=np.column_stack([BCG_ABLAT, BCG_YEAR]),
            method="REML")
    v = vif(r)

    assert v.vifs == pytest.approx([1.53913188694103, 1.53913188694103], abs=1e-9)
    assert v.sifs == pytest.approx([1.24061754257347, 1.24061754257347], abs=1e-9)


def test_vif_rejects_intercept_only():
    r = rma(BCG_YI, vi=BCG_VI, method="REML")
    with pytest.raises(ValueError):
        vif(r)


# ---------------------------------------------------------------------------
# P2: meta measure conversions
# ---------------------------------------------------------------------------

def test_measure_conversions():
    assert asin2ir(1.2, time=12) == pytest.approx(1.39863474151235, abs=1e-12)
    assert asin2p(0.8) == pytest.approx(0.514599761150644, abs=1e-13)
    assert asin2p(0.8, n=100) == pytest.approx(0.514745035367195, abs=1e-13)
    assert p2asin(0.3) == pytest.approx(0.579639740363704, abs=1e-13)
    assert cor2z(0.5) == pytest.approx(0.549306144334055, abs=1e-14)
    assert z2cor(0.5) == pytest.approx(0.46211715726001, abs=1e-14)
    assert logit2p(0.5) == pytest.approx(0.622459331201855, abs=1e-14)
    assert p2logit(0.3) == pytest.approx(-0.847297860387204, abs=1e-14)
    assert logVR2VE(0.2) == pytest.approx(-22.140275816017, abs=1e-11)
    assert VE2logVR(20.0) == pytest.approx(-0.22314355131421, abs=1e-13)


def test_smd2or_or2smd():
    m = smd2or([0.5, -0.3, 0.2], [0.15, 0.12, 0.18], method="HH")
    assert np.asarray(m.TE) == pytest.approx(
        [0.906899682117109, -0.544139809270265, 0.362759872846844], abs=1e-12
    )
    assert np.asarray(m.seTE) == pytest.approx(
        [0.272069904635133, 0.217655923708106, 0.326483885562159], abs=1e-12
    )
    assert m.TE_common == pytest.approx(0.0947507130570113, abs=1e-10)
    assert m.TE_random == pytest.approx(0.226735867702728, abs=1e-10)

    m2 = or2smd([0.8, -0.5, 0.3], [0.2, 0.15, 0.25], method="CS")
    assert np.asarray(m2.TE) == pytest.approx(
        [0.484848484848485, -0.303030303030303, 0.181818181818182], abs=1e-13
    )
    assert m2.TE_common == pytest.approx(0.0182842731607361, abs=1e-10)
    assert m2.TE_random == pytest.approx(0.114520527862694, abs=1e-10)


def test_rd_and_nnt():
    mg = metagen([-0.7, -0.5, -0.9], [0.2, 0.25, 0.3], sm="OR")
    r = rd(mg, p_c=[0.2, 0.4])
    assert r.common["RD"] == pytest.approx(
        [-0.0877070825445973, -0.1477601369805933], abs=1e-12
    )
    n = nnt(mg, p_c=[0.2, 0.4])
    assert n.nnt_common["NNT"] == pytest.approx(
        [11.40158777361589, 6.76772518241059], abs=1e-10
    )


# ---------------------------------------------------------------------------
# P0: hettest / vcalc / simulate / matreg
# ---------------------------------------------------------------------------

def test_hettest_score():
    r = rma(BCG_YI, vi=BCG_VI, method="REML")
    h = hettest(r, test="score")
    assert h.pval == pytest.approx(0.996281845351419, abs=1e-12)
    assert h.k == 13


def test_vcalc_structure():
    vi = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    cl = [1, 1, 2, 2, 1, 3, 3]
    V = vcalc(vi, cluster=cl, rho=0.5)
    assert np.allclose(np.diag(V), vi)
    assert V[0, 1] == pytest.approx(0.5 * math.sqrt(0.1 * 0.2), abs=1e-15)
    assert V[0, 2] == 0.0          # different cluster
    assert np.allclose(V, V.T)


def test_simulate_rma_shape():
    r = rma(BCG_YI, vi=BCG_VI, method="REML")
    sim = simulate_rma(r, nsim=3, seed=1234)
    assert sim.shape == (13, 3)
    assert np.all(np.isfinite(sim))


def test_matreg_runs():
    R = [[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]]
    m = matreg(1, [2, 3], R, n=100)
    assert len(m.beta) == 1
    assert m.k == 2
    assert m.se[0] > 0


# ---------------------------------------------------------------------------
# P2: conv.delta / conv.wald / to.long / to.wide / to.table
# ---------------------------------------------------------------------------

def test_conv_delta():
    d = conv_delta([0.5, -0.3], [0.04, 0.09], np.exp)
    assert d.yi == pytest.approx([1.648721270700128, 0.740818220681718], abs=1e-14)
    assert d.vi == pytest.approx(
        [0.1087312731378373, 0.0493930472477669], abs=1e-12
    )


def test_conv_wald():
    w = conv_wald([0.5, -0.3], [0.1, -0.7], [0.9, 0.1])
    assert w.yi == pytest.approx([0.5, -0.3], abs=1e-15)
    assert w.vi == pytest.approx(
        [0.0416508434603209, 0.0416508434603209], abs=1e-14
    )


def test_conv_fivenum_luo_wan_shi():
    # metafor::conv.fivenum(..., method="luo/wan/shi", test=FALSE)
    r1 = conv_fivenum(min=1, median=5, max=9, n=100, test=False)
    assert r1.mean == pytest.approx(5.0, abs=1e-15)
    assert r1.sd == pytest.approx(1.60090254981974, abs=1e-12)
    assert r1.case == 1

    r2 = conv_fivenum(q1=3, median=5, q3=7, n=100, test=False)
    assert r2.mean == pytest.approx(5.0, abs=1e-15)
    assert r2.sd == pytest.approx(3.00881972104808, abs=1e-12)
    assert r2.case == 2

    r3 = conv_fivenum(min=1, q1=3, median=5, q3=7, max=9, n=100, test=False)
    assert r3.mean == pytest.approx(5.0, abs=1e-15)
    assert r3.sd == pytest.approx(2.34137858682207, abs=1e-12)
    assert r3.case == 3

    r4 = conv_fivenum(min=1, q1=3, median=5, q3=7, max=9, n=20, test=False)
    assert r4.sd == pytest.approx(2.45295669642881, abs=1e-12)


def test_conv_fivenum_wan2014_and_hozo():
    r = conv_fivenum(min=1, q1=3, median=5, q3=7, max=9, n=100,
                     method="wan2014", test=False)
    assert r.mean == pytest.approx(5.0, abs=1e-15)
    assert r.sd == pytest.approx(2.30486113543391, abs=1e-12)

    r = conv_fivenum(min=1, q1=3, median=5, q3=7, max=9, n=20,
                     method="wan2014", test=False)
    assert r.sd == pytest.approx(2.66621733691339, abs=1e-12)

    r = conv_fivenum(min=1, median=5, max=9, n=100, method="hozo2005", test=False)
    assert r.mean == pytest.approx(5.0, abs=1e-15)
    assert r.sd == pytest.approx(1.33333333333333, abs=1e-12)

    # hozo2005 cannot handle the q1/q3 cases
    r = conv_fivenum(q1=3, median=5, q3=7, n=100, method="hozo2005", test=False)
    assert math.isnan(r.mean) and math.isnan(r.sd)


def test_to_long_wide_roundtrip():
    lg = to_long(ai=[4, 6], bi=[119, 300], ci=[11, 29], di=[128, 274])
    assert lg["event"] == pytest.approx([4, 11, 6, 29], abs=0)
    assert lg["n"] == pytest.approx([123, 139, 306, 303], abs=0)
    assert lg["group"] == [1, 2, 1, 2]

    wd = to_wide(lg)
    assert wd["ai"] == pytest.approx([4, 6], abs=0)
    assert wd["bi"] == pytest.approx([119, 300], abs=0)
    assert wd["ci"] == pytest.approx([11, 29], abs=0)
    assert wd["di"] == pytest.approx([128, 274], abs=0)


def test_to_table():
    t = to_table([4, 6], [119, 300], [11, 29], [128, 274])
    assert t == [[[4.0, 119.0], [11.0, 128.0]], [[6.0, 300.0], [29.0, 274.0]]]


# ---------------------------------------------------------------------------
# P1: plotting layer smoke tests
# ---------------------------------------------------------------------------

def _plots():
    import matplotlib.pyplot as plt
    r = rma(BCG_YI, vi=BCG_VI, method="REML")
    rm = rma(BCG_YI, vi=BCG_VI, mods=BCG_ABLAT, method="REML")

    class _FakeRank:
        rank_matrix = np.array([[0.1, 0.6, 0.3], [0.5, 0.3, 0.2]])
        treatments = ["A", "B"]

    class _FakeHeat:
        heat_matrix = np.array([[0.1, 0.2], [0.3, 0.0]])
        designs = ["A:B", "A:C"]

    out = [
        ("radial", radial(BCG_YI, BCG_VI, pooled=r.b[0], pooled_se=r.se[0])),
        ("labbe", labbe([4, 6], [123, 306], [11, 29], [139, 303])),
        ("llplot", llplot(r)),
        ("bubble", bubble(rm)),
        ("drapery", drapery(r)),
        ("qqnorm", qqnorm_rma(r)),
        ("rankogram", rankogram(_FakeRank())),
        ("netheat", netheat(_FakeHeat())),
        ("hasse", hasse(["A", "B", "C"], [("A", "B"), ("A", "C")])),
        ("traffic", traffic_light(["s1", "s2"], [["low", "high"], ["unclear", "low"]])),
    ]
    return out


@pytest.mark.parametrize("name", [
    "radial", "labbe", "llplot", "bubble", "drapery",
    "qqnorm", "rankogram", "netheat", "hasse", "traffic",
])
def test_plotting_layer(name):
    import matplotlib.pyplot as plt
    plots = dict(_plots())
    result = plots[name]
    assert result.ax is not None
    assert result.figure is not None
    plt.close(result.figure)


# ---------------------------------------------------------------------------
# P0: profile.rma.uni / netcomplex
# ---------------------------------------------------------------------------

def test_profile_rma():
    r = rma(BCG_YI, vi=BCG_VI, method="REML")
    p = profile_rma(r)

    assert len(p.tau2) == 20
    # the grid runs from the QP lower to the QP upper bound of tau^2
    assert p.tau2[0] == pytest.approx(0.119695312816638, abs=1e-12)
    assert p.tau2[1] == pytest.approx(0.171894836469206, abs=1e-12)
    assert p.tau2[2] == pytest.approx(0.224094360121775, abs=1e-12)
    # profile log-likelihood (metafor includes +0.5*log|X'X| for REML)
    assert p.ll[0] == pytest.approx(-13.9751072638578, abs=1e-10)
    assert p.ll[1] == pytest.approx(-12.8696048222156, abs=1e-10)
    assert p.ll[2] == pytest.approx(-12.4048092079714, abs=1e-10)
    # the maximum sits at tau^2_hat
    assert p.maxll > max(p.ll[:2])


def test_netcomplexity_is_the_cycle_count():
    class _Fake:
        m = 28
        n = 10
        n_subnets = 1

    assert netcomplexity(_Fake()) == 28 - 10 + 1


# ---------------------------------------------------------------------------
# meta package composites: metabind / metamerge
# ---------------------------------------------------------------------------

def test_metabind_equals_direct_pooling():
    m1 = metagen([-0.7, -0.5, -0.9], [0.2, 0.25, 0.3], sm="RR")
    m2 = metagen([-0.2, -0.4], [0.15, 0.18], sm="RR")

    b = metabind(m1, m2)
    direct = metagen(
        np.concatenate([m1.TE, m2.TE]),
        np.concatenate([m1.seTE, m2.seTE]),
        sm="RR",
    )

    assert b.parts == 2
    assert list(b.TE) == pytest.approx([-0.7, -0.5, -0.9, -0.2, -0.4], abs=1e-15)
    assert b.meta.TE_random == pytest.approx(direct.TE_random, abs=1e-12)
    assert b.meta.TE_common == pytest.approx(direct.TE_common, abs=1e-12)


def test_metamerge_requires_same_k():
    m1 = metagen([-0.7, -0.5, -0.9], [0.2, 0.25, 0.3], sm="RR")
    m2 = metagen([-0.2, -0.4], [0.15, 0.18], sm="RR")
    with pytest.raises(ValueError):
        metamerge(m1, m2)

    mm = metamerge(m1, m1)
    assert len(mm.TE) == 6


# ---------------------------------------------------------------------------
# settings (meta::gs / setvals)
# ---------------------------------------------------------------------------

def test_settings_roundtrip():
    import pimeta as _ma

    assert gs("method.tau") == "REML"
    assert gs("method_tau") == "REML"
    assert gs("level.ma") == 0.95
    assert gs(package="netmeta")["reference.group"] == ""

    before = gs("method.tau")
    try:
        setvals(method_tau="DL")
        assert _ma.settings_meta["method.tau"] == "DL"
        assert gs(None)["method.tau"] == "DL"
    finally:
        _ma.settings_meta["method.tau"] = before

    with pytest.raises(KeyError):
        gs("no-such-setting")
    with pytest.raises(ValueError):
        gs(package="nope")


def test_setvals_positional_pairs():
    import pimeta as _ma

    before = _ma.settings_netmeta.get("nchar.trts")
    try:
        setvals(("nchar.trts", 10), package="netmeta")
        assert _ma.settings_netmeta["nchar.trts"] == 10
    finally:
        _ma.settings_netmeta["nchar.trts"] = before


# ---------------------------------------------------------------------------
# netmetareg (netmeta::netmetareg) — reference values from R 4.6.1
# netmeta 3.7-0 / metafor 5.2-1
# ---------------------------------------------------------------------------

NMR_BETA = [
    -0.3601051148959797, 19.2699999999878457, -1.3688412890302029,
    -2.0205852386654461, -1.7676057852077509, -1.1629551799246056,
    -0.57, 1.7582670874796296, -0.7, -0.2140501077876389,
    -7.8, 0.1620250705465806, 0.5052913980417371,
    0.8055897713051845, -0.0172050847578382, -1.2980784144819848,
]
NMR_SE = [
    0.967570618136879, 15.350971205139974, 0.243870255159132, 4.058160190160569,
    1.322886847741925, 0.302429029891335, 0.355496870143635, 1.724563077955539,
    0.354847156789963, 0.510742414453243, 5.985673557450532, 0.174454603910032,
    1.911686757072107, 1.735565118963198, 0.291274193660608, 1.062319397882920,
]


@pytest.mark.skipif(not SENN_CSV.exists(), reason="Senn 2013 benchmark CSV not present")
def test_netmetareg_senn():
    import csv

    te, sete, t1, t2, stud = [], [], [], [], []
    with SENN_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            te.append(float(row["TE"]))
            sete.append(float(row["seTE"]))
            t1.append(row["treat1"])
            t2.append(row["treat2"])
            stud.append(row["studlab"])

    net = netmeta(te, sete, t1, t2, stud, sm="MD", reference_group="plac")
    cov = [(i + 1) / 10 for i in range(len(te))]

    r = netmetareg(net, covar=cov, reference_group="plac")
    assert isinstance(r, NetmetaregResult)

    # 18 candidate columns minus the 2 collinear ones (sita:covar, vild:covar)
    assert r.k == 27
    assert r.p == 16
    assert r.names[:9] == ["acar", "benf", "metf", "migl", "piog", "rosi",
                           "sita", "sulf", "vild"]
    assert sum(r.coef_na) == 2

    for got, want in zip(r.beta, NMR_BETA):
        assert got == pytest.approx(want, abs=1e-6)
    for got, want in zip(r.se, NMR_SE):
        assert got == pytest.approx(want, abs=1e-5)

    assert r.tau2 == pytest.approx(0.10971121468192, abs=1e-6)
    assert r.QE == pytest.approx(69.8064111559196, abs=1e-6)
    assert r.QEdf == 11
    assert r.QM == pytest.approx(162.511565153185, abs=1e-3)
    assert r.QMdf == 16
    assert r.QMp == pytest.approx(2.6073535834755e-26, rel=1e-4)



# ---------------------------------------------------------------------------
# netmeasures / netimpact / netpath (R netmeta 3.7-0 reference values)
# ---------------------------------------------------------------------------

def _senn_net():
    import csv

    te, sete, t1, t2, stud = [], [], [], [], []
    with SENN_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            te.append(float(row["TE"]))
            sete.append(float(row["seTE"]))
            t1.append(row["treat1"])
            t2.append(row["treat2"])
            stud.append(row["studlab"])
    return netmeta(te, sete, t1, t2, stud, sm="MD", reference_group="plac")


@pytest.mark.skipif(not SENN_CSV.exists(), reason="Senn 2013 benchmark CSV not present")
def test_netmeasures_senn():
    nm = netmeasures(_senn_net())
    assert isinstance(nm, NetmeasuresResult)
    assert nm.proportion["acar:metf"] == pytest.approx(0.281224092924669, abs=1e-9)
    assert nm.proportion["metf:plac"] == pytest.approx(0.560577416023613, abs=1e-9)
    assert nm.proportion["plac:vild"] == pytest.approx(1.0, abs=1e-12)
    assert nm.proportion["acar:benf"] == pytest.approx(0.0, abs=1e-12)
    assert nm.meanpath["plac:rosi"] == pytest.approx(1.30635951454785, abs=1e-9)
    assert nm.minpar["piog:plac"] == pytest.approx(2.58691570472924, abs=1e-9)
    assert nm.minpar_study["plac:rosi"] == pytest.approx(7.28567876831824, abs=1e-9)


@pytest.mark.skipif(not SENN_CSV.exists(), reason="Senn 2013 benchmark CSV not present")
def test_netimpact_senn():
    ni = netimpact(_senn_net())
    assert isinstance(ni, NetimpactResult)
    assert len(ni.studies) == 26
    assert len(ni.comparisons) == 45
    assert ni.seTE_ignore == pytest.approx(466.9, abs=1e-9)
    assert ni.impact_common["DeFronzo1995"]["acar:metf"] == pytest.approx(
        0.0270888617893, abs=1e-9)
    assert ni.impact_common["Willms1999"]["metf:plac"] == pytest.approx(
        0.03335295769579, abs=1e-9)
    assert ni.impact_random["DeFronzo1995"]["acar:plac"] == pytest.approx(
        0.01210755496979, abs=1e-9)
    assert ni.impact_random["Willms1999"]["acar:metf"] == pytest.approx(
        0.285417510918586, abs=1e-9)


@pytest.mark.skipif(not SENN_CSV.exists(), reason="Senn 2013 benchmark CSV not present")
def test_netpath_senn():
    res = netpath(_senn_net(), "metf", "plac")
    assert res["comparison"] == "metf:plac"
    assert len(res["results"]) == 7
    assert res["Sigma"].shape == (7, 7)
    assert res["results"][0]["Q"] == pytest.approx(22.4518410854719, abs=1e-9)
    assert res["results"][0]["df"] == 6
    assert res["results"][0]["pval"] == pytest.approx(
        0.00100247516462062, abs=1e-12)
    assert res["note"] == "Matrix Sigma singular, pseudo-inverse used"

    with pytest.raises(ValueError):
        netpath(_senn_net(), "metf", "nope")


# ---------------------------------------------------------------------------
# conv.2x2 (metafor::conv.2x2) — reference values from R metafor 5.2-1
# ---------------------------------------------------------------------------

def test_conv_2x2_from_odds_ratio():
    r = conv_2x2(ori=[1.5, 2.0, 0.8], n1i=[30, 50, 20], n2i=[40, 50, 25],
                 ni=[70, 100, 45])
    assert r.ai == [19.0, 29.0, 10.0]
    assert r.bi == [11.0, 21.0, 10.0]
    assert r.ci == [21.0, 21.0, 15.0]
    assert r.di == [19.0, 29.0, 10.0]


def test_conv_2x2_from_chi2_and_phi():
    r = conv_2x2(x2i=[3.5, 0.5], n1i=[30, 60], n2i=[40, 60], ni=[70, 120])
    assert (r.ai, r.bi, r.ci, r.di) == ([21.0, 32.0], [9.0, 28.0],
                                        [19.0, 28.0], [21.0, 32.0])
    r = conv_2x2(ri=[0.2, -0.15], n1i=[30, 60], n2i=[40, 60], ni=[70, 120])
    assert (r.ai, r.bi, r.ci, r.di) == ([21.0, 26.0], [9.0, 34.0],
                                        [19.0, 34.0], [21.0, 26.0])


def test_conv_2x2_from_accuracy_measures():
    kw = dict(sens=[0.85, 0.7], spec=[0.9, 0.75], ppv=[0.8, 0.6],
              npv=[0.92, 0.82], ni=[70, 120])
    r = conv_2x2(**kw)                       # method = "optim"
    assert (r.ai, r.bi, r.ci, r.di) == ([20.0, 29.0], [5.0, 20.0],
                                        [3.0, 13.0], [42.0, 58.0])
    r = conv_2x2(method="simple", **kw)
    assert (r.ai, r.bi, r.ci, r.di) == ([19.0, 29.0], [5.0, 19.0],
                                        [3.0, 13.0], [43.0, 59.0])
    r = conv_2x2(method="search", **kw)
    assert (r.ai, r.bi, r.ci, r.di) == ([19.0, 29.0], [5.0, 20.0],
                                        [4.0, 13.0], [42.0, 58.0])


def test_conv_2x2_drop01_and_validation():
    # sens == 1 -> the table is dropped (drop01 = TRUE)
    r = conv_2x2(sens=[1.0], spec=[0.9], ppv=[0.8], npv=[0.92], ni=[70])
    assert r.ai[0] != r.ai[0]  # NaN
    r = conv_2x2(sens=[1.0], spec=[0.9], ppv=[0.8], npv=[0.92], ni=[70],
                 drop01=False)
    assert r.ai[0] == r.ai[0]  # not NaN
    with pytest.raises(ValueError):
        conv_2x2(sens=[1.2], spec=[0.9], ppv=[0.8], npv=[0.92], ni=[70])
    with pytest.raises(ValueError):
        conv_2x2(ori=[1.5], n1i=[80], n2i=[40], ni=[70])
    with pytest.raises(ValueError):
        conv_2x2(sens=[0.5], spec=[0.5], var_names=("a", "b"))


# ---------------------------------------------------------------------------
# metaadd (meta::metaadd)
# ---------------------------------------------------------------------------

def test_metaadd_single_result():
    m = metagen([-0.7, -0.5, -0.9], [0.2, 0.25, 0.3], sm="RR")
    r = metaadd(m, type="common", TE=0.5, se=0.1, lower=0.3, upper=0.7,
                method="Added")
    assert len(r.TE_common) == 2
    assert r.TE_common[0] == pytest.approx(m.TE_common, abs=1e-12)
    assert r.TE_common[1] == pytest.approx(0.5, abs=1e-12)
    assert r.seTE_common[1] == pytest.approx(0.1, abs=1e-12)
    assert r.lower_common[1] == pytest.approx(0.3, abs=1e-12)
    assert r.upper_common[1] == pytest.approx(0.7, abs=1e-12)
    assert r.common is True
    assert r.is_metaadd is True
    assert r.added[0]["text"] == "Added result"


def test_metaadd_from_meta_object():
    m = metagen([-0.7, -0.5, -0.9], [0.2, 0.25, 0.3], sm="RR")
    r = metaadd(m, data=m)
    # common + random + tau2 rows appended
    assert [a["type"] for a in r.added] == ["common", "random", "tau2"]
    assert len(r.TE_common) == 2
    assert len(r.TE_random) == 2
    assert r.TE_random[1] == pytest.approx(m.TE_random, abs=1e-12)


def test_metaadd_returns_input_when_nothing_supplied():
    m = metagen([-0.7, -0.5, -0.9], [0.2, 0.25, 0.3], sm="RR")
    assert metaadd(m) is m


def test_metaadd_requires_arguments():
    m = metagen([-0.7, -0.5, -0.9], [0.2, 0.25, 0.3], sm="RR")
    with pytest.raises(ValueError):
        metaadd(m, type="common", lower=0.3, upper=0.7)          # no TE
    with pytest.raises(ValueError):
        metaadd(m, type="random", TE=0.5)                        # no bounds
    with pytest.raises(ValueError):
        metaadd(m, type="nope", TE=0.5, se=0.1, lower=0.3, upper=0.7)


# ---------------------------------------------------------------------------
# metacr (meta::metacr) — review-level dispatch to metabin/metacont/metagen
# ---------------------------------------------------------------------------

METACR_ROWS = [
    {"comp.no": 1, "outcome.no": 1, "type": "D", "sm": "RR",
     "method": "Inverse", "studlab": "S1", "event.e": 10, "n.e": 50,
     "event.c": 20, "n.c": 50},
    {"comp.no": 1, "outcome.no": 1, "type": "D", "sm": "RR",
     "method": "Inverse", "studlab": "S2", "event.e": 15, "n.e": 60,
     "event.c": 25, "n.c": 60},
    {"comp.no": 2, "outcome.no": 1, "type": "I", "sm": "MD",
     "method": "Inverse", "studlab": "S3", "TE": 0.4, "seTE": 0.2},
    {"comp.no": 2, "outcome.no": 1, "type": "I", "sm": "MD",
     "method": "Inverse", "studlab": "S4", "TE": 0.9, "seTE": 0.3},
]


def test_metacr_binary_comparison():
    r = metacr(METACR_ROWS)
    assert isinstance(r, MetacrResult)
    assert r.type == "D" and r.n_studies == 2
    assert r.meta.fixed.pooled_effect == pytest.approx(
        0.5578003108649138, abs=1e-12)
    direct = metabin([10, 15], [50, 60], [20, 25], [50, 60], sm="RR")
    assert r.meta.fixed.pooled_effect == pytest.approx(
        direct.fixed.pooled_effect, abs=1e-12)


def test_metacr_inverse_variance_comparison():
    r = metacr(METACR_ROWS, comp_no=2)
    assert r.type == "I" and r.n_studies == 2
    direct = metagen([0.4, 0.9], [0.2, 0.3], sm="MD")
    assert r.meta.TE_common == pytest.approx(direct.TE_common, abs=1e-12)
    assert r.meta.TE_random == pytest.approx(direct.TE_random, abs=1e-12)


def test_metacr_empty_selection():
    with pytest.warns(UserWarning):
        assert metacr(METACR_ROWS, comp_no=99) is None


# ---------------------------------------------------------------------------
# metafor matrix helpers (R reference values from metafor 5.2-1)
# ---------------------------------------------------------------------------

def test_bldiag_matches_r():
    out = bldiag(np.array([[1, 2], [3, 4]]), np.array([[5, 6]]), 7)
    assert out.shape == (4, 5)
    assert out.tolist() == [[1, 2, 0, 0, 0], [3, 4, 0, 0, 0],
                            [0, 0, 5, 6, 0], [0, 0, 0, 0, 7]]


def test_vec2mat_matches_r():
    out = vec2mat([0.3, 0.5, 0.7], corr=True)
    assert np.allclose(out, [[1, 0.3, 0.5], [0.3, 1, 0.7], [0.5, 0.7, 1]])
    out2 = vec2mat([1, 0.3, 2, 0.5, 0.7, 3], diag=True, corr=False)
    assert np.allclose(out2, [[1, 0.3, 2], [0.3, 0.5, 0.7], [2, 0.7, 3]])
    with pytest.raises(ValueError):
        vec2mat([1, 2, 3, 4])          # not a triangle


def test_cmicalc_matches_r():
    got = cmicalc([1, 2, 5, 10])
    assert np.isnan(got[0])
    assert got[1] == pytest.approx(0.564189583547756, abs=1e-15)
    assert got[2] == pytest.approx(0.840748682459689, abs=1e-15)
    assert got[3] == pytest.approx(0.922745608053087, abs=1e-15)
    assert cmicalc([5, 10], method="approx").tolist() == pytest.approx(
        [0.842105263157895, 0.923076923076923])
    with pytest.raises(ValueError):
        cmicalc([5], method="none")


def test_replmiss_matches_r():
    assert replmiss([1, None, 3], [9, 2, 8]).tolist() == [1, 2, 3]
    assert replmiss([1, None, None], [9, 2, 8]).tolist() == [1, 2, 8]
    assert replmiss([1, 2, 3], [9, 2, 8]).tolist() == [1, 2, 3]


def test_blsplit_matches_r():
    m = np.arange(1, 17).reshape(4, 4, order="F").astype(float)
    m = m + m.T
    sp = blsplit(m, cluster=[1, 1, 2, 2])
    assert np.allclose(sp[1], [[2, 7], [7, 12]])
    assert np.allclose(sp[2], [[22, 27], [27, 32]])


def test_contrmat_matches_r():
    d = {"t1": ["A", "A", "B"], "t2": ["B", "C", "C"], "x": [1, 2, 3]}
    out = contrmat(d, "t1", "t2")
    assert out["A"] == [1.0, 1.0, 0.0]
    assert out["B"] == [-1.0, 0.0, 1.0]
    assert out["C"] == [0.0, -1.0, -1.0]


def test_rcalc_matches_r():
    r = np.array([[1, 0.3, 0.4], [0.3, 1, 0.5], [0.4, 0.5, 1]])
    res = rcalc(r, ni=100)
    assert res.dat["var1"] == ["x1", "x1", "x2"]
    assert res.dat["var2"] == ["x2", "x3", "x3"]
    assert res.dat["yi"] == pytest.approx([0.3, 0.4, 0.5])
    assert res.V[0] == pytest.approx(
        [0.008364646464646, 0.003484848484848, 0.002287878787879], abs=1e-13)
    resz = rcalc(r, ni=100, rtoz=True)
    assert resz.dat["yi"][0] == pytest.approx(0.309519604203112, abs=1e-14)
    assert np.diag(resz.V) == pytest.approx([0.0103] * 3, abs=5e-5)


def test_pairmat_matches_r():
    res = rma([0.1, 0.2, 0.3, 0.4], vi=[0.01, 0.02, 0.03, 0.04],
              mods=[1.0, 2.0, 3.0, 4.0])
    pm = pairmat(res, btt=[1, 2])
    assert pm["matrix"].shape == (1, 2)
    assert pm["matrix"].tolist() == [[-1.0, 1.0]]
    assert len(pm["names"]) == 1 and "-" in pm["names"][0]


# ---------------------------------------------------------------------------
# fit statistics / BLUPs (dat.bcg, metafor 5.2-1)
# ---------------------------------------------------------------------------

BCG_YI = [-0.889311333920205449, -1.585388657201430629, -1.348073148299693269,
          -1.441551190021305384, -0.217547322211295580, -0.786115585818863982,
          -1.620898223598391752, 0.011952333523840508, -0.469417648738149396,
          -1.371344803472784424, -0.339358828338390595, 0.445913400571378737,
          -0.017313948216879815]
BCG_VI = [0.3255847650039613295, 0.1945811213981438470, 0.4153679653679653860,
          0.0200100319022475728, 0.0512101721696308632, 0.0069056184559087574,
          0.2230172475723151693, 0.0039615792978177295, 0.0564342104632489655,
          0.0730247936130289099, 0.0124122139715597199, 0.5325058452001527609,
          0.0714046596839862935]


def test_fitstats_bcg():
    res = rma(BCG_YI, vi=BCG_VI, method="REML")
    fs = fitstats(res)
    assert fs.REML["ll"] == pytest.approx(-12.2023714154937, abs=1e-11)
    assert fs.REML["dev"] == pytest.approx(24.4047428309873, abs=1e-11)
    assert fs.REML["AIC"] == pytest.approx(28.4047428309873, abs=1e-11)
    assert fs.REML["BIC"] == pytest.approx(29.3745561305633, abs=1e-11)
    assert fs.REML["AICc"] == pytest.approx(29.7380761643206, abs=1e-11)
    assert fs.ML["ll"] == pytest.approx(-12.6877717498605, abs=1e-11)
    assert fs.ML["dev"] == pytest.approx(37.1614117732324, abs=1e-11)
    assert fs.ML["AIC"] == pytest.approx(29.3755434997210, abs=1e-11)
    assert fs.ML["BIC"] == pytest.approx(30.5054422146441, abs=1e-11)
    assert fs.ML["AICc"] == pytest.approx(30.5755434997210, abs=1e-11)


def test_aic_bic_se_dfresidual_bcg():
    res = rma(BCG_YI, vi=BCG_VI, method="REML")
    assert aic_rma(res) == pytest.approx(28.4047428309873, abs=1e-11)
    assert bic_rma(res) == pytest.approx(29.3745561305633, abs=1e-11)
    assert aic_rma(res, correct=True) == pytest.approx(29.7380761643206,
                                                       abs=1e-11)
    assert se_rma(res)["intrcpt"] == pytest.approx(0.179781531780361, abs=1e-13)
    assert df_residual_rma(res) == 12
    assert model_matrix_rma(res)["matrix"].shape == (13, 1)


def test_blup_bcg():
    res = rma(BCG_YI, vi=BCG_VI, method="REML")
    b = blup(res)
    ref_pred = [-0.80023357459396727, -1.25170602725994096, -0.98690313052621537,
                -1.39789770575490846, -0.28737979434894345,
                -0.78457153410704583, -1.24396351544210759,
                0.00287924975981697, -0.50683635337311050,
                -1.24717303898150678, -0.35365839777827990,
                -0.28473365815812113, -0.14674304818198738]
    assert b.pred == pytest.approx(ref_pred, abs=1e-12)


def test_estimates_and_forest_args():
    m = metagen([-0.7, -0.5, -0.9], [0.2, 0.25, 0.3], sm="RR")
    est = estimates(m)
    assert len(est.rows) >= 5           # 3 studies + common + random
    assert est.rows[0]["type"] == "study"
    assert any(r["type"] == "common" for r in est.rows)
    assert any(r["type"] == "random" for r in est.rows)
    fa = forest_args(random=True, common=True, prediction=False)
    assert fa["labels"][0] == "Random effects model"
    assert fa["labels"][1] == "Common effect model"


# ---------------------------------------------------------------------------
# netmatrix / netbind / netcomparison (R netmeta 3.7-0)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SENN_CSV.exists(), reason="Senn 2013 benchmark CSV not present")
def test_netmatrix_senn():
    net = _senn_net()
    cov = [(i + 1) / 10 for i in range(len(net.TE_raw))]
    nm = netmatrix(net, cov)
    assert nm.func == "mode"
    pos = {t: i for i, t in enumerate(nm.trts)}
    assert nm.matrix[pos["acar"], pos["metf"]] == pytest.approx(0.3)
    assert nm.matrix[pos["acar"], pos["plac"]] == pytest.approx(1.6)
    assert nm.matrix[pos["acar"], pos["sulf"]] == pytest.approx(1.5)
    assert np.isnan(nm.matrix[pos["acar"], pos["benf"]])

    nmm = netmatrix(net, cov, func="mean")
    assert nmm.matrix[pos["acar"], pos["plac"]] == pytest.approx(2.2)
    assert nmm.matrix[pos["acar"], pos["sulf"]] == pytest.approx(1.5)


@pytest.mark.skipif(not SENN_CSV.exists(), reason="Senn 2013 benchmark CSV not present")
def test_netbind_senn():
    net = _senn_net()
    nb = netbind(net, name="Senn2013", reference_group="plac")
    assert len(nb.common) == 10          # one row per treatment (incl. ref)
    row = nb.common[0]
    assert row["name"] == "Senn2013" and row["treat"] == "acar"
    assert row["TE"] == pytest.approx(-0.827368972010636, abs=1e-11)
    assert row["seTE"] == pytest.approx(0.108520845439489, abs=1e-11)
    assert row["lower"] == pytest.approx(-1.04006592064387, abs=1e-11)
    assert row["upper"] == pytest.approx(-0.614672023377399, abs=1e-11)
    assert row["statistic"] == pytest.approx(-7.62405571629989, abs=1e-11)
    assert nb.fixed is nb.common


def test_netcomparison_cnma():
    te = [0.5, 0.3, 0.4, 0.1, 0.2]
    se = [0.2, 0.25, 0.2, 0.15, 0.2]
    t1 = ["A", "A", "A", "A", "B"]
    t2 = ["B", "B", "A+B", "A+B", "A+B"]
    nc = discomb(te, se, t1, t2, sm="MD", sep_comps="+")
    assert nc.trts == ["A", "A+B", "B"]
    ncp = netcomparison(nc, treat1="A+B", treat2="A")
    assert ncp.comparison == ["B"]
    assert ncp.TE_common[0] == pytest.approx(-0.283655274888559, abs=1e-12)
    assert ncp.seTE_common[0] == pytest.approx(0.108481344388046, abs=1e-12)
    assert ncp.TE_random[0] == pytest.approx(-0.300393401418644, abs=1e-12)
    assert ncp.seTE_random[0] == pytest.approx(0.130177853137663, abs=1e-12)


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------

RM5_XML = """<?xml version="1.0" encoding="UTF-8"?>
<COCHRANE_REVIEW>
  <TITLE>Test review</TITLE>
  <DATA_AND_ANALYSES>
    <COMPARISON>
      <NAME>Drug A vs placebo</NAME>
      <OUTCOME>
        <EFFECT_MEASURE>2</EFFECT_MEASURE>
        <GROUP LABEL="Drug A">
          <DATA>
            <STUDY>Smith 2001</STUDY><E1>10</E1><N1>50</N1>
            <E2>20</E2><N2>50</N2><METHOD>2</METHOD>
          </DATA>
          <DATA>
            <STUDY>Jones 2003</STUDY><E1>15</E1><N1>60</N1>
            <E2>25</E2><N2>60</N2><METHOD>2</METHOD>
          </DATA>
        </GROUP>
      </OUTCOME>
    </COMPARISON>
  </DATA_AND_ANALYSES>
</COCHRANE_REVIEW>
"""


def test_read_rm5(tmp_path):
    import gzip

    p = tmp_path / "r.rm5"
    p.write_bytes(gzip.compress(RM5_XML.encode("utf-8")))
    r = read_rm5(str(p))
    assert r.title == "Test review"
    assert len(r.data) == 2
    assert r.data[0]["type"] == "D"
    assert r.data[0]["sm"] == "RR"
    assert r.data[0]["studlab"] == "Smith 2001"
    assert r.data[0]["event.e"] == 10.0
    assert r.data[0]["n.c"] == 50.0


def test_read_cdir(tmp_path):
    txt = ("@title\tTest review\n"
           "@comparison\tDrug A vs placebo\n"
           "@outcome\tMortality\n"
           "@type\tD\n"
           "@measure\tRR\n"
           "@method\tInverse\n"
           "Study\tEvent1\tTotal1\tEvent2\tTotal2\n"
           "Smith 2001\t10\t50\t20\t50\n"
           "Jones 2003\t15\t60\t25\t60\n")
    p = tmp_path / "r.cdir"
    p.write_text(txt, encoding="utf-8")
    r = read_cdir(str(p))
    assert r.title == "Test review"
    assert len(r.data) == 2
    assert r.data[0]["sm"] == "RR"
    assert r.data[0]["event.c"] == 20.0


def test_read_mtv(tmp_path):
    txt = ("id,comp,outcome,treat,n,event\n"
           "Smith 2001,1,1,A,50,10\n"
           "Smith 2001,1,1,P,50,20\n"
           "Jones 2003,1,1,A,60,15\n"
           "Jones 2003,1,1,P,60,25\n")
    p = tmp_path / "r.mtv"
    p.write_text(txt, encoding="utf-8")
    r = read_mtv(str(p))
    assert len(r.data) == 2
    assert r.data[0]["studlab"] == "Smith 2001"
    assert r.data[0]["event.e"] == 10.0
    assert r.data[0]["event.c"] == 20.0


def test_readers_feed_metacr(tmp_path):
    p = tmp_path / "r.mtv"
    p.write_text("id,comp,outcome,treat,n,event\n"
                 "S1,1,1,A,50,10\nS1,1,1,P,50,20\n"
                 "S2,1,1,A,60,15\nS2,1,1,P,60,25\n", encoding="utf-8")
    r = read_mtv(str(p))
    mc = metacr(r.data)
    assert mc.n_studies == 2
    assert mc.meta.fixed.pooled_effect == pytest.approx(0.5578003108649138,
                                                        abs=1e-12)


# ---------------------------------------------------------------------------
# foreign-object converters
# ---------------------------------------------------------------------------

def test_gemtc2netmeta():
    g = {"network": {"data.re": {"study": ["S1", "S1", "S2", "S2"],
                                 "treatment": ["A", "B", "A", "C"],
                                 "diff": [0.0, 0.5, 0.0, 0.3],
                                 "std.err": [0.0, 0.2, 0.0, 0.15]}}}
    r = gemtc2netmeta(g, sm="MD")
    assert r.trts == ["A", "B", "C"]
    assert r.k == 2 and r.m == 2 and r.n == 3


def test_crossnma2netmeta():
    c = {"data": {"studlab": ["S1", "S2"], "treat1": ["A", "A"],
                  "treat2": ["B", "C"], "TE": [0.5, 0.3],
                  "seTE": [0.2, 0.15]}, "sm": "MD"}
    r = crossnma2netmeta(c)
    assert r.trts == ["A", "B", "C"]
    assert r.k == 2 and r.m == 2


# ---------------------------------------------------------------------------
# heatplot / regplot
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SENN_CSV.exists(), reason="Senn 2013 benchmark CSV not present")
def test_heatplot_senn():
    import matplotlib.pyplot as plt

    net = _senn_net()
    npth = netpath(net, "metf", "plac")
    pr = heatplot(npth)
    assert pr.ax is not None
    plt.close(pr.figure)


def test_regplot_smoke():
    import matplotlib.pyplot as plt

    mod = [44.0, 55.0, 42.0, 52.0, 13.0, 44.0, 19.0, 13.0, 27.0, 42.0,
           18.0, 33.0, 33.0]
    res = rma(BCG_YI, vi=BCG_VI, mods=mod, method="REML")
    pr = regplot(res, mod=mod, ci=True, pi=True, perm=20, seed=0)
    assert pr.ax is not None
    plt.close(pr.figure)


# ---------------------------------------------------------------------------
# remaining exports: escale binds, cidprop, longarm, reporter, addpoly, rob
# ---------------------------------------------------------------------------

def test_aggregate_escalc_matches_r():
    d = escalc(measure="RR", ai=[5, 8, 3, 9], bi=[45, 42, 47, 41],
               ci=[10, 12, 6, 11], di=[40, 38, 44, 39])
    agg = aggregate_escalc(d, cluster=["A", "A", "B", "B"], rho=0.5)
    assert agg.yi == pytest.approx([-0.489134012684736, -0.236680033974190],
                                   abs=1e-14)
    assert agg.vi == pytest.approx([0.149797956755570, 0.160154212553206],
                                   abs=1e-14)


def test_cbind_rbind_escalc():
    a = escalc(measure="RR", ai=[5, 8], bi=[45, 42], ci=[10, 12], di=[40, 38])
    b = escalc(measure="RR", ai=[3, 4], bi=[47, 46], ci=[6, 7], di=[44, 43])
    cb = cbind_escalc(a, b)
    assert set(cb["yi"]) == {"escalc1", "escalc2"}
    rb, slab = rbind_escalc(a, a)
    assert len(rb.yi) == 4
    assert len(slab) == 4
    with pytest.raises(ValueError):
        cbind_escalc(a, escalc(measure="RR", ai=[3], bi=[47], ci=[6], di=[44]))


def test_cidprop_matches_r():
    m = metagen([-0.7, -0.5, -0.9, -0.4, -0.6, -0.8],
                [0.2, 0.25, 0.3, 0.18, 0.22, 0.28], sm="RR",
                method_tau="REML", null_effect=1.0)
    r = cidprop(m, cid=[0.8, 0.9])
    assert r.prop_cid_below_null == pytest.approx(
        [0.995423736589206, 0.998520621416962], abs=1e-13)
    assert r.prop_cid_above_null == pytest.approx(
        [0.000300759481423432, 0.000146121868124063], abs=1e-15)
    assert r.prop_within_cid == pytest.approx(0.00117861910161472, abs=1e-15)


def test_longarm_binary_matches_r():
    la = longarm(treat1=["A", "B"], treat2=["B", "C"],
                 event1=[10, 12], n1=[50, 60], event2=[20, 15], n2=[50, 60],
                 studlab=["S1", "S2"])
    assert la.type == "binary"
    got = [(r["studlab"], r["treat"], r["n"], r["events"], r["nonevents"])
           for r in la.rows]
    assert got == [("S1", "A", 50.0, 10.0, 40.0),
                   ("S1", "B", 50.0, 20.0, 30.0),
                   ("S2", "B", 60.0, 12.0, 48.0),
                   ("S2", "C", 60.0, 15.0, 45.0)]


def test_longarm_continuous_matches_r():
    la = longarm(treat1=["A"], treat2=["B"], n1=50, mean1=10, sd1=2,
                 n2=50, mean2=12, sd2=3, studlab="S1")
    assert la.type == "continuous"
    assert [(r["treat"], r["n"], r["mean"], r["sd"]) for r in la.rows] == [
        ("A", 50.0, 10.0, 2.0), ("B", 50.0, 12.0, 3.0)]


def test_cilayout_and_format_ci():
    try:
        cilayout("[", ", ")
        assert format_ci(0.1234, 0.5678) == "[0.123, 0.568]"
        cilayout("(", "; ", lower_blank=True)
        assert format_ci(0.1, 0.5) == "(; 0.500)"
    finally:
        cilayout("[", ", ")
    with pytest.raises(ValueError):
        cilayout("]")


def test_forest_dims():
    d = forest_dims(k=10)
    assert d["units"] == "in" and d["height"] > 0 and d["width"] > 0
    dc = forest_dims(k=10, units="cm")
    assert dc["height"] == pytest.approx(d["height"] * 2.54, rel=1e-12)


def test_reporter_and_emmprep():
    res = rma(BCG_YI, vi=BCG_VI, method="REML")
    rep = reporter(res, title="BCG")
    assert "BCG" in rep.text
    assert "tau^2" in rep.text
    em = emmprep(res)
    assert em["coef"] == {"(Intercept)": pytest.approx(-0.714532348365102,
                                                       abs=1e-12)}
    assert em["sigma"] == pytest.approx(math.sqrt(res.tau2), abs=1e-12)


def test_addpoly_and_rob(tmp_path):
    import matplotlib.pyplot as plt

    pr = addpoly(yi=0.5, ci_lb=0.2, ci_ub=0.8, rows=1, mlab="Pooled")
    assert pr.ax is not None
    plt.close(pr.figure)

    out = tmp_path / "rob.png"
    pr2 = rob(["S1", "S2"], {"D1": ["low", "high"], "D2": ["unclear", "low"]},
              filename=str(out))
    assert out.exists()
    plt.close(pr2.figure)


# ---------------------------------------------------------------------------
# S3-style generics
# ---------------------------------------------------------------------------

def test_s3_value_accessors_bcg():
    res = rma(BCG_YI, vi=BCG_VI, method="REML")
    assert coef(res)["intrcpt"] == pytest.approx(-0.714532348365102, abs=1e-12)
    assert vcov(res).shape == (1, 1)
    assert nobs(res) == 13
    assert df_residual(res) == 12
    assert formula(res) == "~ 1"
    assert logLik(res) == pytest.approx(-12.2023714154937, abs=1e-11)
    assert deviance(res) == pytest.approx(24.4047428309873, abs=1e-11)
    assert AIC(res) == pytest.approx(28.4047428309873, abs=1e-11)
    assert BIC(res) == pytest.approx(29.3745561305633, abs=1e-11)
    assert fitted(res)[0] == pytest.approx(res.b[0], abs=1e-12)
    assert residuals(res)[0] == pytest.approx(BCG_YI[0] - res.b[0], abs=1e-12)
    assert len(s3_weights(res)) == 13
    assert len(hatvalues_rma(res)) == 13
    assert len(rstandard_rma(res)) == 13
    assert len(ranef_rma(res).pred) == 13


def test_s3_confint_predict_simulate():
    res = rma(BCG_YI, vi=BCG_VI, method="REML")
    ci = confint(res)
    assert ci["ci_lb"] == pytest.approx(0.119695312816638, abs=1e-11)
    assert ci["ci_ub"] == pytest.approx(1.111486262215440, abs=1e-11)
    pr = predict(res)
    assert pr.pred == pytest.approx(-0.714532348365102, abs=1e-12)
    sim = simulate(res, nsim=3, seed=7)
    assert np.asarray(sim).shape == (13, 3)


def test_s3_summary_metagen():
    m = metagen([-0.7, -0.5, -0.9], [0.2, 0.25, 0.3], sm="RR")
    s = summary(m)
    assert s["class"] == "MetagenResult"
    assert s["common"]["estimate"] == pytest.approx(m.TE_common, abs=1e-12)
    assert "heterogeneity" in s
    txt = summary_text(m)
    assert "Random effects model" in txt and "Heterogeneity" in txt


def test_s3_plot_dispatch():
    import matplotlib.pyplot as plt

    res = rma(BCG_YI, vi=BCG_VI, method="REML")
    for which in ("forest", "funnel", "radial", "qqnorm"):
        pr = plot(res, which=which)
        ax = getattr(pr, "ax", pr)
        assert ax is not None
        plt.close(getattr(pr, "figure", ax.figure))
    with pytest.raises(ValueError):
        plot(res, which="nope")


# ---------------------------------------------------------------------------
# netrank (netmeta::netrank) — reference values from R 4.6.1 / netmeta 3.7-0
# ---------------------------------------------------------------------------

NR_TE = [0.1, -0.2, 0.05, -0.15, 0.3, 0.25]
NR_SE = [0.1] * 6
NR_T1 = ["A", "B", "C", "A", "B", "C"]
NR_T2 = ["B", "C", "A", "D", "D", "D"]
NR_STUD = [f"s{i}" for i in range(6)]

# netrank(nm)$ranking.common  (P-scores despite the R field name)
NR_RANKING_COMMON = [
    0.5885459838316730, 0.4673121575230837, 0.0187429069758184, 0.9253989516694250,
]
# netrank(nm)$ranking.random
NR_RANKING_RANDOM = [
    0.574122044110847, 0.493908926264800, 0.131803985820764, 0.800165043803589,
]


def test_netrank_r_field_names_and_values():
    nm = netmeta(TE=NR_TE, seTE=NR_SE, treat1=NR_T1, treat2=NR_T2,
                 studlab=NR_STUD, sm="MD", reference_group="A")
    nr = netrank(nm, small_values="desirable")

    assert nr.trts == ["A", "B", "C", "D"]
    assert list(nr.ranking_common) == pytest.approx(NR_RANKING_COMMON, abs=1e-12)
    assert list(nr.ranking_random) == pytest.approx(NR_RANKING_RANDOM, abs=1e-12)

    # R: ranking.fixed is an alias of ranking.common; Pscore.* alias ranking.*
    assert list(nr.ranking_fixed) == pytest.approx(NR_RANKING_COMMON, abs=1e-12)
    assert list(nr.Pscore_fixed) == pytest.approx(NR_RANKING_COMMON, abs=1e-12)
    assert list(nr.Pscore_random) == pytest.approx(NR_RANKING_RANDOM, abs=1e-12)

    # R: Pmatrix.common[1, 2] / Pmatrix.random[1, 2]
    assert nr.Pmatrix_common[0, 1] == pytest.approx(0.6381631950841189, abs=1e-12)
    assert nr.Pmatrix_random[0, 1] == pytest.approx(0.569445863886284, abs=1e-12)
    assert np.isnan(nr.Pmatrix_random[0, 0])

    # Backwards-compatible spellings hold the same numbers.
    assert list(nr.pscore_common) == pytest.approx(NR_RANKING_COMMON, abs=1e-12)
    assert list(nr.pscore_random) == pytest.approx(NR_RANKING_RANDOM, abs=1e-12)

    # Misc netmeta 3.7-0 fields.
    assert nr.method == "P-score"
    assert nr.nsim is None
    assert nr.small_values == "desirable"
    assert nr.common is True and nr.random is True

    d = nr.to_dict()
    assert d["method"] == "P-score"
    assert d["ranking_random"] == pytest.approx(NR_RANKING_RANDOM, abs=1e-12)
    assert d["Pmatrix_common"][0][1] == pytest.approx(0.6381631950841189, abs=1e-12)
