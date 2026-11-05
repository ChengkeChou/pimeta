"""Discovery layer: what the package can do, and how far each part is trusted.

``pimeta`` exposes several hundred names.  A user who knows the statistics but
not this package has no way to learn, from ``dir(pimeta)``, which entry point
fits their data, what its assumptions are, or -- the question that matters most
-- whether its output has been *checked*.  This module answers those three
questions from one table.

The intended use is::

    >>> import pimeta as ma
    >>> ma.list_models()
    ['metabin', 'metacont', 'netmeta', 'rma', 'rma.glmm', 'rma.mv', ...]
    >>> info = ma.describe("predict_region")
    >>> info["validation_status"]
    'unvalidated'
    >>> info["limitations"]
    ['...measured joint coverage 66.3% at k=10 against a claimed 95%...']

Design rules, because a catalogue of claims is worse than none if it drifts:

* **Every entry is a statement this repository can back with a file.**  The
  ``evidence`` key names the report or test that produced the number quoted in
  ``validation_status``.  Nothing here is a guess about what a method "should"
  do.
* **The status vocabulary is fixed and short**: ``validated`` (the quantity was
  measured and the claim held), ``conditional`` (measured, and it holds within
  stated conditions), ``measured-shortfall`` (measured, and it does *not* meet
  the nominal claim -- the numbers are quoted), ``parity`` (implementation
  agrees with a reference implementation, which says nothing about statistical
  calibration), ``unvalidated`` (no measurement exists in this repository).
* **A proportion is quoted with its Monte-Carlo error** wherever one is quoted,
  and never as a bare percentage.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "describe",
    "list_diagnostics",
    "list_effect_sizes",
    "list_estimators",
    "list_models",
]

#: The four statuses, in increasing order of how much is known.  Kept as a
#: tuple so that ``describe`` can refuse a status outside it.
STATUSES = ("unvalidated", "parity", "conditional", "validated",
            "measured-shortfall")

_REQUIRED = ("kind", "entry_point", "assumptions", "methods", "default",
             "validation_status", "limitations", "references", "evidence")


def _entry(**kwargs: Any) -> dict[str, Any]:
    missing = [k for k in _REQUIRED if k not in kwargs]
    if missing:
        raise ValueError(f"catalogue entry is missing {missing}")
    if kwargs["validation_status"] not in STATUSES:
        raise ValueError(
            f"validation_status must be one of {STATUSES}, "
            f"got {kwargs['validation_status']!r}")
    return kwargs


_CATALOGUE: dict[str, dict[str, Any]] = {
    "rma": _entry(
        kind="model",
        entry_point="rma",
        summary="Univariate random/fixed-effects meta-analysis (the workhorse).",
        assumptions=[
            "independent sampling errors with known variances vi",
            "normal random effect theta_i ~ N(mu, tau^2)",
        ],
        methods=["FE", "DL", "HE", "HS", "SJ", "PM", "EB", "ML", "REML",
                 "GENQ", "PMM", "GENQM"],
        default="REML with a z test; test='knha' for the Hartung-Knapp interval",
        validation_status="parity",
        limitations=[
            "The Wald/knha intervals are not calibrated at small k: the "
            "coverage simulation for this estimator has not been run in this "
            "repository, so no claim is made either way.",
            "tau^2 is bounded below by zero; a fit on the boundary is an "
            "estimate, not a failure, but the interval is then conditional on "
            "tau^2 = 0.",
        ],
        references=[
            "Viechtbauer (2005) J Stat Softw 36(3) - the model and REML",
            "Hartung & Knapp (2001) Stat Med 20 - the knha interval",
            "DerSimonian & Laird (1986) Control Clin Trials 7 - DL",
        ],
        evidence="tests/test_pimeta.py, tests/test_coverage_baselines.py",
    ),
    "rma.mv": _entry(
        kind="model",
        entry_point="rma_mv",
        summary="Multilevel / multivariate model with a structured Sigma.",
        assumptions=[
            "rows of the same study share a random effect with covariance Sigma",
            "sampling covariance V is known",
        ],
        methods=["CS", "HCS", "HAR", "UN", "DIAG", "GEN"],
        default="struct='CS' for one random term; 'UN' for multivariate",
        validation_status="conditional",
        limitations=[
            "UN with three or more inner levels: measured over the D grid "
            "(8 cells, 250 replicates), the share of fits stopping below the "
            "independently computed maximum is 0.4-6.0% after 1.1.7, and the "
            "largest gap is 0.000 in every J=3 cell; at k=30, J=5 the share is "
            "3.0% and 6.0% with worst gaps of 0.041 and 0.066 log-likelihood "
            "units, which are flat ridges rather than wrong basins.",
            "A UN fit can land on the cone boundary, where Sigma is singular "
            "and the likelihood is flat; two correct implementations may stop "
            "at different points on that ridge.",
        ],
        references=[
            "Viechtbauer (2010) J Stat Softw 36(3) - rma.mv",
            "Berkey et al. (1998) Stat Med 17 - the multivariate model",
        ],
        evidence="metabench/reports/D_covariance.md; families D1-D8",
    ),
    "predict_region": _entry(
        kind="inference",
        entry_point="predict_region",
        summary="Joint confidence region for the mean effect vector.",
        assumptions=[
            "Sigma-hat is treated as known",
            "the region is an ellipsoid from a chi-square (or F) reference",
        ],
        methods=["z", "t"],
        default="test='z' (chi-square reference)",
        validation_status="measured-shortfall",
        limitations=[
            "Measured joint coverage of the *prediction* region is 66.3% at "
            "k=10 and 87.0% at k=50 against a claimed 95% (chi-square "
            "reference, 6 cells, MCSE about 0.05 at n=250); the deficit shrinks "
            "like 1/k. The region for the *mean* is calibrated (93.3-99.3%), "
            "so the cause is the prediction step, not the estimate.",
            "Do not read the nominal level as coverage at small k.",
        ],
        references=[
            "Riley, Higgins & Deeks (2011) Stat Med 30 - multivariate prediction",
            "Cheung (2015) Res Synth Methods 6 - multivariate meta-analysis",
        ],
        evidence="metabench/reports/B_joint_region.md, tests/reference/coverage_predict_region.json",
    ),
    "ipd_two_stage_curve": _entry(
        kind="model",
        entry_point="ipd_two_stage_curve",
        summary="Two-stage IPD dose-response curve with a structured "
                  "between-study covariance.",
        assumptions=[
            "study-level curves are estimated first, then pooled",
            "the between-study covariance of the basis coefficients is "
            "structured (HCS by default) or unstructured",
        ],
        methods=["HCS", "UN", "CS"],
        default="HCS with lkj_eta=1 (unpenalised)",
        validation_status="conditional",
        limitations=[
            "With 12 studies and 10 variance parameters the between-study "
            "covariance is weakly identified: measured over 12 cohorts, the "
            "unpenalised fit puts a correlation on +/-1 in 2 of 12 and "
            "lkj_eta=2 in 0 of 12, while the penalty moves the boundary into "
            "the variances in all 12. lkj_eta is a penalty, not a prior.",
            "Edge coverage of the band was measured at 0.733 with struct='HCS', "
            "which is why the curve is reported with its prediction band and "
            "not as a point estimate.",
        ],
        references=[
            "Ruecker et al. (2014) the dose-response model",
            "tests/test_ipd_curve.py documents the measured boundary rates",
        ],
        evidence="tests/test_coverage_reproduction.py; see also tools/coverage_ipd_curve.py",
    ),
    "netmeta": _entry(
        kind="model",
        entry_point="netmeta",
        summary="Frequentist network meta-analysis.",
        assumptions=[
            "consistency between direct and indirect evidence",
            "a common between-study variance (or a design-specific one)",
        ],
        methods=["DL", "REML", "ML", "SJ", "PM", "FE"],
        default="random effects with DL tau^2, consistency model",
        validation_status="parity",
        limitations=[
            "Inconsistency tests have low power in small networks; a "
            "non-significant Q_inc is not evidence of consistency.",
            "The ranking probabilities are a function of the fitted model, not "
            "an independent source of evidence.",
        ],
        references=[
            "Ruecker (2012) Res Synth Methods 3 - graph-theoretical approach",
            "Krahn, Binder & Koenig (2013) - Q decomposition",
            "Balduzzi et al. (2023) J Stat Softw 106(4) - netmeta",
        ],
        evidence="tests/test_netmeta_api.py, tests/test_nma_bayes.py",
    ),
    "rma.glmm": _entry(
        kind="model",
        entry_point="rma_glmm",
        summary="Mixed-effects logistic/Poisson models for 2x2 and single-arm "
                  "data, with adaptive Gauss-Hermite quadrature.",
        assumptions=["binomial or Poisson likelihood", "normal random effect"],
        methods=["ML", "penalised likelihood (Firth-type) for separation"],
        default="ML with nAGQ=7",
        validation_status="parity",
        limitations=[
            "With few studies per arm the quadrature can be inaccurate; the "
            "penalised likelihood is available for separation but has not been "
            "calibrated by simulation here.",
        ],
        references=[
            "Viechtbauer (2010) - rma.glmm",
            "Firth (1993) Biometrika 80 - penalised likelihood",
        ],
        evidence="tests/test_coverage_gaps.py",
    ),
    "dose-response": _entry(
        kind="model",
        entry_point="dosres",
        summary="Nonlinear dose-response meta-regression on a spline basis.",
        assumptions=["the dose-response curve is smooth", "normal random effect"],
        methods=["ML", "REML"],
        default="REML with a restricted cubic spline basis",
        validation_status="unvalidated",
        limitations=[
            "The degrees-of-freedom approximation for the non-linearity test "
            "has not been assessed by simulation in this repository; treat the "
            "p-value as approximate.",
        ],
        references=["Desquilbet & Mariotti (2010) Stat Med 29 - spline basis"],
        evidence="(no measurement yet)",
    ),
    "metabin": _entry(
        kind="model",
        entry_point="metabin",
        summary="Binary-outcome meta-analysis with continuity corrections.",
        assumptions=["2x2 tables per study"],
        methods=["MH", "Inverse", "Peto", "GLMM", "LRP"],
        default="MH with the 0.5 correction where a cell is zero",
        validation_status="parity",
        limitations=[
            "The continuity correction is a choice, not a neutral default: on "
            "a diagnostic-accuracy dataset measured here it was applied to all "
            "14 studies because 2 needed it; pass correction_control='single' "
            "to correct only those.",
        ],
        references=["Viechtbauer (2010)", "Schwarzer et al. (2015) - meta"],
        evidence="tests/test_coverage_gaps.py, tests/test_mh_peto_methods.py",
    ),
}

_ESTIMATORS = {
    "FE": "fixed effects, tau^2 = 0",
    "DL": "DerSimonian & Laird (1986)",
    "HE": "Hedges (1983), unbiased under normality",
    "HS": "Hunter & Schmidt (1990)",
    "SJ": "Sidik & Jonkman (2005)",
    "PM": "Paule & Mandel (1982)",
    "EB": "Morris (1983), empirical Bayes",
    "ML": "maximum likelihood",
    "REML": "restricted maximum likelihood",
    "GENQ": "generalised Q-profile",
    "PMM": "profile-method-of-moments",
    "GENQM": "generalised Q median",
}

_EFFECT_SIZES = {
    "mean difference": "MD, SMD (Hedges' g), ROM",
    "binary": "RR, OR, RD, PETO, ASD, with continuity correction",
    "time-to-event": "HR, log-HR, and time-varying HR (tvhr)",
    "correlation": "ZCOR, COR",
    "proportion": "PLO, PR, PFT, PRAW, PAS, PHS",
    "incidence": "IRR, IRLN, IRFT, IRP",
    "diagnostic": "sens/spec/PPV/NPV via escalc_dta",
}

_DIAGNOSTICS = {
    "diagnostics": "per-fit record: converged, boundary hits, matrix checks",
    "diagnose": "structured Diagnosis objects with a severity",
    "identifiability": "boundary and identification report for an IPD curve",
    "confint_tau2": "profile-likelihood interval for tau^2",
    "checks": "the warnings a fit emits, with the measurement behind each",
}


def list_models() -> list[str]:
    """Names accepted by :func:`describe`."""
    return sorted(_CATALOGUE)


def describe(name: str) -> dict[str, Any]:
    """Everything this repository can say about one entry point.

    Returns a dict with ``kind``, ``entry_point``, ``summary``,
    ``assumptions``, ``methods``, ``default``, ``validation_status``,
    ``limitations``, ``references`` and ``evidence``.  The status is one of
    ``validated``, ``conditional``, ``parity`` or ``unvalidated``; see the
    module docstring for what each means, and ``evidence`` for the file that
    backs it.
    """
    key = name.strip()
    if key not in _CATALOGUE:
        known = ", ".join(list_models())
        raise KeyError(f"unknown model {name!r}; known: {known}")
    return {k: (list(v) if isinstance(v, list) else v)
            for k, v in _CATALOGUE[key].items()}


def list_estimators() -> dict[str, str]:
    """The tau^2 estimators, by the name accepted in ``method=``."""
    return dict(_ESTIMATORS)


def list_effect_sizes() -> dict[str, str]:
    """The effect-size families, and the measures in each."""
    return dict(_EFFECT_SIZES)


def list_diagnostics() -> dict[str, str]:
    """The diagnostics a fit can report, and what each answers."""
    return dict(_DIAGNOSTICS)
