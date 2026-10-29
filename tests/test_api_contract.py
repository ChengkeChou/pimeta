"""The public result types, asserted where the documentation states them.

1.1.4 shipped a README that called ``result.b`` an ``np.ndarray`` while the
code returned ``list[float]``: ``result.b.tolist()`` raised ``AttributeError``
and ``result.b @ x`` raised ``TypeError`` for every user who trusted the table.
The generated API reference was right, which is exactly why the prose drifted --
nothing compared the two.

This module is the start of the public-API contract: the fields whose type the
README promises are checked here, so the promise cannot rot silently.  It is
deliberately about *types and shapes*, not values; the numbers are pinned by
the reference tests.
"""

import json
import math

import numpy as np
import pytest

import pimeta as ma

YI = [0.20, 0.40, 0.10, 0.35]
VI = [0.04, 0.05, 0.03, 0.02]

#: Fields the README calls vectors: always a list, even for one coefficient.
VECTOR_FIELDS = ("b", "se", "zval", "pval", "ci_lb", "ci_ub",
                 "yi", "vi", "weights", "weights_pct", "fitted", "residuals")


@pytest.fixture(scope="module")
def fit():
    return ma.rma(YI, vi=VI, method="REML")


def test_vector_fields_are_lists_of_float(fit):
    for name in VECTOR_FIELDS:
        value = getattr(fit, name)
        assert isinstance(value, list), f"{name} is {type(value).__name__}"
        assert value, f"{name} is empty"
        assert all(isinstance(x, float) for x in value), name


def test_matrix_fields_are_arrays(fit):
    assert isinstance(fit.vb, np.ndarray)
    assert isinstance(fit.X, np.ndarray)
    p = len(fit.b)
    assert fit.vb.shape == (p, p)
    assert fit.X.ndim == 2 and fit.X.shape[1] == p
    assert fit.X.shape[0] == fit.k


def test_scalar_fields_are_python_numbers(fit):
    for name in ("tau2", "s2w", "I2", "H2", "QE", "QEp", "QM", "QMp", "vt"):
        value = getattr(fit, name)
        assert isinstance(value, float), f"{name} is {type(value).__name__}"
        assert math.isfinite(value), name
    for name in ("k", "k_f", "k_eff", "k_all", "p", "p_eff"):
        assert isinstance(getattr(fit, name), int), name


def test_the_lengths_agree(fit):
    n = len(fit.b)
    assert len(fit.se) == n
    for name in ("zval", "pval", "ci_lb", "ci_ub"):
        assert len(getattr(fit, name)) == n, name
    k = fit.k
    for name in ("yi", "vi", "weights", "fitted", "residuals"):
        assert len(getattr(fit, name)) == k, name


def test_the_record_serialises(fit):
    payload = fit.to_dict()
    text = json.dumps(payload, allow_nan=False)
    assert "tau2" in text


def test_a_moderated_fit_reports_mods_as_an_array():
    mods = [0.10, -0.20, 0.05, 0.30]
    fit = ma.rma(YI, vi=VI, mods=mods, method="REML")
    assert isinstance(fit.mods, np.ndarray)
    # (k, 1), not (k,): a moderator is a column of the model matrix, and a fit
    # with three moderators gives (k, 3).  The contract is "2-D, one column per
    # moderator", which is what the API reference says.
    assert fit.mods.shape == (len(YI), 1)
    # btt is 1-based over the model matrix including the intercept, as in R:
    # one moderator is column 2.  A list of ints, not an array.
    assert isinstance(fit.btt, list) and fit.btt == [2]
