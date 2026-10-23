"""Multivariate and dependent-effect-size tooling.

The estimator (``rma_mv``) lives in :mod:`pimeta.rma_mv`; this subpackage is
the layer around it — building the sampling covariance, reading the
between-outcome structure back out, and showing what an assumption is worth.

===========================  ==================================================
``mv.vcov``                  ``vcalc``: the sampling covariance of dependent
                             effect sizes, and the shared-condition
                             correlation it derives
``mv.corr``                  ``tau_cor``: between-outcome correlations of a
                             fitted model
``mv.sens``                  ``rho_sensitivity``: the same model along a grid
                             of assumed within-study correlations
===========================  ==================================================
"""

from __future__ import annotations

from .corr import TauCorResult, tau_cor
from .sens import RhoSensitivityResult, rho_sensitivity
from .vcov import shared_control_corr, vcalc

__all__ = [
    "TauCorResult",
    "RhoSensitivityResult",
    "rho_sensitivity",
    "shared_control_corr",
    "tau_cor",
    "vcalc",
]
