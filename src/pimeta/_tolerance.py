"""The numerical resolutions the diagnostics are allowed to assume.

Separate from any estimator on purpose: a threshold that is quoted in three
modules drifts, and a threshold that is chosen without a reason becomes a
source of disagreement between two correct implementations.

## Why a variance "at zero" means ``<= 1e-8``

A variance component lives on the scale of the data's sampling variances --
typically ``1e-2`` to ``1e-1`` here -- and its standard error is of the same
order.  ``1e-8`` is therefore six orders of magnitude below anything the
evidence could distinguish from zero.

The number used to be ``1e-10``, which was not a decision but an accident, and
it sat *below* the optimiser's own resolution: L-BFGS-B configured with
``ftol=1e-12`` stops within about ``1e-9`` of the boundary in the parameter
itself, so whether a component "was at zero" depended on which of two
equivalent boundary points the search happened to reach.  Measured on an IPD
cohort whose likelihood puts every variance at zero: 1.1.5 stopped at exactly
``0.0`` and reported all four components at the boundary, while 1.1.6 -- same
model, same data, one more starting point -- stopped at ``9e-10`` and reported
three.  Nothing about the data had changed; only the last digits had.

A diagnostic that flips on the last digits is worse than no diagnostic, so the
tolerance is stated once, here, above the noise floor of the search.
"""

from __future__ import annotations

#: A variance component at or below this is zero for every purpose here.
BOUNDARY_TOL = 1e-8

#: A correlation within this of +-1 is on the edge of the cone.
RHO_BOUNDARY_TOL = 1e-6

__all__ = ["BOUNDARY_TOL", "RHO_BOUNDARY_TOL"]
