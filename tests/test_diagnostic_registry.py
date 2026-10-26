"""The diagnostic registry: every code in the source, in one auditable list.

Codes are the contract of `pimeta.checks` — messages may be reworded, codes may
not. That only works if the set of codes is knowable without grepping, so this
module pins it: the scan below finds every ``code="..."`` literal in the package
and compares it with :data:`REGISTRY`. A new code therefore has to be added here
deliberately, which is also where its one-line meaning lives.

The reason for a static scan rather than a behavioural one is the lesson from
`studies_dropped`: a check that cannot fire is worse than no check, and the way
to notice one is to look at what can be *constructed*, not at what a few
hand-built fits happen to produce.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "pimeta"

_CODE = re.compile(r'code\s*=\s*"([a-z][a-z0-9_]*)"')

#: Every diagnostic code the package can emit, with its meaning. Grouped by the
#: module that owns it; the grouping is documentation, the set is the contract.
REGISTRY: dict[str, str] = {
    # checks.py — the per-fit checks run by `diagnose()`
    "tau2_zero": "the estimated heterogeneity is zero",
    "variance_component_zero": "a variance component was estimated at its boundary",
    "ill_conditioned": "the design or the covariance is numerically ill-conditioned",
    "boundary_correlation": "a correlation was estimated on its +-1 boundary",
    "non_finite_estimate": "a reported number is not finite",
    "zero_weight_study": "a study contributed no information",
    "negative_variance": "a sampling variance is negative, so the weights are not weights",
    "zero_variance": "a sampling variance is zero: that study is known exactly",
    "not_converged": "the optimiser stopped without converging",
    "studies_dropped": "rows were removed before fitting",
    "few_studies": "too few studies for the reference to be nominal",
    "posterior_equals_prior": "the data did not move the prior",
    # multiverse.py — the grid-level findings on `MultiverseResult`
    "no_usable_specification": "no specification produced a usable estimate",
    "specification_failed": "some specifications produced no estimate at all",
    "specification_excluded": "some were dropped for an error-severity diagnostic",
}


def found_codes() -> set[str]:
    codes: set[str] = set()
    for path in SRC.rglob("*.py"):
        codes.update(_CODE.findall(path.read_text(encoding="utf-8")))
    return codes


def test_no_diagnostic_code_is_missing_from_the_registry():
    missing = sorted(found_codes() - set(REGISTRY))
    assert not missing, (
        "these codes are emitted but not registered: "
        f"{missing}. Add them to REGISTRY with their meaning.")


def test_the_registry_has_no_code_the_package_cannot_emit():
    # The other direction: a registered code that nothing constructs is a
    # documented behaviour that does not exist.
    unused = sorted(set(REGISTRY) - found_codes())
    assert not unused, f"registered but never emitted: {unused}"


def test_every_registered_code_reads_as_a_finding_not_a_symptom():
    for code in REGISTRY:
        assert code == code.lower()
        assert " " not in code and "-" not in code
        assert not code.endswith("_"), code


def test_every_registered_code_has_a_meaning():
    for code, meaning in REGISTRY.items():
        assert meaning and meaning[0].islower(), code
        assert len(meaning) > 20, f"{code}: a meaning shorter than this says nothing"
