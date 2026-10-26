# Field map: pimeta against metafor and netmeta

Porting code between R and Python fails in three different ways, and only two of
them are visible from the documentation of either package:

* **the same statistic under a different name** — `x$ci.lb` is `x.ci_lb`, because
  `.` is attribute access in Python. Visible, mechanical, harmless once known;
* **the same name for a different statistic** — a field spelled identically in
  both packages that is not the same quantity. This is the dangerous one: a
  ported line keeps working and quietly changes meaning;
* **no counterpart** — a field that exists on one side only.

This page lists all three for `rma`, which is the model everything else is built
on, and names the differences for the network and multilevel families. It is
checked by `tests/test_field_map.py`: every pimeta field named here is read off a
real fitted object, every "same name" claim is verified to be the same name, and
every "no counterpart" claim is verified to be genuinely absent.

## 1. Same name, same statistic

These carry the same quantity in both packages, computed the same way. They are
verified field by field against R in the parity tests, so a difference here is a
porting bug, not a naming decision.

| Field | Meaning |
|---|---|
| `b`, `se` | coefficients and their standard errors |
| `zval`, `pval` | test statistic and p-value per coefficient |
| `vb` | coefficient covariance matrix |
| `tau2`, `se_tau2` | between-study variance and its standard error |
| `I2`, `H2`, `R2` | heterogeneity and variance-explained proportions |
| `QE`, `QEp`, `QM`, `QMdf`, `QMp` | the homogeneity and omnibus tests |
| `k`, `p` | studies and coefficients |
| `weights`, `fitted`, `residuals` | per-study weights, fits and residuals |
| `method`, `test`, `level` | estimator, reference distribution, confidence level |
| `yi`, `vi`, `X`, `mods` | the data the fit used |
| `coefnames`, `slab`, `measure` | names and labels |

## 2. Same statistic, different name

R's dotted names are not valid Python attributes, so they become underscores. The
mapping is mechanical: replace `.` with `_`. Two rows are *not* mechanical and are
the ones to check when porting:

| metafor | pimeta | Note |
|---|---|---|
| `ci.lb`, `ci.ub` | `ci_lb`, `ci_ub` | mechanical |
| `se.tau2` | `se_tau2` | mechanical |
| `tau2.fix` | `tau2_fix` | mechanical |
| `k.all` | `k_all` | mechanical |
| `k.eff`, `p.eff` | `k_eff`, `p_eff` | mechanical |
| `not.na` | `not_na` | mechanical |
| `int.only`, `int.incl` | `int_only`, `int_incl` | mechanical |
| **`weight`** | **`weights_pct`** | **not mechanical**: R's singular `weight` is the per-study *percentage*, while R's plural `weights` is the raw inverse-variance weight. pimeta keeps the plural for the raw weight and calls the percentage `weights_pct`, so a ported `x$weight` that becomes `x.weight` would fail — which is better than succeeding with the wrong quantity |
| **`R2`** | **`R2`** | same name, but only defined for a model with moderators: it is `1 - tau2(model) / tau2(intercept-only)`, and pimeta returns `None` rather than a number when there is no null model to compare against |

## 3. No counterpart in metafor

| pimeta | What it is |
|---|---|
| `diagnostics` | the structured `Diagnosis` for this fit — codes, severity, message, action. R prints a warning and forgets it; this is the same finding as data |
| `missing` | the missing-data policy the fit actually applied, with counts. R warns and drops |
| `to_dict()` | the versioned record (`schema_version`, `software`, `diagnostics`, every field) |
| `analysis` / `settings` | on a saved record: the digest that identifies the analysis, and the settings that produced it |

## 4. No counterpart in pimeta

| metafor | Why it is absent |
|---|---|
| `W`, `M`, `H` (`rma.uni`), `G` (`rma.mv`) | the k × k working matrices. They are dense `k × k` arrays — 32 MB at `k = 2000`, 3 GB at `k = 20000` — and every statistic computed from them has a closed form in `p × p` matrices. `pimeta._projection` computes `y'Py`, `tr(P)` and `tr(P²)` in `O(kp)` instead. Nothing is lost: `tests/test_projection_helpers.py` compares each `O(kp)` expression against the explicit matrix |
| `call`, `digits`, `verbose`, `...` | R's call record and printing options. The equivalent is `build_record(...)`, which stores the call and the settings in a form that replays |
| `V` (the full study-level covariance) | available as an input and used internally; `rma_mv` exposes the fitted `sigma2`/`rho` instead of the assembled matrix |

## 5. Network and multilevel families

The three categories apply unchanged. The differences worth knowing:

* `netmeta` returns `TE`, `seTE`, `treat1`, `treat2`, `k`, `tau2` under the same
  names; the league table is `netleague()` and the ranking is `netrank()`, both
  present under those names.
* `rma_mv` stores `tau2` and `sigma2` as **lists**, not the scalars `rma` uses,
  because a multilevel model has several variance components. Code that ports
  `fit.tau2` from `rma` and calls `float()` on it will fail on `rma_mv` — by
  design, since `float()` of a list is meaningless.
* `rho` appears only where a correlation is estimated (`rma_mv` with `struct=`,
  `selmodel`, `dta`, the time-course families); its absence means the model has
  none, not that it was not computed.

## See also

- [Differences from R](differences.md) — names that cannot exist in Python, and
  where numbers legitimately differ
- [Reproducible analyses](../guide/reproducibility.md) — the record format
