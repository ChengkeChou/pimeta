# Network meta-analysis

*53 public names, generated from `pimeta.__all__` and the docstrings in `src/pimeta/`. Do not edit this page by hand — run `python tools/gen_api_docs.py`.*

| Name | Mirrors in R |
|---|---|
| [`netmeta`](#netmeta) | — |
| [`NetmetaResult`](#netmetaresult) | — |
| [`netcomb`](#netcomb) | — |
| [`NetcombResult`](#netcombresult) | — |
| [`NetcomplexResult`](#netcomplexresult) | netcomplex |
| [`discomb`](#discomb) | — |
| [`netmetabin`](#netmetabin) | — |
| [`NetmetabinResult`](#netmetabinresult) | — |
| [`netpairwise`](#netpairwise) | — |
| [`NetpairwiseResult`](#netpairwiseresult) | — |
| [`netrank`](#netrank) | — |
| [`NetrankResult`](#netrankresult) | netrank |
| [`netsplit`](#netsplit) | — |
| [`NetsplitResult`](#netsplitresult) | — |
| [`netcontrib`](#netcontrib) | — |
| [`NetcontribResult`](#netcontribresult) | — |
| [`netleague`](#netleague) | — |
| [`NetleagueResult`](#netleagueresult) | — |
| [`decomp_design`](#decomp_design) | — |
| [`DecompDesignResult`](#decompdesignresult) | — |
| [`netconnection`](#netconnection) | — |
| [`NetconnectionResult`](#netconnectionresult) | — |
| [`netdistance`](#netdistance) | — |
| [`NetdistanceResult`](#netdistanceresult) | — |
| [`netcomplex`](#netcomplex) | netcomplex |
| [`netcomplexity`](#netcomplexity) | — |
| [`netmetareg`](#netmetareg) | netmetareg |
| [`NetmetaregResult`](#netmetaregresult) | netmetareg |
| [`netmeasures`](#netmeasures) | netmeasures |
| [`NetmeasuresResult`](#netmeasuresresult) | netmeasures |
| [`netmatrix`](#netmatrix) | netmatrix |
| [`NetmatrixResult`](#netmatrixresult) | — |
| [`netcomparison`](#netcomparison) | netcomparison |
| [`NetcomparisonResult`](#netcomparisonresult) | netcomparison |
| [`netbind`](#netbind) | netbind |
| [`NetbindResult`](#netbindresult) | netbind |
| [`crossnma2netmeta`](#crossnma2netmeta) | crossnma2netmeta |
| [`gemtc2netmeta`](#gemtc2netmeta) | gemtc2netmeta |
| [`treats`](#treats) | treats |
| [`vikor`](#vikor) | vikor |
| [`VikorResult`](#vikorresult) | — |
| [`subgroup`](#subgroup) | subgroup |
| [`NetsubgroupResult`](#netsubgroupresult) | — |
| [`nettable`](#nettable) | nettable |
| [`NettableResult`](#nettableresult) | — |
| [`createC`](#createc) | createC |
| [`invmat`](#invmat) | invmat |
| [`hatmatrix`](#hatmatrix) | hatmatrix |
| [`comps`](#comps) | comps |
| [`combinations`](#combinations) | combinations |
| [`netimpact`](#netimpact) | netimpact |
| [`NetimpactResult`](#netimpactresult) | netimpact |
| [`netpath`](#netpath) | netpath |

## `netmeta`

```python
netmeta(TE, seTE, treat1, treat2, studlab=None, sm: 'str' = '', level: 'float' = 0.95, common: 'bool' = True, random: 'bool' = True, method_tau: 'str' = 'DL', tau_preset: 'float | None' = None, reference_group: 'str' = '', baseline_reference: 'bool' = True, small_values: 'str' = 'desirable', n1=None, n2=None, event1=None, event2=None, incr2=None, mean1=None, mean2=None, sd1=None, sd2=None, time1=None, time2=None, data=None) -> 'NetmetaResult'
```

Frequentist graph-theoretical network meta-analysis (Rücker 2012).

**Args**

- `TE` — per-comparison treatment effects (m).
- `seTE` — per-comparison standard errors (m).
- `treat1, treat2` — treatment labels for each comparison (m).
- `studlab` — study labels (m). If None, each comparison is its own (two-arm) study.
- `sm` — summary measure (informational, e.g. "MD", "RR").
- `level` — confidence level (0.95 -> 95%).
- `common` — fit the common-effects model.
- `random` — fit the random-effects model.
- `method_tau` — tau^2 estimator.  ``"DL"`` (default) is Jackson et al.'s network method of moments; ``"ML"`` and ``"REML"`` reproduce netmeta's ``metafor::rma.mv`` fit on the star-shaped per-study design (needs ``reference_group`` semantics identical to R, and optionally the arm-level data below for the blocked ``V``).
- `tau_preset` — optional preset tau (square-root of tau^2).
- `reference_group` — reference treatment label (informational).
- `baseline_reference` — whether the reference group is the baseline (comparisons are expressed as "other treatment vs reference").
- `small_values` — "desirable" if smaller effects are beneficial (P-scores).
- `n1, n2` — arm sample sizes (only used by ``method_tau="ML", "REML"``).
- `event1, event2` — arm event counts.
- `incr2` — continuity increment applied to the reference arm.
- `mean1, mean2, sd1, sd2` — arm means / SDs.
- `time1, time2` — arm person-times.

## `NetmetaResult`

```python
NetmetaResult(trts: 'list[str]', k: 'int', m: 'int', n: 'int', sm: 'str', level: 'float', method_tau: 'str', TE_common: 'np.ndarray' = <factory>, seTE_common: 'np.ndarray' = <factory>, lower_common: 'np.ndarray' = <factory>, upper_common: 'np.ndarray' = <factory>, statistic_common: 'np.ndarray' = <factory>, pval_common: 'np.ndarray' = <factory>, TE_random: 'np.ndarray' = <factory>, seTE_random: 'np.ndarray' = <factory>, lower_random: 'np.ndarray' = <factory>, upper_random: 'np.ndarray' = <factory>, statistic_random: 'np.ndarray' = <factory>, pval_random: 'np.ndarray' = <factory>, TE_direct_common: 'np.ndarray' = <factory>, seTE_direct_common: 'np.ndarray' = <factory>, TE_direct_random: 'np.ndarray' = <factory>, seTE_direct_random: 'np.ndarray' = <factory>, lower_direct_common: 'np.ndarray' = <factory>, upper_direct_common: 'np.ndarray' = <factory>, lower_direct_random: 'np.ndarray' = <factory>, upper_direct_random: 'np.ndarray' = <factory>, Q_direct: 'np.ndarray' = <factory>, tau2_direct: 'np.ndarray' = <factory>, I2_direct: 'np.ndarray' = <factory>, prop_direct_common: 'np.ndarray' = <factory>, prop_direct_random: 'np.ndarray' = <factory>, TE_indirect_common: 'np.ndarray' = <factory>, seTE_indirect_common: 'np.ndarray' = <factory>, lower_indirect_common: 'np.ndarray' = <factory>, upper_indirect_common: 'np.ndarray' = <factory>, statistic_indirect_common: 'np.ndarray' = <factory>, pval_indirect_common: 'np.ndarray' = <factory>, TE_indirect_random: 'np.ndarray' = <factory>, seTE_indirect_random: 'np.ndarray' = <factory>, lower_indirect_random: 'np.ndarray' = <factory>, upper_indirect_random: 'np.ndarray' = <factory>, statistic_indirect_random: 'np.ndarray' = <factory>, pval_indirect_random: 'np.ndarray' = <factory>, A_matrix: 'np.ndarray' = <factory>, B_matrix: 'np.ndarray' = <factory>, L_matrix_common: 'np.ndarray' = <factory>, Lplus_matrix_common: 'np.ndarray' = <factory>, L_matrix_random: 'np.ndarray' = <factory>, Lplus_matrix_random: 'np.ndarray' = <factory>, H_matrix_common: 'np.ndarray' = <factory>, H_matrix_random: 'np.ndarray' = <factory>, Cov_common: 'np.ndarray' = <factory>, Cov_random: 'np.ndarray' = <factory>, Q: 'float' = nan, df_Q: 'float' = nan, pval_Q: 'float' = nan, I2: 'float' = nan, lower_I2: 'float' = nan, upper_I2: 'float' = nan, tau: 'float' = nan, tau2: 'float' = nan, Q_heterogeneity: 'float' = nan, df_Q_heterogeneity: 'float' = nan, pval_Q_heterogeneity: 'float' = nan, Q_inconsistency: 'float' = nan, df_Q_inconsistency: 'float' = nan, pval_Q_inconsistency: 'float' = nan, Q_decomp: 'list[dict]' = <factory>, P_common: 'np.ndarray' = <factory>, P_random: 'np.ndarray' = <factory>, pscore_common: 'np.ndarray' = <factory>, pscore_random: 'np.ndarray' = <factory>, reference_group: 'str' = '', baseline_reference: 'bool' = True, TE_raw: 'list' = <factory>, seTE_raw: 'list' = <factory>, treat1_raw: 'list' = <factory>, treat2_raw: 'list' = <factory>, studlab_raw: 'list' = <factory>, narms_raw: 'list' = <factory>, treat1_pos: 'list' = <factory>, treat2_pos: 'list' = <factory>, seTE_adj_common: 'list' = <factory>, seTE_adj_random: 'list' = <factory>) -> None
```

Results of a (common + random effects) network meta-analysis.

Treatment-effect quantities are n x n matrices indexed by treatment name
(rows/columns in the sorted treatment order). ``TE[i, j]`` is the network
estimate of the effect of treatment *i* relative to treatment *j*
(i.e. ``theta[i] - theta[j]``).

## `netcomb`

```python
netcomb(x, inactive: 'str | None' = None, sep_comps: 'str' = '+', common: 'bool | None' = None, random: 'bool | None' = None, tau_preset: 'float | None' = None) -> 'NetcombResult'
```

Additive component network meta-analysis (netmeta::netcomb).

``x`` is a :class:`NetmetaResult` from :func:`netmeta` whose treatment
labels encode combination treatments (e.g. ``"A+B"``). The additive CNMA
model assumes a combination's effect is the sum of its components'
effects: ``delta = B theta`` with ``theta = C beta`` (Rücker et al. 2020),
estimated by weighted least squares.

## `NetcombResult`

```python
NetcombResult(trts: 'list[str]', comps: 'list[str]', k: 'int', m: 'int', n: 'int', d: 'int', c: 'int', sm: 'str', level: 'float', method_tau: 'str', reference_group: 'str', inactive: 'str | None', sep_comps: 'str', TE_common: 'np.ndarray' = <factory>, seTE_common: 'np.ndarray' = <factory>, lower_common: 'np.ndarray' = <factory>, upper_common: 'np.ndarray' = <factory>, statistic_common: 'np.ndarray' = <factory>, pval_common: 'np.ndarray' = <factory>, TE_random: 'np.ndarray' = <factory>, seTE_random: 'np.ndarray' = <factory>, lower_random: 'np.ndarray' = <factory>, upper_random: 'np.ndarray' = <factory>, statistic_random: 'np.ndarray' = <factory>, pval_random: 'np.ndarray' = <factory>, Comp_common: 'np.ndarray' = <factory>, seComp_common: 'np.ndarray' = <factory>, lower_Comp_common: 'np.ndarray' = <factory>, upper_Comp_common: 'np.ndarray' = <factory>, statistic_Comp_common: 'np.ndarray' = <factory>, pval_Comp_common: 'np.ndarray' = <factory>, Comp_random: 'np.ndarray' = <factory>, seComp_random: 'np.ndarray' = <factory>, lower_Comp_random: 'np.ndarray' = <factory>, upper_Comp_random: 'np.ndarray' = <factory>, statistic_Comp_random: 'np.ndarray' = <factory>, pval_Comp_random: 'np.ndarray' = <factory>, Comb_common: 'np.ndarray' = <factory>, seComb_common: 'np.ndarray' = <factory>, lower_Comb_common: 'np.ndarray' = <factory>, upper_Comb_common: 'np.ndarray' = <factory>, statistic_Comb_common: 'np.ndarray' = <factory>, pval_Comb_common: 'np.ndarray' = <factory>, Comb_random: 'np.ndarray' = <factory>, seComb_random: 'np.ndarray' = <factory>, lower_Comb_random: 'np.ndarray' = <factory>, upper_Comb_random: 'np.ndarray' = <factory>, statistic_Comb_random: 'np.ndarray' = <factory>, pval_Comb_random: 'np.ndarray' = <factory>, TE_nma_common: 'np.ndarray' = <factory>, seTE_nma_common: 'np.ndarray' = <factory>, lower_nma_common: 'np.ndarray' = <factory>, upper_nma_common: 'np.ndarray' = <factory>, TE_nma_random: 'np.ndarray' = <factory>, seTE_nma_random: 'np.ndarray' = <factory>, lower_nma_random: 'np.ndarray' = <factory>, upper_nma_random: 'np.ndarray' = <factory>, TE_cnma_common: 'np.ndarray' = <factory>, seTE_cnma_common: 'np.ndarray' = <factory>, lower_cnma_common: 'np.ndarray' = <factory>, upper_cnma_common: 'np.ndarray' = <factory>, TE_cnma_random: 'np.ndarray' = <factory>, seTE_cnma_random: 'np.ndarray' = <factory>, lower_cnma_random: 'np.ndarray' = <factory>, upper_cnma_random: 'np.ndarray' = <factory>, Q_additive: 'float' = nan, df_Q_additive: 'float' = nan, pval_Q_additive: 'float' = nan, Q_standard: 'float' = nan, df_Q_standard: 'float' = nan, pval_Q_standard: 'float' = nan, Q_diff: 'float' = nan, df_Q_diff: 'float' = nan, pval_Q_diff: 'float' = nan, tau: 'float' = nan, tau2: 'float' = nan, I2: 'float' = nan, lower_I2: 'float' = nan, upper_I2: 'float' = nan, A_matrix: 'np.ndarray' = <factory>, B_matrix: 'np.ndarray' = <factory>, X_matrix: 'np.ndarray' = <factory>, C_matrix: 'np.ndarray' = <factory>, L_matrix_common: 'np.ndarray' = <factory>, Lplus_matrix_common: 'np.ndarray' = <factory>, H_matrix_common: 'np.ndarray' = <factory>, Cov_common: 'np.ndarray' = <factory>, L_matrix_random: 'np.ndarray' = <factory>, Lplus_matrix_random: 'np.ndarray' = <factory>, H_matrix_random: 'np.ndarray' = <factory>, Cov_random: 'np.ndarray' = <factory>) -> None
```

Additive component network meta-analysis (netmeta::netcomb).

## `NetcomplexResult`

```python
NetcomplexResult(complex: 'list[str]', comps: 'list[str]', C_matrix: 'np.ndarray', Comb_common: 'np.ndarray' = <factory>, seComb_common: 'np.ndarray' = <factory>, lower_Comb_common: 'np.ndarray' = <factory>, upper_Comb_common: 'np.ndarray' = <factory>, statistic_Comb_common: 'np.ndarray' = <factory>, pval_Comb_common: 'np.ndarray' = <factory>, Comb_random: 'np.ndarray' = <factory>, seComb_random: 'np.ndarray' = <factory>, lower_Comb_random: 'np.ndarray' = <factory>, upper_Comb_random: 'np.ndarray' = <factory>, statistic_Comb_random: 'np.ndarray' = <factory>, pval_Comb_random: 'np.ndarray' = <factory>, inactive: 'str | None' = None, level: 'float' = 0.95, common: 'bool' = True, random: 'bool' = True) -> None
```

*Mirrors R's `netcomplex`.*

Result of :func:`netcomplex` (``netmeta::netcomplex``).

## `discomb`

```python
discomb(TE, seTE, treat1, treat2, studlab=None, inactive: 'str | None' = None, sep_comps: 'str' = '+', sm: 'str' = '', level: 'float' = 0.95, common: 'bool' = True, random: 'bool' = True, reference_group: 'str' = '', tau_preset: 'float | None' = None) -> 'NetcombResult'
```

Additive CNMA for possibly disconnected networks (netmeta::discomb).

Unlike :func:`netcomb` (which starts from a :class:`NetmetaResult` and
therefore requires a connected network), ``discomb`` takes raw
per-comparison data and works even when the network is disconnected.  The
additive degrees of freedom are ``df.Q.additive = n.a - k - rank(X)`` with
``n.a`` the total number of study arms and ``k`` the number of studies.
When the network is connected the standard heterogeneity Q is taken from an
internal :func:`netmeta` fit; otherwise it is undefined (NaN).

## `netmetabin`

```python
netmetabin(event1, n1, event2, n2, treat1, treat2, studlab=None, sm: 'str' = 'OR', method: 'str' = 'MH', level: 'float' = 0.95, level_ma: 'float | None' = None, common: 'bool' = True, random: 'bool' = True, reference_group: 'str' = '', baseline_reference: 'bool' = True, cc_pooled: 'bool' = False, incr: 'float' = 0.5, tau_preset: 'float | None' = None)
```

Binary network meta-analysis (netmeta::netmetabin).

**Args**

- `event1, n1` — events and group size of the ``treat1`` arm (per study).
- `event2, n2` — events and group size of the ``treat2`` arm (per study).
- `treat1, treat2` — treatment labels for each comparison.
- `studlab` — study labels. If None, each comparison is its own study.
- `sm` — summary measure ("OR" for MH; "OR"/"RR" for Inverse).
- `method` — "MH" (Mantel-Haenszel, default), "NCH" (conditional non-central hypergeometric likelihood), "LRP" (Jeffreys-penalised logistic regression, Firth -- the only method with a random-effects counterpart, obtained by inflating the SEs by ``phi``), or "Inverse" (classic NMA via :func:`netmeta`).
- `level, level_ma` — confidence levels.
- `common, random` — fit common / random effects (Inverse method only).
- `cc_pooled` — pool zero cells across studies (MH method).
"MH" method reproduces Stijnen et al. (2010); "Inverse" delegates to
- `c` — `netmeta` after computing per-comparison log-OR estimates.

## `NetmetabinResult`

```python
NetmetabinResult(trts: 'list', k: 'int', m: 'int', n: 'int', d: 'int', designs: 'list', sm: 'str', method: 'str', level: 'float', level_ma: 'float', reference_group: 'str', cc_pooled: 'bool', TE_common: 'np.ndarray' = <factory>, seTE_common: 'np.ndarray' = <factory>, lower_common: 'np.ndarray' = <factory>, upper_common: 'np.ndarray' = <factory>, statistic_common: 'np.ndarray' = <factory>, pval_common: 'np.ndarray' = <factory>, TE_direct_common: 'np.ndarray' = <factory>, seTE_direct_common: 'np.ndarray' = <factory>, lower_direct_common: 'np.ndarray' = <factory>, upper_direct_common: 'np.ndarray' = <factory>, statistic_direct_common: 'np.ndarray' = <factory>, pval_direct_common: 'np.ndarray' = <factory>, Q_direct: 'np.ndarray' = <factory>, tau2_direct: 'np.ndarray' = <factory>, I2_direct: 'np.ndarray' = <factory>, Q: 'float' = nan, df_Q: 'float' = nan, pval_Q: 'float' = nan, Q_inconsistency: 'float' = nan, df_Q_inconsistency: 'float' = nan, pval_Q_inconsistency: 'float' = nan, I2: 'float' = nan, tau: 'float' = nan, tau2: 'float' = nan, TE_random: 'np.ndarray' = <factory>, seTE_random: 'np.ndarray' = <factory>, lower_random: 'np.ndarray' = <factory>, upper_random: 'np.ndarray' = <factory>, phi: 'float' = nan, A_matrix: 'np.ndarray' = <factory>, H_matrix_common: 'np.ndarray' = <factory>, TE: 'list' = <factory>, seTE: 'list' = <factory>, treat1: 'list' = <factory>, treat2: 'list' = <factory>, studlab: 'list' = <factory>, event1: 'list' = <factory>, event2: 'list' = <factory>, n1: 'list' = <factory>, n2: 'list' = <factory>) -> None
```

Binary network meta-analysis (netmeta::netmetabin), common-effects.

Implements the Mantel-Haenszel NMA of Stijnen et al. (2010): per-design
log-OR contrasts ``L.xy`` with variances/covariances ``V`` are combined by
weighted least squares into basic contrasts ``TE.basic``, then expanded to
all treatment pairs via the hat-like matrix ``H``.  All effect quantities
are reported on the **log-OR** scale (matching ``netmetabin(...)$TE.common``
etc. before back-transformation).

## `netpairwise`

```python
netpairwise(x, separate_trts: 'bool' = True, reference_group: 'str | None' = None, sm: 'str' = '', level: 'float' = 0.95, method_tau: 'str' = 'DL', common: 'bool' = True, random: 'bool' = True) -> 'NetpairwiseResult'
```

Pairwise (meta-analysis) view of a network meta-analysis.

Transcribes ``netmeta::netpairwise``. ``x`` is a :class:`NetmetaResult`
(from :func:`netmeta`). The result is a metagen-style object whose overall
pooled estimates come from a DerSimonian-Laird random-effects pairwise
meta-analysis of all direct comparisons, plus a ``pairs`` table giving the
per-comparison direct evidence (common/random effect, Q, I^2, tau^2).

## `NetpairwiseResult`

```python
NetpairwiseResult(trts: 'list[str]', k: 'int', reference_group: 'str', sm: 'str', level: 'float', method_tau: 'str', TE_common: 'float' = nan, seTE_common: 'float' = nan, lower_common: 'float' = nan, upper_common: 'float' = nan, statistic_common: 'float' = nan, pval_common: 'float' = nan, TE_random: 'float' = nan, seTE_random: 'float' = nan, lower_random: 'float' = nan, upper_random: 'float' = nan, statistic_random: 'float' = nan, pval_random: 'float' = nan, tau2: 'float' = nan, tau: 'float' = nan, I2: 'float' = nan, lower_I2: 'float' = nan, upper_I2: 'float' = nan, Q: 'float' = nan, df_Q: 'float' = nan, pval_Q: 'float' = nan, TE: 'list' = <factory>, seTE: 'list' = <factory>, treat1: 'list' = <factory>, treat2: 'list' = <factory>, studlab: 'list' = <factory>, pairs: 'list' = <factory>) -> None
```

Pairwise meta-analysis of the direct comparisons of a network.

``netmeta::netpairwise`` re-expresses a network meta-analysis as a plain
pairwise meta-analysis (a ``metagen``-style object) over the direct
comparisons, plus a per-pair direct-evidence table. Every comparison is
oriented as "non-reference treatment vs reference treatment" when a
reference group is set, and alphabetically (treat1 < treat2) otherwise.

## `netrank`

```python
netrank(result: 'NetmetaResult', small_values: 'str' = 'desirable') -> 'NetrankResult'
```

P-scores (Rücker & Schwarzer 2015), exact transcription of netrank.R.

``P[i, j]`` = probability treatment *i* is better than treatment *j*,
computed from the two-sided p-value matrix as a one-sided p-value and
averaged over rows. ``small_values`` = "desirable" means smaller effects
are beneficial.

## `NetrankResult`

```python
NetrankResult(trts: 'list[str]', pscore_common: 'np.ndarray', pscore_random: 'np.ndarray', P_common: 'np.ndarray', P_random: 'np.ndarray', small_values: 'str', method: 'str' = 'P-score', nsim: 'int | None' = None) -> None
```

*Mirrors R's `netrank`.*

P-scores from a network meta-analysis (``netmeta::netrank``).

Field naming follows ``netrank()`` in netmeta 3.7-0, where - despite the
name - ``ranking.common`` / ``ranking.random`` hold **P-scores**, and the
pairwise probability matrices live in ``Pmatrix.common`` /
``Pmatrix.random``. ``Pscore.common`` does not exist in R.

Both the R spelling (``ranking_common``, ``Pmatrix_common``, ``method``,
``nsim``, ...) and the earlier snake_case spelling (``pscore_common``,
``P_common``, ...) are available.

## `netsplit`

```python
netsplit(x: 'NetmetaResult', method: 'str' = 'Back-calculation', common: 'bool' = True, random: 'bool' = True, tol_direct: 'float' = 0.0005) -> 'NetsplitResult'
```

Split direct and indirect evidence in a network meta-analysis.

Replicates ``netmeta::netsplit`` with ``method = "Back-calculation"``
(the default for non-binary ``netmeta`` objects). The indirect estimate is
back-calculated from the network and direct estimates using the direct
evidence proportion, already stored in ``x`` by :func:`netmeta`.

**Args**

- `x` — a :class:`NetmetaResult` from :func:`netmeta`.
- `method` — splitting method (only ``"Back-calculation"``).
- `common` — include common-effects results.
- `random` — include random-effects results.
- `tol_direct` — maximum deviation of the direct evidence proportion from 0 or 1 to classify a comparison as indirect-only or direct-only.

## `NetsplitResult`

```python
NetsplitResult(comparison: 'list[str]', k: 'list[int]', prop_common: 'list[float]', prop_random: 'list[float]', method: 'str' = 'Back-calculation', common: 'dict' = <factory>, direct_common: 'dict' = <factory>, indirect_common: 'dict' = <factory>, compare_common: 'dict' = <factory>, random: 'dict' = <factory>, direct_random: 'dict' = <factory>, indirect_random: 'dict' = <factory>, compare_random: 'dict' = <factory>) -> None
```

Split of direct and indirect evidence per comparison.

``comparison`` lists every treatment pair in the upper-triangle order
(``trts[i] : trts[j]`` with ``i < j``). ``k`` is the number of studies
providing direct evidence (``A.matrix``); ``prop_common`` / ``prop_random``
are the direct-evidence proportions. The ``common`` / ``direct_common`` /
``indirect_common`` / ``compare_common`` dicts (and their ``*_random``
counterparts) hold aligned lists keyed by ``"TE"``, ``"seTE"``, ``"lower"``,
``"upper"``, ``"statistic"``, ``"p"`` (``compare_*`` also has ``"z"``).
Missing values (R ``NA``) are ``None``.

## `netcontrib`

```python
netcontrib(x: 'NetmetaResult', method: 'str' = 'shortestpath') -> 'NetcontribResult'
```

Contribution matrix of a network meta-analysis.

Transcribes ``netmeta::netcontrib`` (default "shortestpath" method). Each
row of the returned ``common`` / ``random`` matrices corresponds to a
treatment pair; each column to a direct comparison (edge). Rows sum to 1.

## `NetcontribResult`

```python
NetcontribResult(trts: 'list[str]', comparisons: 'list[str]', directs: 'list[str]', common: 'np.ndarray', random: 'np.ndarray', method: 'str' = 'shortestpath') -> None
```

Contribution matrix of a network meta-analysis (netmeta::netcontrib).

``common[r, c]`` (and ``random[r, c]``) is the proportion of the pooled
estimate for comparison ``comparisons[r]`` (a treatment pair, row) that is
contributed by the direct comparison ``directs[c]`` (column). Rows sum to
1 (within numerical precision). Uses the default "shortestpath"
(Papakonstantinou et al. 2018) flow-decomposition method.

## `netleague`

```python
netleague(x: 'NetmetaResult', common: 'bool' = True, random: 'bool' = True, seq: 'list[str] | None' = None, ci: 'bool' = True, backtransf: 'bool' = False, digits: 'int' = 4, text_na: 'str' = '.', details: 'bool' = True) -> 'NetleagueResult'
```

League table for a network meta-analysis.

Replicates ``netmeta::netleague`` for a single ``netmeta`` object: the
lower triangle gives the network estimates (column vs row) and the upper
triangle the direct estimates (row vs column).

**Args**

- `x` — a :class:`NetmetaResult` from :func:`netmeta`.
- `common` — include the common-effects table.
- `random` — include the random-effects table.
- `seq` — optional row/column treatment order (default: ``x.trts``).
- `ci` — include confidence intervals.
- `backtransf` — back-transform to the original scale (``exp`` for ratio measures; ``False`` for ``MD``).
- `digits` — number of decimals.
- `text_na` — string used for missing cells.
- `details` — include the explanatory caption.

## `NetleagueResult`

```python
NetleagueResult(trts: 'list[str]', common: 'list[list[str]]' = <factory>, random: 'list[list[str]]' = <factory>, details: 'str' = '') -> None
```

League table (network estimates in the lower triangle, direct estimates
in the upper triangle), replicating ``netmeta::netleague``.

``common`` and ``random`` are matrices (lists of string rows) of formatted
cells: ``"TE [lower; upper]"``, ``"."`` for missing, treatment names on the
diagonal. ``trts`` is the row/column order. ``details`` is the explanatory
caption text used by R's print method.

## `decomp_design`

```python
decomp_design(result: 'NetmetaResult', tau_preset=None) -> 'DecompDesignResult | None'
```

Decomposition of the Q statistic (netmeta::decomp.design, Krahn 2013).

``result`` must be a :class:`NetmetaResult` produced with the *raw*
(un-normalized) comparison directions retained.

## `DecompDesignResult`

```python
DecompDesignResult(Q_decomp: 'dict', Q_het_design: 'list[dict]', Q_inc_detach: 'list[dict]', Q_inc_design: 'np.ndarray', Q_inc_design_names: 'list[str]', Q_inc_random: 'dict', Q_inc_random_preset: 'dict | None', residuals_inc_detach: 'np.ndarray', tau_preset: 'float | None', tau_within: 'float', d: 'int') -> None
```

Decomposition of designs (netmeta::decomp.design).

## `netconnection`

```python
netconnection(treat1, treat2, studlab=None, sep_trts: 'str' = ':', title: 'str' = '', details_disconnected: 'bool' = False) -> 'NetconnectionResult'
```

Determine the (sub)network structure of a network meta-analysis.

Replicates ``netmeta::netconnection``: builds the weighted adjacency matrix
from ``treat1``/``treat2``, computes the graph-distance matrix, splits the
network into its connected subnetworks and reports the number of studies
(``k``), pairwise comparisons (``m``), treatments (``n``), subnetworks
(``n_subnets``) and designs (``d``), together with the ordered distance /
adjacency / Laplacian matrices.

**Args**

- `treat1, treat2` — the two treatments of each pairwise comparison (``m`` vectors, e.g. ``d["treat1"]`` / ``d["treat2"]``).
- `studlab` — study labels; ``None`` treats every comparison as its own study.
- `sep_trts` — separator used in the ``comparisons`` labels.

## `NetconnectionResult`

```python
NetconnectionResult(treat1: 'list' = <factory>, treat2: 'list' = <factory>, studlab: 'list' = <factory>, design: 'list' = <factory>, subnet: 'list' = <factory>, k: 'int' = 0, m: 'int' = 0, n: 'int' = 0, n_subnets: 'int' = 0, d: 'int' = 0, seq: 'list[str]' = <factory>, D_matrix: 'np.ndarray' = <factory>, A_matrix: 'np.ndarray' = <factory>, L_matrix: 'np.ndarray' = <factory>, designs: 'list[str]' = <factory>, comparisons: 'list[str]' = <factory>, subnet_comparisons: 'list[int]' = <factory>) -> None
```

Network connectivity structure (netmeta::netconnection).

Groups the treatment network into its connected subnetworks and reports
the graph distance matrix ``D_matrix``, the weighted adjacency matrix
``A_matrix`` (``A[i,j]`` = number of studies directly comparing ``i`` and
``j``) and the Laplacian ``L_matrix``, all indexed by ``seq``.

## `netdistance`

```python
netdistance(x, sort: 'bool' = True) -> 'NetdistanceResult'
```

Graph distance matrix (netmeta::netdistance).

Accepts either a weighted/0-1 adjacency ``ndarray``, a
:class:`NetmetaResult`, or a :class:`NetconnectionResult`. For a
``NetmetaResult`` the adjacency is sorted by the ``netconnection`` order
when ``sort`` is true, matching ``netmeta::netdistance.netmeta``.

## `NetdistanceResult`

```python
NetdistanceResult(D: 'np.ndarray', labels: 'list[str]' = <factory>, order: 'list[int]' = <factory>) -> None
```

Graph distance matrix (netmeta::netdistance).

``D[i, j]`` is the number of edges on the shortest path between nodes
``i`` and ``j`` (``inf`` if they lie in different subnetworks), diagonal
zero. ``order`` is the ``hclust`` leaf order R attaches for display
purposes; ``labels`` are the row/column names.

## `netcomplex`

```python
netcomplex(result, complex=None, common=True, random=True, level=None)
```

*Mirrors R's `netcomplex`.*

``netmeta::netcomplex`` — combination effects of a component NMA.

Estimates the effect of arbitrary treatment *complexes* (combinations of
components) from a fitted :func:`netcomb` model::

    Comb = C %*% Comp
    seComb = sqrt(diag(C %*% Lplus %*% t(C)))

**Args**

- `result` — a :class:`NetcombResult`.
- `complex` — which complexes to evaluate.  ``None`` (the default) uses every treatment of the network; otherwise a sequence of complex labels such as ``["A+B", "A+C+B"]`` (a 2-D array is read rowwise).
- `common, random` — which models to fill in.
- `level` — confidence level (defaults to the model's).

**Returns**

- `:class` — `NetcomplexResult`.

## `netcomplexity`

```python
netcomplexity(result) -> 'int'
```

Independent cycles of a comparison graph (a pimeta extra).

The number of independent cycles is ``m - n + n_subnets`` (with ``m``
comparisons, ``n`` treatments and ``n_subnets`` connected components).  A
network without cycles has complexity 0.

``netmeta`` itself does not export such a function; the equivalent
quantity in R is ``netconnection(...)$d``.  This helper is provided
because it was historically (and misleadingly) called ``netcomplex`` here —
that name now matches R's ``netmeta::netcomplex``, which computes
combination effects of a component network meta-analysis.

**Args**

- `result` — a :class:`NetmetaResult` (or any object exposing ``m``, ``n`` and ``n_subnets``).

**Returns**

The network complexity as an ``int``.

## `netmetareg`

```python
netmetareg(result, covar, method_tau='REML', reference_group=None, level=0.95, rho=0.5, assumption='independent', consistency=True, covar_name='covar', direction1=None, direction2=None, max_ia=False)
```

*Mirrors R's `netmetareg`.*

``netmeta::netmetareg`` — network meta-regression.

Fits the treatment-by-covariate interaction model that netmeta uses::

    theta_k = (mu_t2 - mu_t1) + (beta_t2 - beta_t1) * x_k

Comparisons are first re-oriented so that every study contributes contrasts
against its own reference treatment (the network reference when the study
contains it, otherwise the alphabetically last treatment of that study), the
design matrix is ``[I(treat) | I(treat) * x]`` over the non-reference
treatments, and the model is estimated by (restricted) maximum likelihood
with a block-equicorrelated within-study covariance
``V = diag(se^2) + tau2 * CS(rho)`` — the same structure as metafor's
``rma.mv(..., random = ~ comparison | study, rho = 0.5)``.

**Args**

- `result` — a :class:`NetmetaResult`.
- `covar` — study-level covariate (one value per raw comparison row, like ``netmetareg(net, covar = vector)`` in R).
- `method_tau` — ``"REML"`` (default), ``"ML"`` or ``"FE"``.
- `reference_group` — reference treatment; defaults to the network's reference (or the first treatment).
- `level` — confidence level.
- `rho` — within-study correlation of the random effects (metafor default 0.5).
- `assumption` — ``"independent"`` (treatment-specific interactions, the default) or ``"common"`` (a single common interaction).
- `consistency` — ``True`` (default) fits the consistency model; ``False`` fits the inconsistency model, which adds one covariate interaction per comparison (``assumption="independent"``) or a single common one (``assumption="common"``).
- `covar_name` — label used for the interaction terms (R derives it from the deparsed ``covar`` expression).
- `direction1, direction2` — per-comparison direction-of-effect values (-1/0/1) for the two arms, only used with ``consistency=False``. Without them netmeta falls back to treatment order (``mkIval``: the alphabetically first treatment of each study is the direction-0 arm).
- `max_ia` — keep every comparison-level interaction column, including those contributed by a single study (R's ``max.ia``).

**Returns**

- `:class` — `NetmetaregResult`.

**Note**

Exactly collinear design columns (which ``rma.mv`` drops) are removed before fitting; ``coef_na`` marks the dropped positions in the full column set.

## `NetmetaregResult`

```python
NetmetaregResult(beta: 'list[float]', se: 'list[float]', zval: 'list[float]', pval: 'list[float]', ci_lb: 'list[float]', ci_ub: 'list[float]', vb: 'np.ndarray', tau2: 'float', names: 'list[str]', coef_na: 'list[bool]', QE: 'float', QEdf: 'int', QM: 'float', QMdf: 'int', QMp: 'float', k: 'int', p: 'int', trts: 'list[str]', covar: 'np.ndarray', level: 'float') -> None
```

*Mirrors R's `netmetareg`.*

Result of :func:`netmetareg` (``netmeta::netmetareg``).

## `netmeasures`

```python
netmeasures(result, random=None, tau_preset=None)
```

*Mirrors R's `netmeasures`.*

``netmeta::netmeasures`` — network path measures (Koenig et al. 2013).

**Args**

- `result` — a :class:`NetmetaResult`.
- `random` — use the random-effects model.  Defaults to ``True``; forced to ``True`` when ``tau_preset`` is given.
- `tau_preset` — fixed heterogeneity variance.

**Returns**

- `:class` — `NetmeasuresResult` with ``proportion`` (share of the network
estimate's variance that is direct evidence), ``meanpath``, ``minpar``
and ``minpar_study``.

## `NetmeasuresResult`

```python
NetmeasuresResult(proportion, meanpath, minpar, minpar_study, H_tilde, random, tau_preset)
```

*Mirrors R's `netmeasures`.*

Result of :func:`netmeasures` (``netmeta::netmeasures``).

## `netmatrix`

```python
netmatrix(result, var, levels=None, labels=None, func='mode', ties_method='first', seed=None)
```

*Mirrors R's `netmatrix`.*

``netmeta::netmatrix`` — summarise a study-level variable per comparison.

Builds an ``n x n`` matrix holding, for every pair of treatments that is
compared directly, a summary (``"mode"``, ``"min"``, ``"max"``, ``"mean"``,
``"median"`` or ``"sum"``) of a study-level variable.  Unobserved pairs and
the diagonal are ``NaN``.

**Args**

- `result` — a :class:`NetmetaResult`.
- `var` — one value per study/comparison.
- `levels, labels` — optional recoding of the summary values.
- `func` — summary function name.
- `ties_method` — ``"first"``, ``"last"`` or ``"random"`` (for the mode).
- `seed` — RNG seed used when ``ties_method="random"``.

**Returns**

- `:class` — `NetmatrixResult`.

## `NetmatrixResult`

```python
NetmatrixResult(matrix, trts, func, levels, labels)
```

Result of :func:`netmatrix` — a treatment x treatment summary matrix.

## `netcomparison`

```python
netcomparison(result, treat1=None, treat2=None, common=True, random=True, level=0.95, nchar_comps=None, backtransf=False)
```

*Mirrors R's `netcomparison`.*

``netmeta::netcomparison`` — estimate comparisons between combinations.

Works on a :class:`NetcombResult`: the requested ``treat1`` vs ``treat2``
combinations (e.g. ``"A+B"`` vs ``"plac"``) are expressed as contrasts of
the estimated component effects.

**Args**

- `result` — a :class:`NetcombResult`.
- `treat1, treat2` — combinations to compare.  Defaults mirror R: all treatments vs the reference group.
- `common, random` — which models to report.
- `level` — confidence level.
- `nchar_comps` — component-name abbreviation length (unused, kept for API compatibility).
- `backtransf` — back-transform the estimates.

**Returns**

- `:class` — `NetcomparisonResult`.

## `NetcomparisonResult`

```python
NetcomparisonResult(**kw)
```

*Mirrors R's `netcomparison`.*

Result of :func:`netcomparison` (``netmeta::netcomparison``).

## `netbind`

```python
netbind(*results, name=None, common=None, random=None, reference_group=None, baseline_reference=True)
```

*Mirrors R's `netbind`.*

``netmeta::netbind`` — collect reference-comparison estimates.

For each network the estimates of every treatment against
``reference_group`` are extracted from the common-effect and random-effects
matrices and stacked into one long table (used to draw several networks in
a single forest plot).

**Args**

- `*results` — :class:`NetmetaResult` objects.
- `name` — labels for the networks (default ``"Network 1"``, ...).
- `common, random` — restrict the output to one model.
- `reference_group` — reference treatment (default: the first network's).
- `baseline_reference` — keep the reference as the *column* (``True``) or as the *row* (``False``).

**Returns**

- `:class` — `NetbindResult`.

## `NetbindResult`

```python
NetbindResult(common, random, sm, reference_group, baseline_reference, names, nets)
```

*Mirrors R's `netbind`.*

Result of :func:`netbind` (``netmeta::netbind``).

## `crossnma2netmeta`

```python
crossnma2netmeta(x, sm='', method_tau='DL', reference_group='', level=0.95)
```

*Mirrors R's `crossnma2netmeta`.*

``netmeta::crossnma2netmeta`` — convert a ``crossnma`` fit to netmeta.

``crossnma`` (Hamza et al.) fits a network meta-analysis that combines
randomised and non-randomised evidence.  Its fit object carries a study
level table with the columns ``studlab``, ``treat1``, ``treat2``, ``TE`` and
``seTE`` (accessed here either directly or through a ``data`` entry).

**Args**

- `x` — the ``crossnma`` object as a dict / mapping.
- `sm` — effect measure label.
- `method_tau` — heterogeneity estimator.
- `reference_group` — reference treatment.
- `level` — confidence level.

**Returns**

- `:class` — `NetmetaResult`.

## `gemtc2netmeta`

```python
gemtc2netmeta(x, sm='', method_tau='DL', reference_group='', level=0.95)
```

*Mirrors R's `gemtc2netmeta`.*

``netmeta::gemtc2netmeta`` — convert a ``gemtc`` network/fit to netmeta.

``gemtc`` (van Valkenhoef & Kuiper) stores the evidence either arm-based
(``data.ab``: one row per arm with ``study``, ``treatment`` and the outcome
columns) or contrast-based (``data.re``: ``study``, ``treatment``,
``diff``, ``std.err``).  Contrast data are used directly; arm-based data are
converted into consecutive within-study contrasts.

**Args**

- `x` — the ``gemtc`` object as a dict / mapping (a ``network`` or a fit).
- `sm, method_tau, reference_group, level` — forwarded to ``netmeta``.

**Returns**

- `:class` — `NetmetaResult`.

## `treats`

```python
treats(x, nchar_trts=8, row=True)
```

*Mirrors R's `treats`.*

``netmeta::treats`` — abbreviate treatment labels to ``nchar_trts`` chars.

**Args**

- `x` — a list of treatment labels, or a matrix whose row/column names are the treatments.
- `nchar_trts` — maximum label length.
- `row` — use the row names when ``x`` is a matrix.

**Returns**

list of abbreviations.  When truncation would create duplicates the labels are abbreviated with R's ``abbreviate()`` (drop spaces, then lower-case vowels, then consonants, from the end of words), e.g. ``["aa1", "aa2", "bb"]`` -> ``["a1", "a2", "bb"]``.

## `vikor`

```python
vikor(x, weights=None, v=0.5)
```

*Mirrors R's `vikor`.*

``netmeta::vikor`` — VIKOR multicriteria ranking.

``x`` holds one score in ``[0, 1]`` per treatment and outcome (higher is
better).  Equations (transcribed from ``netmeta:::vikor_internal``)::

    dist = 1 - x;  wnm = dist * weights
    S_i  = sum_j wnm_ij            (group utility)
    R_i  = max_j wnm_ij            (individual regret)
    Q_i  = v·(S_i - min S)/(max S - min S)
           + (1 - v)·(R_i - min R)/(max R - min R)

**Args**

- `x` — ``NetmetaResult`` (P-scores are used), ``pandas.DataFrame`` or a 2-D array-like of scores; row names are the treatments.
- `weights` — outcome weights (default: equal).
- `v` — weight of the group-utility strategy (``0.5`` = balanced).

**Returns**

- `:class` — `VikorResult` sorted by increasing ``Q`` (best first).

## `VikorResult`

```python
VikorResult(Q, S, R, trts, weights, v, dropped)
```

Result of :func:`vikor` — VIKOR ranking statistics.

## `subgroup`

```python
subgroup(result, subgroup, only_connected=False, method_tau=None, level_ma=None, common=True, random=True)
```

*Mirrors R's `subgroup`.*

``netmeta::subgroup`` — refit the network within each subgroup.

**Args**

- `result` — a :class:`NetmetaResult`.
- `subgroup` — one subgroup label per **study**.
- `only_connected` — error out when a subgroup network is disconnected.
- `method_tau` — heterogeneity estimator (defaults to the model's).
- `level_ma` — confidence level (defaults to the model's).
- `common, random` — which models to record.

**Returns**

- `:class` — `NetsubgroupResult` with per-level 3-D arrays
``TE_common``/``seTE_common``/``TE_random``/``seTE_random`` of shape
``(n, n, n_levels)`` and the fitted networks.

## `NetsubgroupResult`

```python
NetsubgroupResult(**kw)
```

Result of :func:`subgroup` for a netmeta object.

## `nettable`

```python
nettable(result, level=0.95)
```

*Mirrors R's `nettable`.*

``netmeta::nettable`` — direct evidence per comparison.

Returns one row per treatment comparison with the number of studies, the
number of two-arm studies, the direct estimate and the network estimate.

**Args**

- `result` — a :class:`NetmetaResult`.
- `level` — confidence level (kept for API parity).

**Returns**

- `:class` — `NettableResult`.

## `NettableResult`

```python
NettableResult(rows, trts)
```

Result of :func:`nettable` — per-comparison summary of the network.

## `createC`

```python
createC(trts, inactive=None, sep_comps='+')
```

*Mirrors R's `createC`.*

``netmeta::createC`` — component incidence matrix ``(C, comps, names)``.

## `invmat`

```python
invmat(x)
```

*Mirrors R's `invmat`.*

``netmeta::invmat`` — Moore-Penrose pseudoinverse of a Laplacian.

## `hatmatrix`

```python
hatmatrix(x, method='Ruecker', type='full', common=None, random=None)
```

*Mirrors R's `hatmatrix`.*

``netmeta::hatmatrix`` — hat matrix of a network.

Supported: ``method="Ruecker"`` (the default, as in netmeta 3.7-0 — the hat
matrix stored on the fitted object), ``method="Davies"`` with ``type`` in
``short``/``long``/``full`` (via ``hatmatrix.aggr``), and
``method="Krahn"`` (the design-based hat matrix from the decomposition).

## `comps`

```python
comps(x, trts=None, sep_comps='+')
```

*Mirrors R's `comps`.*

``netmeta::comps`` — split treatment names into components.

**Args**

- `x` — a treatment label, or a sequence of them.
- `trts` — optional reference list of *components*.  When given, every component found in ``x`` must appear there (netmeta's ``setref`` check) and the components are returned in ``trts`` order, which is what ``netmeta:::comps`` does with its ``trts`` argument.
- `sep_comps` — component separator.

**Returns**

The component list for a single label, or a list of component lists.

## `combinations`

```python
combinations(x, sep_comps='+')
```

*Mirrors R's `combinations`.*

``netmeta::combinations`` — all treatment combinations of ``x``.

## `netimpact`

```python
netimpact(result, seTE_ignore=None)
```

*Mirrors R's `netimpact`.*

``netmeta::netimpact`` — leave-one-study-out impact on all comparisons.

Each study is down-weighted (its standard errors are replaced by
``seTE_ignore``) and the network is re-estimated with the original ``tau``
as ``tau_preset``; the reported impact is ``1 - (seTE / seTE_removed)^2``.

**Returns**

- `:class` — `NetimpactResult` — ``impact_common``/``impact_random`` map
- `study labels to ``{comparison` — impact}`` dictionaries.

## `NetimpactResult`

```python
NetimpactResult(impact_common, impact_random, comparisons, studies, ignored_comparisons, seTE_ignore)
```

*Mirrors R's `netimpact`.*

Result of :func:`netimpact` (``netmeta::netimpact``).

## `netpath`

```python
netpath(result, node1, node2, random=True)
```

*Mirrors R's `netpath`.*

``netmeta::netpath`` — path-by-path inconsistency (Koenig et al. 2013).

Builds the evidence-flow network from the row of the Davies hat matrix that
corresponds to ``node1`` vs ``node2``, enumerates every simple path between
the two nodes, and tests whether the direct evidence along those paths is
consistent with the network estimate.

**Args**

- `result` — a :class:`NetmetaResult`.
- `node1, node2` — the treatment contrast of interest.
- `random` — use the random-effects model.

**Returns**

A dict with ``results`` (per-path ``Q``, ``df``, ``pval``), ``A_matrix``, ``Sigma`` and ``theta_p`` — the fields of the R ``netpath`` object.
