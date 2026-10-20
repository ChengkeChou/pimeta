# Effect sizes

*93 public names, generated from `pimeta.__all__` and the docstrings in `src/pimeta/`. Do not edit this page by hand — run `python tools/gen_api_docs.py`.*

| Name | Mirrors in R |
|---|---|
| [`escalc`](#escalc) | — |
| [`EscalcResult`](#escalcresult) | — |
| [`RATIO_TYPES`](#ratio_types) | — |
| [`DIFF_TYPES`](#diff_types) | — |
| [`fmtp`](#fmtp) | fmtp |
| [`fmtp2`](#fmtp2) | fmtp2 |
| [`fmtt`](#fmtt) | fmtt |
| [`fmtx`](#fmtx) | fmtx |
| [`dfround`](#dfround) | dfround |
| [`asin2ir`](#asin2ir) | asin2ir |
| [`asin2p`](#asin2p) | asin2p |
| [`p2asin`](#p2asin) | p2asin |
| [`cor2z`](#cor2z) | cor2z |
| [`z2cor`](#z2cor) | z2cor |
| [`logit2p`](#logit2p) | logit2p |
| [`p2logit`](#p2logit) | p2logit |
| [`smd2or`](#smd2or) | smd2or |
| [`or2smd`](#or2smd) | or2smd |
| [`logVR2VE`](#logvr2ve) | logVR2VE |
| [`VE2logVR`](#ve2logvr) | VE2logVR |
| [`rd`](#rd) | rd |
| [`nnt`](#nnt) | nnt |
| [`RdResult`](#rdresult) | — |
| [`NntResult`](#nntresult) | — |
| [`conv_delta`](#conv_delta) | conv.delta |
| [`conv_wald`](#conv_wald) | conv.wald |
| [`conv_fivenum`](#conv_fivenum) | conv.fivenum |
| [`conv_2x2`](#conv_2x2) | conv.2x2 |
| [`Conv2x2Result`](#conv2x2result) | conv.2x2 |
| [`ConvDeltaResult`](#convdeltaresult) | conv.delta |
| [`ConvWaldResult`](#convwaldresult) | conv.wald |
| [`ConvFivenumResult`](#convfivenumresult) | conv.fivenum |
| [`to_long`](#to_long) | to.long |
| [`to_wide`](#to_wide) | to.wide |
| [`to_table`](#to_table) | to.table |
| [`transf_abt`](#transf_abt) | transf.abt |
| [`transf_ahw`](#transf_ahw) | transf.ahw |
| [`transf_arcsin`](#transf_arcsin) | transf.arcsin |
| [`transf_clestod`](#transf_clestod) | transf.clestod |
| [`transf_dtobesd`](#transf_dtobesd) | transf.dtobesd |
| [`transf_dtocles`](#transf_dtocles) | transf.dtocles |
| [`transf_dtocles_int`](#transf_dtocles_int) | transf.dtocles.int |
| [`transf_dtocliffd`](#transf_dtocliffd) | transf.dtocliffd |
| [`transf_dtolnor_logis`](#transf_dtolnor_logis) | transf.dtolnor.logis |
| [`transf_dtolnor_norm`](#transf_dtolnor_norm) | transf.dtolnor.norm |
| [`transf_dtomd`](#transf_dtomd) | transf.dtomd |
| [`transf_dtoovl`](#transf_dtoovl) | transf.dtoovl |
| [`transf_dtorbis`](#transf_dtorbis) | transf.dtorbis |
| [`transf_dtorpb`](#transf_dtorpb) | transf.dtorpb |
| [`transf_dtou1`](#transf_dtou1) | transf.dtou1 |
| [`transf_dtou2`](#transf_dtou2) | transf.dtou2 |
| [`transf_dtou3`](#transf_dtou3) | transf.dtou3 |
| [`transf_exp_int`](#transf_exp_int) | transf.exp.int |
| [`transf_exp_mode`](#transf_exp_mode) | transf.exp.mode |
| [`transf_iabt`](#transf_iabt) | transf.iabt |
| [`transf_iabt_int`](#transf_iabt_int) | transf.iabt.int |
| [`transf_iabt_mode`](#transf_iabt_mode) | transf.iabt.mode |
| [`transf_iahw`](#transf_iahw) | transf.iahw |
| [`transf_iahw_int`](#transf_iahw_int) | transf.iahw.int |
| [`transf_iahw_mode`](#transf_iahw_mode) | transf.iahw.mode |
| [`transf_iarcsin`](#transf_iarcsin) | transf.iarcsin |
| [`transf_iarcsin_int`](#transf_iarcsin_int) | transf.iarcsin.int |
| [`transf_iarcsin_mode`](#transf_iarcsin_mode) | transf.iarcsin.mode |
| [`transf_iirft`](#transf_iirft) | transf.iirft |
| [`transf_ilogit`](#transf_ilogit) | transf.ilogit |
| [`transf_ilogit_int`](#transf_ilogit_int) | transf.ilogit.int |
| [`transf_ilogit_mode`](#transf_ilogit_mode) | transf.ilogit.mode |
| [`transf_ipft`](#transf_ipft) | transf.ipft |
| [`transf_ipft_hm`](#transf_ipft_hm) | transf.ipft.hm |
| [`transf_iprobit`](#transf_iprobit) | transf.iprobit |
| [`transf_iprobit_int`](#transf_iprobit_int) | transf.iprobit.int |
| [`transf_iprobit_mode`](#transf_iprobit_mode) | transf.iprobit.mode |
| [`transf_irft`](#transf_irft) | transf.irft |
| [`transf_isqrt`](#transf_isqrt) | transf.isqrt |
| [`transf_lnortod_logis`](#transf_lnortod_logis) | transf.lnortod.logis |
| [`transf_lnortod_norm`](#transf_lnortod_norm) | transf.lnortod.norm |
| [`transf_lnortord`](#transf_lnortord) | transf.lnortord |
| [`transf_lnortorr`](#transf_lnortorr) | transf.lnortorr |
| [`transf_lnortortet_digby`](#transf_lnortortet_digby) | transf.lnortortet.digby |
| [`transf_lnortortet_pearson`](#transf_lnortortet_pearson) | transf.lnortortet.pearson |
| [`transf_logit`](#transf_logit) | transf.logit |
| [`transf_pft`](#transf_pft) | transf.pft |
| [`transf_probit`](#transf_probit) | transf.probit |
| [`transf_r2toz`](#transf_r2toz) | transf.r2toz |
| [`transf_rpbtod`](#transf_rpbtod) | transf.rpbtod |
| [`transf_rpbtorbis`](#transf_rpbtorbis) | transf.rpbtorbis |
| [`transf_rtod`](#transf_rtod) | transf.rtod |
| [`transf_rtorpb`](#transf_rtorpb) | transf.rtorpb |
| [`transf_rtoz`](#transf_rtoz) | transf.rtoz |
| [`transf_ztor`](#transf_ztor) | transf.ztor |
| [`transf_ztor_int`](#transf_ztor_int) | transf.ztor.int |
| [`transf_ztor_mode`](#transf_ztor_mode) | transf.ztor.mode |
| [`transf_ztor2`](#transf_ztor2) | transf.ztor2 |

## `escalc`

```python
escalc(measure: 'str', ai=None, bi=None, ci=None, di=None, n1i=None, n2i=None, m1i=None, m2i=None, sd1i=None, sd2i=None, x1i=None, x2i=None, t1i=None, t2i=None, xi=None, ni=None, ri=None, add: 'float' = 0.5, drop00: 'bool' = True, vtype: 'str' = 'LS', data=None) -> 'EscalcResult'
```

Compute effect sizes and variances.

Column inputs are parallel lists. For 2x2 binary measures pass ``ai, bi,
ci, di`` (treatment/control events and non-events) or ``ai, ci, n1i, n2i``
(events plus group sizes). ``add`` is the continuity correction for OR
(0.5 by default, matching metafor). ``drop00`` drops rows with zero events
in both arms (metafor's default for RR).  ``vtype`` selects the variance
estimator for SMD/SMDH (metafor's ``vtype``, default ``"LS"``).

``data`` lets every column argument be given as a *name* instead of a
vector (``escalc("RR", "ai", "bi", "ci", "di", data=df)``); it accepts a
dict, a pandas/polars ``DataFrame``, a structured array or a list of
records.

## `EscalcResult`

```python
EscalcResult(measure: 'str', yi: 'list[float]', vi: 'list[float]', ni: 'list[float]', k: 'int') -> None
```

Vectorized effect-size output (one row per study).

## `RATIO_TYPES`

*Constant (`set`):* `HR`, `IRR`, `OR`, `PETO`, `PLN`, `PLO`, `RR`

## `DIFF_TYPES`

*Constant (`set`):* `AS`, `MD`, `PAS`, `PFT`, `RD`, `SMD`, `WMD`

## `fmtp`

```python
fmtp(p, digits=4, pname='', equal=False, sep=False, add0=False, quote=False)
```

*Mirrors R's `fmtp`.*

``fmtp`` — format p-values (``"<0.0001"`` style).

**Args**

- `p` — p-value(s).
- `digits` — number of decimals shown above the cut-off.
- `pname` — optional name prefix (e.g. ``"p"``).
- `equal` — insert ``"="`` before the value.
- `sep` — insert a space between the name/equal sign and the value.
- `add0` — keep the leading zero in ``"<0.0001"``.
- `quote` — keep the result as ``str`` (R sets ``noquote`` when ``False``).

**Returns**

A formatted ``str`` (scalar input) or list of ``str``.

## `fmtp2`

```python
fmtp2(p, cutoff=(0.001, 0.06), pname='p', sep=True, add0=False, quote=False)
```

*Mirrors R's `fmtp2`.*

``fmtp2`` — format p-values with two cut-offs (meta's ``pval7`` style).

## `fmtt`

```python
fmtt(val, tname='', df=None, df1=None, df2=None, pval=None, digits=4, pname='p-val', format=1, sep=True, quote=False, call=False, **kwargs)
```

*Mirrors R's `fmtt`.*

``fmtt`` — format a test statistic together with its p-value.

**Args**

- `val` — test statistic (scalar).
- `tname` — name of the statistic, e.g. ``"z"``, ``"t"``, ``"Q"``, ``"F"``.
- `df` — single degrees-of-freedom value (``format=1``).
- `df1, df2` — numerator/denominator df (used instead of ``df``).
- `pval` — the p-value.
- `digits` — int or ``(test_digits, pval_digits)``.
- `pname` — label used for the p-value.
- `format` — 1 → ``"t(df = x) = ..., p-val = ..."``, 2 → ``"t = ..., df = ..., p-val = ..."``.
- `sep` — use spaces around ``=``.
- `quote` — keep as ``str``.

## `fmtx`

```python
fmtx(x, digits=4, flag='', quote=False, **kwargs)
```

*Mirrors R's `fmtx`.*

``fmtx`` — format numbers with fixed decimals (``formatC`` semantics).

Keyword options mirroring metafor: ``addwidth``, ``drop0ifint``, ``add0``,
``thresh``, ``postfix``.

## `dfround`

```python
dfround(x, digits=0, drop0=True)
```

*Mirrors R's `dfround`.*

``dfround`` — round the numeric columns of a table to ``digits``.

Works on a ``{name: column}`` mapping (returned as a new dict) or on a pandas
``DataFrame`` when pandas happens to be installed (returned as a DataFrame).
Non-numeric columns are passed through untouched.

**Args**

- `x` — mapping of column name -> values, or a pandas ``DataFrame``.
- `digits` — int or per-column sequence.
- `drop0` — round (``True``) or format as fixed-decimal strings (``False``).

## `asin2ir`

```python
asin2ir(x, time=None)
```

*Mirrors R's `asin2ir`.*

``meta::asin2ir`` �?inverse Freeman-Tukey double arcsine to an incidence rate.

## `asin2p`

```python
asin2p(x, n=None)
```

*Mirrors R's `asin2p`.*

``meta::asin2p`` �?inverse Freeman-Tukey double arcsine to a proportion.

## `p2asin`

```python
p2asin(x)
```

*Mirrors R's `p2asin`.*

``meta::p2asin`` �?proportion to arcsine.

## `cor2z`

```python
cor2z(x)
```

*Mirrors R's `cor2z`.*

``meta::cor2z`` �?correlation to Fisher's z.

## `z2cor`

```python
z2cor(x)
```

*Mirrors R's `z2cor`.*

``meta::z2cor`` �?Fisher's z to a correlation.

## `logit2p`

```python
logit2p(x)
```

*Mirrors R's `logit2p`.*

``meta::logit2p`` �?logit to a proportion.

## `p2logit`

```python
p2logit(x)
```

*Mirrors R's `p2logit`.*

``meta::p2logit`` �?proportion to logit.

## `smd2or`

```python
smd2or(smd, se_smd, method='HH')
```

*Mirrors R's `smd2or`.*

``meta::smd2or`` �?SMD to log odds ratio, then pool with ``metagen``.

**Args**

- `smd` — standardized mean differences.
- `se_smd` — their standard errors.
- `method` — ``"HH"`` (Hasselblad-Hedges, ``pi/sqrt(3)``) or ``"CS"`` (Chinn, ``1.65``).

**Returns**

- `:class` — `~.metagen.MetagenResult` on the log-OR scale (``sm="OR"``).

## `or2smd`

```python
or2smd(lnOR, selnOR, method='HH')
```

*Mirrors R's `or2smd`.*

``meta::or2smd`` �?log odds ratio to SMD, then pool with ``metagen``.

**Args**

- `lnOR` — log odds ratios.
- `selnOR` — their standard errors.
- `method` — ``"HH"`` (``sqrt(3)/pi``) or ``"CS"`` (``1/1.65``).

**Returns**

- `:class` — `~.metagen.MetagenResult` on the SMD scale (``sm="SMD"``).

## `logVR2VE`

```python
logVR2VE(x)
```

*Mirrors R's `logVR2VE`.*

``meta::logVR2VE`` �?log variance ratio to a percentage reduction.

## `VE2logVR`

```python
VE2logVR(x)
```

*Mirrors R's `VE2logVR`.*

``meta::VE2logVR`` �?percentage reduction to a log variance ratio.

## `rd`

```python
rd(result, p_c=None, common=True, random=True, small_values='desirable')
```

*Mirrors R's `rd`.*

``meta::rd`` �?risk difference / absolute risk reduction from a pooled result.

**Args**

- `result` — :class:`~.metagen.MetagenResult` with ``sm`` in ``{"RD", "RR", "OR", "HR"}``.
- `p_c` — control-group risk(s). Defaults to ``0.1 .. 0.9`` by 0.1.
- `common` — include the common-effect estimate.
- `random` — include the random-effects estimate.
- `small_values` — ``"desirable"`` or ``"undesirable"`` (sign convention).

**Returns**

- `:class` — `RdResult` with ``common``/``random`` dicts of arrays.

## `nnt`

```python
nnt(result, p_c=None, common=True, random=True, small_values='desirable')
```

*Mirrors R's `nnt`.*

``meta::nnt`` �?number needed to treat from a pooled result.

See :func:`rd` for the arguments; ``NNT = 1 / RD`` with the sign convention
from ``small_values``.

## `RdResult`

```python
RdResult(common, random, sm, small_values)
```

Container returned by :func:`rd`.

## `NntResult`

```python
NntResult(sm, common, random, nnt_common, nnt_random)
```

Container returned by :func:`nnt`.

## `conv_delta`

```python
conv_delta(yi, vi, transf)
```

*Mirrors R's `conv.delta`.*

``metafor::conv.delta`` — transform an effect size and its variance.

Applies ``transf`` to ``yi`` and propagates ``vi`` with the first-order
delta method, i.e. ``vi_new = transf'(yi)^2 * vi``.

**Args**

- `yi` — effect sizes.
- `vi` — sampling variances.
- `transf` — callable applied elementwise to ``yi``.

**Returns**

- `:class` — `ConvDeltaResult` with the transformed ``yi`` and ``vi``.

## `conv_wald`

```python
conv_wald(out, ci_lb, ci_ub, level=95.0, transf=None)
```

*Mirrors R's `conv.wald`.*

``metafor::conv.wald`` — recover ``yi``/``vi`` from an estimate and its CI.

``vi = ((ci.ub - ci.lb) / (2 * qnorm(1 - (1 - level/100)/2)))^2``.

**Args**

- `out` — reported estimates (on the analysis scale unless ``transf`` is given, in which case ``transf`` maps them onto that scale).
- `ci_lb, ci_ub` — the corresponding confidence limits.
- `level` — confidence level in **percent** (metafor convention, 95 = 95%).
- `transf` — optional transform applied to ``out``/``ci_lb``/``ci_ub``.

**Returns**

- `:class` — `ConvWaldResult`.

## `conv_fivenum`

```python
conv_fivenum(min=None, q1=None, median=None, q3=None, max=None, n=None, method='luo/wan/shi', test=True)
```

*Mirrors R's `conv.fivenum`.*

``metafor::conv.fivenum`` — reconstruct mean/SD from a five-number summary.

Transcribes metafor's three cases (``case1``: min/median/max only,
``case2``: q1/median/q3 only, ``case3``: all five) for the
``"luo/wan/shi"`` (default), ``"wan2014"``, ``"hozo2005"`` and
``"bland2015"`` estimators.

The default (Luo-Wan-Shi) uses the weighted combinations of the range- and
IQR-based approximations that metafor implements; ``"wan2014"`` uses the
unweighted Wan et al. (2014) formulas (for ``case3`` the SD is the equally
weighted mean of the range- and IQR-based estimates).

**Args**

- `min, q1, median, q3, max` — the five-number summary (arrays or scalars).
- `n` — sample size(s).
- `method` — estimator name.
- `test` — when ``True`` (metafor default), skip observations whose summary is significantly skewed (the estimated value is then ``NaN``).

**Returns**

- `:class` — `ConvFivenumResult` with ``mean``, ``sd``, ``case`` and ``sig``.

**Note**

Only ``dist="norm"`` is supported; metafor's log-normal back-transform (``dist="lnorm"``) and the ``estmeansd``/``metaBLUE`` based methods (``qe``/``bc``/``mln``/``blue``) are not transcribed.

## `conv_2x2`

```python
conv_2x2(ori=None, ri=None, x2i=None, ni=None, n1i=None, n2i=None, sens=None, spec=None, ppv=None, npv=None, correct=True, drop01=True, method='optim', var_names=('ai', 'bi', 'ci', 'di'))
```

*Mirrors R's `conv.2x2`.*

``metafor::conv.2x2`` — reconstruct 2x2 cell frequencies.

Cell counts can be recovered from either an odds ratio (``ori``), a phi
coefficient (``ri``) or a chi-square statistic (``x2i``) together with the
marginal counts ``n1i``/``n2i``, or from at least three of
sensitivity/specificity/PPV/NPV.  This is a faithful transcription of
``metafor::conv.2x2`` including the ``"simple"``, ``"optim"`` and
``"search"`` reconstruction methods.

**Args**

- `ori, ri, x2i` — odds ratio, phi coefficient, chi-square statistic.
- `ni` — total sample size.
- `n1i, n2i` — marginal counts (e.g. number of cases and controls).
- `sens, spec, ppv, npv` — diagnostic accuracy measures.
- `correct` — continuity correction for the ``x2i`` -> ``ri`` conversion.
- `drop01` — skip tables whose sens/spec/PPV/NPV equal 0 or 1.
- `method` — ``"simple"``, ``"optim"`` (default) or ``"search"``.
- `var_names` — names of the four output cells.

**Returns**

- `:class` — `Conv2x2Result` with ``ai``, ``bi``, ``ci``, ``di``.

## `Conv2x2Result`

```python
Conv2x2Result(ai, bi, ci, di, var_names)
```

*Mirrors R's `conv.2x2`.*

Result of :func:`conv_2x2` (``metafor::conv.2x2``).

## `ConvDeltaResult`

```python
ConvDeltaResult(yi, vi)
```

*Mirrors R's `conv.delta`.*

Result of :func:`conv_delta` (``metafor::conv.delta``).

## `ConvWaldResult`

```python
ConvWaldResult(yi, vi)
```

*Mirrors R's `conv.wald`.*

Result of :func:`conv_wald` (``metafor::conv.wald``).

## `ConvFivenumResult`

```python
ConvFivenumResult(mean, sd, case, sig)
```

*Mirrors R's `conv.fivenum`.*

Result of :func:`conv_fivenum` (``metafor::conv.fivenum``).

## `to_long`

```python
to_long(measure='RR', ai=None, bi=None, ci=None, di=None, n1i=None, n2i=None, event=None, n=None, grp=None, study=None)
```

*Mirrors R's `to.long`.*

``metafor::to.long`` — reshape arm-level data from wide to long format.

Accepts either a 2x2 layout (``ai``/``bi``/``ci``/``di`` or
``n1i``/``n2i``/``event``) and returns a long table with one row per arm.

**Args**

- `measure` — effect-size measure (kept for parity with metafor).
- `ai, bi, ci, di` — 2x2 cell counts (experimental/control events and non-events).
- `n1i, n2i` — arm sample sizes (used when ``event``/``n`` are given).
- `event, n` — generic per-arm events and totals.
- `grp` — optional arm labels (defaults to ``1, 2``).
- `study` — optional study labels.

**Returns**

``dict`` of columns with 2 rows per study.

## `to_wide`

```python
to_wide(data, study='study', group='group', event='event', n='n')
```

*Mirrors R's `to.wide`.*

``metafor::to.wide`` — reshape a long arm-level table back to wide format.

**Args**

- `data` — mapping of column name -> values (the output of :func:`to_long`).
- `study, group, event, n` — column names to use.

**Returns**

``dict`` with ``ai``/``bi``/``ci``/``di`` and ``n1i``/``n2i``.

## `to_table`

```python
to_table(ai, bi, ci, di, study=None)
```

*Mirrors R's `to.table`.*

``metafor::to.table`` — build per-study 2x2 contingency tables.

**Args**

- `ai, bi, ci, di` — 2x2 cell counts.
- `study` — optional study labels.

**Returns**

List of ``2x2`` nested lists, one per study.

## `transf_abt`

```python
transf_abt(xi)
```

*Mirrors R's `transf.abt`.*

``transf.abt`` — ``-log(1-x)`` (back-transformed proportion).

## `transf_ahw`

```python
transf_ahw(xi)
```

*Mirrors R's `transf.ahw`.*

``transf.ahw`` — ``1-(1-x)^(1/3)`` (Arcsine-Hellinger / Whittemore).

## `transf_arcsin`

```python
transf_arcsin(xi)
```

*Mirrors R's `transf.arcsin`.*

``transf.arcsin`` — arcsine (angular) transform of a proportion.

## `transf_clestod`

```python
transf_clestod(xi)
```

*Mirrors R's `transf.clestod`.*

``transf.clestod`` — CLES (common language effect size) to Cohen's d.

## `transf_dtobesd`

```python
transf_dtobesd(xi)
```

*Mirrors R's `transf.dtobesd`.*

``transf.dtobesd`` — Cohen's d to Bonett's unbiased d.

## `transf_dtocles`

```python
transf_dtocles(xi)
```

*Mirrors R's `transf.dtocles`.*

``transf.dtocles`` — Cohen's d to common-language effect size.

## `transf_dtocles_int`

```python
transf_dtocles_int(xi, targs=None)
```

*Mirrors R's `transf.dtocles.int`.*

``transf.dtocles.int`` — E[Phi(theta/sqrt(2))] under ``N(xi, tau2)``.

## `transf_dtocliffd`

```python
transf_dtocliffd(xi)
```

*Mirrors R's `transf.dtocliffd`.*

``transf.dtocliffd`` — Cohen's d to Cliff's delta.

## `transf_dtolnor_logis`

```python
transf_dtolnor_logis(xi)
```

*Mirrors R's `transf.dtolnor.logis`.*

``transf.dtolnor.logis`` — d to log odds ratio (logistic distribution).

## `transf_dtolnor_norm`

```python
transf_dtolnor_norm(xi)
```

*Mirrors R's `transf.dtolnor.norm`.*

``transf.dtolnor.norm`` — d to log odds ratio (normal distribution, 1.65).

## `transf_dtomd`

```python
transf_dtomd(xi, targs=None)
```

*Mirrors R's `transf.dtomd`.*

``transf.dtomd`` — Cohen's d to a raw mean difference (needs ``sd``).

## `transf_dtoovl`

```python
transf_dtoovl(xi)
```

*Mirrors R's `transf.dtoovl`.*

``transf.dtoovl`` — d to the overlap coefficient (U1).

## `transf_dtorbis`

```python
transf_dtorbis(xi, n1i=None, n2i=None)
```

*Mirrors R's `transf.dtorbis`.*

``transf.dtorbis`` — Cohen's d to the biserial correlation.

## `transf_dtorpb`

```python
transf_dtorpb(xi, n1i=None, n2i=None)
```

*Mirrors R's `transf.dtorpb`.*

``transf.dtorpb`` — Cohen's d to the point-biserial correlation.

## `transf_dtou1`

```python
transf_dtou1(xi)
```

*Mirrors R's `transf.dtou1`.*

``transf.dtou1`` — d to Cohen's U1.

## `transf_dtou2`

```python
transf_dtou2(xi)
```

*Mirrors R's `transf.dtou2`.*

``transf.dtou2`` — d to Cohen's U2.

## `transf_dtou3`

```python
transf_dtou3(xi)
```

*Mirrors R's `transf.dtou3`.*

``transf.dtou3`` — d to Cohen's U3.

## `transf_exp_int`

```python
transf_exp_int(xi, targs=None)
```

*Mirrors R's `transf.exp.int`.*

``transf.exp.int`` — ``exp(xi + tau2/2)`` (mean of the log-normal).

## `transf_exp_mode`

```python
transf_exp_mode(xi, targs=None)
```

*Mirrors R's `transf.exp.mode`.*

``transf.exp.mode`` — ``exp(xi - tau2)`` (mode of the log-normal).

## `transf_iabt`

```python
transf_iabt(xi)
```

*Mirrors R's `transf.iabt`.*

``transf.iabt`` — inverse of the log(1-x) transform.

## `transf_iabt_int`

```python
transf_iabt_int(xi, targs=None)
```

*Mirrors R's `transf.iabt.int`.*

``transf.iabt.int`` — E[1-exp(-theta)] under a truncated normal.

## `transf_iabt_mode`

```python
transf_iabt_mode(xi, targs=None)
```

*Mirrors R's `transf.iabt.mode`.*

``transf.iabt.mode`` — mode of the inverse log(1-x) transform.

## `transf_iahw`

```python
transf_iahw(xi)
```

*Mirrors R's `transf.iahw`.*

``transf.iahw`` — inverse of the ``1-(1-x)^(1/3)`` transform.

## `transf_iahw_int`

```python
transf_iahw_int(xi, targs=None)
```

*Mirrors R's `transf.iahw.int`.*

``transf.iahw.int`` — E[1-(1-theta)^3] under a truncated normal.

## `transf_iahw_mode`

```python
transf_iahw_mode(xi, targs=None)
```

*Mirrors R's `transf.iahw.mode`.*

``transf.iahw.mode`` — mode of the inverse ``1-(1-x)^(1/3)`` transform.

## `transf_iarcsin`

```python
transf_iarcsin(xi)
```

*Mirrors R's `transf.iarcsin`.*

``transf.iarcsin`` — inverse arcsine (truncated to [0, 1]).

## `transf_iarcsin_int`

```python
transf_iarcsin_int(xi, targs=None)
```

*Mirrors R's `transf.iarcsin.int`.*

``transf.iarcsin.int`` — E[sin(theta)^2] under ``theta ~ N(xi, tau2)``.

## `transf_iarcsin_mode`

```python
transf_iarcsin_mode(xi, targs=None)
```

*Mirrors R's `transf.iarcsin.mode`.*

``transf.iarcsin.mode`` — mode of the back-transformed arcsine.

## `transf_iirft`

```python
transf_iirft(xi, ti)
```

*Mirrors R's `transf.iirft`.*

``transf.iirft`` — inverse of ``transf.irft`` (back to a rate).

## `transf_ilogit`

```python
transf_ilogit(xi)
```

*Mirrors R's `transf.ilogit`.*

``transf.ilogit`` — inverse logit.

## `transf_ilogit_int`

```python
transf_ilogit_int(xi, targs=None)
```

*Mirrors R's `transf.ilogit.int`.*

``transf.ilogit.int`` — E[logistic(theta)] under ``theta ~ N(xi, tau2)``.

## `transf_ilogit_mode`

```python
transf_ilogit_mode(xi, targs=None)
```

*Mirrors R's `transf.ilogit.mode`.*

``transf.ilogit.mode`` — mode of the back-transformed logit.

Mirrors metafor's implementation: scan ``seq(0, 1, length=1e5)`` and only
return a root when the density is unimodal.

## `transf_ipft`

```python
transf_ipft(xi, ni)
```

*Mirrors R's `transf.ipft`.*

``transf.ipft`` — inverse Freeman-Tukey double arcsine.

## `transf_ipft_hm`

```python
transf_ipft_hm(xi, targs)
```

*Mirrors R's `transf.ipft.hm`.*

``transf.ipft.hm`` — inverse FT using the harmonic mean sample size.

## `transf_iprobit`

```python
transf_iprobit(xi)
```

*Mirrors R's `transf.iprobit`.*

``transf.iprobit`` — inverse probit.

## `transf_iprobit_int`

```python
transf_iprobit_int(xi, targs=None)
```

*Mirrors R's `transf.iprobit.int`.*

``transf.iprobit.int`` — E[Phi(theta)] under ``theta ~ N(xi, tau2)``.

## `transf_iprobit_mode`

```python
transf_iprobit_mode(xi, targs=None)
```

*Mirrors R's `transf.iprobit.mode`.*

``transf.iprobit.mode`` — mode of the back-transformed probit.

## `transf_irft`

```python
transf_irft(xi, ti)
```

*Mirrors R's `transf.irft`.*

``transf.irft`` — inverse of the square-root rate transform.

## `transf_isqrt`

```python
transf_isqrt(xi)
```

*Mirrors R's `transf.isqrt`.*

``transf.isqrt`` — inverse square-root transform (``x^2``).

## `transf_lnortod_logis`

```python
transf_lnortod_logis(xi)
```

*Mirrors R's `transf.lnortod.logis`.*

``transf.lnortod.logis`` — log OR to Cohen's d (logistic).

## `transf_lnortod_norm`

```python
transf_lnortod_norm(xi)
```

*Mirrors R's `transf.lnortod.norm`.*

``transf.lnortod.norm`` — log OR to Cohen's d (normal).

## `transf_lnortord`

```python
transf_lnortord(xi, pc)
```

*Mirrors R's `transf.lnortord`.*

``transf.lnortord`` — log OR to a risk difference (needs ``pc``).

## `transf_lnortorr`

```python
transf_lnortorr(xi, pc)
```

*Mirrors R's `transf.lnortorr`.*

``transf.lnortorr`` — log OR to a risk ratio (needs ``pc``).

## `transf_lnortortet_digby`

```python
transf_lnortortet_digby(xi)
```

*Mirrors R's `transf.lnortortet.digby`.*

``transf.lnortortet.digby`` — log OR to tetrachoric r (Digby).

## `transf_lnortortet_pearson`

```python
transf_lnortortet_pearson(xi)
```

*Mirrors R's `transf.lnortortet.pearson`.*

``transf.lnortortet.pearson`` — log OR to tetrachoric r (Pearson).

## `transf_logit`

```python
transf_logit(xi)
```

*Mirrors R's `transf.logit`.*

``transf.logit`` — logit transform of a proportion.

## `transf_pft`

```python
transf_pft(xi, ni)
```

*Mirrors R's `transf.pft`.*

``transf.pft`` — Freeman-Tukey double arcsine of a proportion.

## `transf_probit`

```python
transf_probit(xi)
```

*Mirrors R's `transf.probit`.*

``transf.probit`` — probit transform of a proportion.

## `transf_r2toz`

```python
transf_r2toz(xi)
```

*Mirrors R's `transf.r2toz`.*

``transf.r2toz`` — R^2 to Fisher's z (clamped to [0, 1]).

## `transf_rpbtod`

```python
transf_rpbtod(xi, n1i=None, n2i=None)
```

*Mirrors R's `transf.rpbtod`.*

``transf.rpbtod`` — point-biserial r to Cohen's d.

## `transf_rpbtorbis`

```python
transf_rpbtorbis(xi, pi=0.5)
```

*Mirrors R's `transf.rpbtorbis`.*

``transf.rpbtorbis`` — point-biserial r to the biserial correlation.

## `transf_rtod`

```python
transf_rtod(xi, n1i=None, n2i=None)
```

*Mirrors R's `transf.rtod`.*

``transf.rtod`` — biserial correlation to Cohen's d.

## `transf_rtorpb`

```python
transf_rtorpb(xi, pi=0.5)
```

*Mirrors R's `transf.rtorpb`.*

``transf.rtorpb`` — point-biserial r to Cohen's d.

## `transf_rtoz`

```python
transf_rtoz(xi)
```

*Mirrors R's `transf.rtoz`.*

``transf.rtoz`` — Fisher's r-to-z (clamped to [-1, 1]).

## `transf_ztor`

```python
transf_ztor(xi)
```

*Mirrors R's `transf.ztor`.*

``transf.ztor`` — Fisher's z-to-r.

## `transf_ztor_int`

```python
transf_ztor_int(xi, targs=None)
```

*Mirrors R's `transf.ztor.int`.*

``transf.ztor.int`` — E[tanh(theta)] under ``theta ~ N(xi, tau2)``.

## `transf_ztor_mode`

```python
transf_ztor_mode(xi, targs=None)
```

*Mirrors R's `transf.ztor.mode`.*

``transf.ztor.mode`` — mode of tanh(theta).

## `transf_ztor2`

```python
transf_ztor2(xi)
```

*Mirrors R's `transf.ztor2`.*

``transf.ztor2`` — z-to-r, squared (coefficient of determination).
