# Dynamic heterogeneity: is there more of it now than there used to be?

This page is the development record for `ma.tau_trend` and
`ma.bayes_tau_trend`: what the problem is, which estimator belongs to which
sample size, what was implemented, what was measured, and what is still not
possible. It is written to be read before the API reference, because the
arguments in the reference only make sense against these decisions.

## The problem is not "add a covariate"

A meta-analysis reports one :math:`\hat\tau^2`. The clinical question behind
"has the evidence become more consistent?" is about :math:`\tau^2(t)` — the
between-study variance as a function of time — and moving from one number to a
function changes three things at once:

1. **Identification.** :math:`\tau^2` is a variance component with a boundary
   at zero. A *function* of it has a shape as well as a level, and the shape is
   estimated from the same handful of studies that determine the level.
2. **The reference distribution.** A test of "the shape does not change with
   time" is not the same test as "there is no heterogeneity", and the two have
   different null distributions (see [Where the boundary is](#where-the-boundary-is)).
3. **What has to be reported.** A global :math:`I^2` stops meaning anything
   when :math:`\tau^2` moves; the summary a reader can use is a curve with a
   prediction band, not a point estimate with a footnote.

## Which estimator, at which `k`

There is no single best method here; there is a method that is defensible at
each sample size, and using the wrong one is a silent failure rather than a
loud one.

| studies | what to fit | why |
|---|---|---|
| `k < 15` | two periods, `tau="linear"` on a 0/1 indicator | a continuous :math:`\tau^2(t)` has almost no power; `exp(alpha[1])` is then the ratio of the two periods' variances, which is the honest question |
| `15 <= k <= 45` | `ma.bayes_tau_trend` | the shrinkage prior on the slope is what keeps three influential early studies from deciding the shape; the frequentist profile likelihood is unbiased but very noisy here |
| `25 <= k <= 50`, non-monotone | `tau="spline"` (see [The spline scale](#the-spline-scale)) | a bump in the middle of the time range has no linear or quadratic trend to speak of, and a spline is the smallest model that can follow one |
| `k > 45` | `ma.tau_trend` | the profile likelihood is well behaved, and the fit takes milliseconds instead of minutes |

The two-period route needs no new code: pass a 0/1 column as `time` with
`tau="linear"` and read `exp(alpha[1])`.

## The model

$$
y_i \sim N\big(x_i'\beta,\ v_i + \tau_i^2\big), \qquad
\log \tau_i^2 = z_i'\alpha .
$$

The log link is not decoration: without it the optimiser has to be constrained
to keep :math:`\tau_i^2 \ge 0`, and a constrained optimum lands on the boundary
where the asymptotics are worst. With it, the parameter space is
:math:`\mathbb{R}^q` and no constraint is ever active.

Time is **centred** on its mean before either design is built, so
:math:`\beta_0` is the pooled effect at the average time and
:math:`\alpha_1` is per unit of the original scale. This matters for
interpretation: on a year scale :math:`\alpha_1 = 0.09` is a 9 % increase in
:math:`\tau^2` per year, which compounds to :math:`e^{0.09 \times 35} = 23`
over a 35-year literature.

`mu` and `tau` each take `"none"`, `"linear"` or `"quadratic"`, which gives
nine models and one of them — `mu="none"` with a non-constant `tau` — warns,
for the reason in the next section.

## Rule 1: the mean model is not optional

If the pooled effect rises with time and the heterogeneity does not, the early
and late studies sit far from one common mean. Their squared residuals grow,
and a scale model with nothing else to explain them reports growth as
heterogeneity.

Measured, on a 60-study simulation with a **constant** :math:`\tau^2 = 0.15`
and a mean effect that climbs by 1.5 units across the time range:

| fitted mean | :math:`\hat\tau^2` at the mean time | at the low end | at the high end |
|---|---|---|---|
| `mu="none"` | 0.28 | **0.50** | 0.16 |
| `mu="linear"` | 0.12 | 0.15 | 0.10 |

The wrong model does not merely add noise: it reports twice the heterogeneity at
one end of the literature and half at the other, with the sign of the apparent
trend determined by the shape of the mean trend rather than by any change in
consistency. `tests/test_dynhet.py::test_omitting_the_mean_trend_inflates_tau2`
pins these numbers.

## How the fit is computed

`metafor` optimises over every parameter with `nlminb`. `pimeta` profiles the
location coefficients out: for a given :math:`\alpha` the weights
:math:`w_i = (v_i + \tau_i^2)^{-1}` are known, so

$$
\hat\beta(\alpha) = (X'WX)^{-1}X'Wy, \qquad
RSS(\alpha) = \sum_i w_i (y_i - x_i'\hat\beta)^2
$$

and the profile likelihood is an explicit function of :math:`\alpha` alone.
The analytic gradient is used, because a wrong gradient still converges — just
to the wrong place — so it is checked against central differences in the tests.
Two consequences worth knowing:

- the optimiser only sees two or three parameters, so the fit is fast and the
  surface can be polished with a derivative-free pass as well;
- the observed information is computed numerically for the **joint**
  :math:`(\beta, \alpha)` likelihood, whose :math:`\alpha` block of the inverse
  is the inverse of the profile information (Schur complement). That is what
  reproduces `metafor`'s `va`.

## Rule 2: Knapp–Hartung, and what it actually fixes

`metafor`'s location variance in a location-scale model is
:math:`(X'WX)^{-1}` evaluated at :math:`\hat\tau^2` — the estimated scale is
treated as known. The obvious suspicion is that this makes the interval too
narrow, and the obvious fix is to propagate the scale uncertainty through the
joint information.

That fix turns out to matter much less than the suspicion suggests, and the
reason is worth knowing: in a Gaussian model the **expected** information is
block-diagonal between location and scale, so the two blocks are orthogonal to
first order and the difference is :math:`O(1/k)`. Measured on the 36-study
fixture: the observed cross-information is 4 % of the diagonal and the
profile-based location SE is **0.25 %** larger than the GLS one. Both are on
the result (`se_beta` and `se_beta_profile`), and `se="profile"` selects which
one the intervals use.

What does matter is the reference distribution, and here the measurement is
less tidy than the advice:

| interval | coverage, 120 replications, `k = 40`, nominal 95 % |
|---|---|
| :math:`\beta_1`, z | 0.917 |
| :math:`\beta_1`, Knapp–Hartung | 0.925 |
| :math:`\alpha_1`, z | 0.950 |
| :math:`\alpha_1`, t on `k - q` | 0.967 |

The Monte Carlo standard error of a coverage rate at 120 replications is 0.025,
so the two location figures are **not** distinguishable from each other. What
the study establishes:

- the **scale** interval is calibrated, with the z reference at nominal and the
t reference slightly conservative;
- the **location** interval is about three points short of nominal and
Knapp–Hartung does not repair it — the shortfall is not in the reference
  distribution, and it is not the scale uncertainty either (0.25 %);
- it does **not** establish which reference is better at `k = 40`. The
  recommendation to use `test="knha"` stands on its behaviour at smaller `k` in
  the literature, and this study is not precise enough to add to that evidence.

The rule itself is `metafor`'s, and it is easy to get wrong in a location-scale
model because the two parts are treated differently:

- the **location** variance is scaled by :math:`s^2_w = RSS/(k-p)` and its tests
  use a :math:`t_{k-p}` reference;
- the **scale** variance is *not* rescaled, and its tests use
  :math:`t_{k-q}`;
- the omnibus test of the scale coefficients becomes an
  :math:`F(q-1, k-q)`.

The tests pin the scale factor to `1e-6` and the scale standard errors to
`1e-6` *unchanged* from the z-based fit, which is where a plausible-looking
implementation goes wrong.

## Where the boundary is

The test of :math:`\alpha_1 = 0` is often described as a boundary test needing a
mixture chi-square. That is half right, and the half matters:

- when heterogeneity is **present** and only its *shape* is in question,
  :math:`\alpha_1 = 0` is an interior point — heterogeneity may shrink as well
  as grow — so the likelihood-ratio statistic is asymptotically
  :math:`\chi^2_1`, and neither a mixture nor a bootstrap is required;
- when the truth is **no heterogeneity at all**, :math:`\alpha_0 \to -\infty`,
  :math:`\alpha_1` stops being identified, and the :math:`\chi^2_1` reference is
  wrong in the usual boundary way.

`test_trend` therefore offers three references and says which null each assumes:

| method | statistic | reference | valid when |
|---|---|---|---|
| `"wald"` | Wald on the scale slopes | :math:`\chi^2_{q-1}` or :math:`F(q-1, k-q)` | heterogeneity present |
| `"lrt"` | likelihood ratio vs `tau="none"` | :math:`\chi^2_{q-1}` | heterogeneity present |
| `"bootstrap"` | the same LRT, re-simulated under the fitted null | parametric bootstrap | both |

The bootstrap costs `n_boot` pairs of refits and is the only one of the three
whose level survives the no-heterogeneity null. When the scale coefficients are
not identified at all — which happens when :math:`\hat\tau^2` is driven to zero
— the Wald statistic is refused with an error rather than reported as a number.

## Rule 3: the time-varying prediction interval

A global :math:`I^2` has no meaning once :math:`\tau^2` moves. What a reader can
use is

$$
\hat\mu(t) \pm t_{df}\sqrt{\widehat{\mathrm{Var}}\{\hat\mu(t)\} + \hat\tau^2(t)},
$$

which is `predict(t)`. It is reported as a band over time, and its width is the
thing to look at: a narrowing band is the evidence-base converging, a widening
one is not, and neither statement is available from a single number.

That interval does **not** include the uncertainty of :math:`\hat\tau^2(t)`; no
frequentist interval of this form does. `ma.bayes_tau_trend` does, because the
prediction comes from the draws, and the difference is visible: on the fixture
the Bayesian prediction interval at the last year is 19 % wider than the
frequentist one.

## The spline scale

`tau="spline"` puts a restricted cubic spline on :math:`\log\tau^2(t)`, with
`n_knots = 4` by default. Two decisions make it estimable at `k = 60` rather
than merely fittable:

1. **The bending columns are orthogonalised.** The spline contributes
   `nk - 2 = 2` non-linear columns. They are projected off :math:`[1, t]` and
   Gram–Schmidt'ed, so the basis satisfies :math:`B'B = I` and
   :math:`B'[1, t] = 0`. The first property turns a single wiggliness
   hyperparameter into one ridge penalty that shrinks every direction equally;
   the second makes the level and the linear trend keep their own coefficients,
   so `alpha[0]` and `alpha[1]` mean the same thing as in the linear model.
2. **The wiggliness is a variance component, not a free parameter.**
   :math:`\gamma \sim N(0, \sigma_s^2 I)`, and :math:`\sigma_s` is estimated by
   the Laplace approximation to the marginal likelihood — the penalised-spline
   estimator written as the mixed model it is (Eilers & Marx 1996; Ruppert,
   Wand & Carroll 2003). Profiling the *penalised* objective over
   :math:`\sigma_s` would not work, and it is worth saying why: the penalty is
   smallest at :math:`\gamma = 0`, so that objective improves monotonically as
   :math:`\sigma_s \to 0` and the fit would always report the linear model. The
   two terms that make the comparison honest are the prior normalisation
   :math:`-\tfrac{m}{2}\log\sigma_s^2` and the Laplace volume
   :math:`\tfrac12\log\det(J + \lambda I)`.

The effective degrees of freedom the bending block spends is
:math:`\mathrm{edf} = \operatorname{tr}\{(J + \lambda I)^{-1} J\}`, and it is
folded back into the scale reference: `df_alpha = k - 2 - round(edf)` for
Knapp–Hartung, and the likelihood-ratio test against a constant scale is
referred to :math:`\chi^2` with :math:`1 + \mathrm{edf}` degrees of freedom.
There is deliberately **no Wald test** for a spline: a penalised fit has no
fixed number of scale coefficients to form one from, and the method says so
instead of reporting a statistic with an unknown reference distribution.

### What it does on the two fixtures

`dynhet.csv` (linear truth) — the spline must collapse: it returns the linear
coefficients to 1e-4, :math:`\sigma_s = 0.001` at the lower bound of the search,
`edf = 0` and a log-likelihood identical to the linear fit.

`dynhet_spline.csv` (a bump: :math:`\tau^2` 0.050 at the ends, 0.246 in the
middle) — the linear model sees nothing (loglik −11.58), the spline finds it
(−8.29) with :math:`\sigma_s = 4.12` and `edf = 1.53`. The likelihood-ratio
test against a constant scale gives 6.73 on 2.53 degrees of freedom,
p = 0.057: a borderline answer, which is what one draw of 60 studies *should*
give for a bump this shallow.

### The Bayesian check, and what it agrees about

`tools/r_reference/tau2t_spline.R` fits the same model in Stan (CmdStan 2.40.0
through cmdstanr), on the same basis, with a half-normal(5) prior on
:math:`\sigma_s` so that the comparison is of estimators rather than of priors:

| quantity | penalised REML | Stan posterior mean |
|---|---|---|
| `alpha[0]` | −3.0632 | −3.0613 (sd 0.365) |
| `alpha[1]` | −0.00527 | −0.00613 (sd 0.043) |
| `sigma_s` | 4.12 | 4.60 (sd 2.63) |
| :math:`\tau^2` at 1980 / 1997.5 / 2015 | 0.0250 / 0.0945 / 0.0089 | 0.046 / 0.106 / 0.024 |

The level, the trend and the shape agree; :math:`\sigma_s` agrees only to about
30 %, which is the honest statement — it is the weakly identified direction, and
the posterior says so itself. Both estimates put the peak in the middle, which
is the substantive claim.

## How this was validated

Four layers, in the order that lets each mean something.

**1. `metafor` parity.** `metafor::rma(yi, vi, mods = ~ time, scale = ~ time)`
fits the same model. Four nestings times two estimators, plus `test="knha"`:

| quantity | worst absolute difference over the eight fits |
|---|---|
| location coefficients | 1e-10 |
| location standard errors | 2e-9 |
| scale coefficients | 5e-7 |
| scale standard errors | 8e-9 |
| maximised log-likelihood | 1e-14 |
| Knapp–Hartung scale factor | 6e-8 |

Two nestings are checked without R as well: `mu="none", tau="none"` has to
reproduce `ma.rma`'s REML :math:`\tau^2` (it does, to 1e-5), and the profile
likelihood has to equal the joint one at the optimum.

**2. An exact posterior.** For the Bayesian version the sampler is not compared
with another sampler. Given :math:`\alpha` the model is Gaussian in
:math:`\beta`, so :math:`p(d \mid \alpha)` is closed form and the posterior is a
two-dimensional integral on a grid — `tools/quadrature_dynhet.py`, no MCMC at
all. The sampler lands within Monte Carlo error of it:

| parameter | exact | sampled (4 chains x 4000 draws) | z |
|---|---|---|---|
| :math:`\alpha_0` | -2.556 (0.391) | -2.560 (0.399) | -0.01 |
| :math:`\alpha_1` | 0.096 (0.040) | 0.097 (0.040) | 0.03 |
| :math:`\beta_0` | 0.246 (0.064) | 0.247 (0.064) | 0.01 |
| :math:`\beta_1` | 0.0178 (0.0060) | 0.0178 (0.0060) | 0.00 |

**3. Coverage.** `tools/coverage_dynhet_ci.py` simulates from a known
:math:`\mu(t)` and :math:`\tau^2(t)` and counts how often a nominal 95 %
interval contains the truth — no reference implementation in the loop. The
numbers are in the next section.

**4. Known truth.** The fixture was simulated with
:math:`\tau^2` growing by a factor of 16 across its range; the fitted curve
contains the true mean at both ends of the time range and the fitted ratio
:math:`e^{\hat\alpha_1}` is 1.096 per year against a true 1.083.

## What was measured, and what it cost

The interval table in [Rule 2](#rule-2-knapphartung-and-what-it-actually-fixes)
is the coverage study: 0.917 and 0.925 for the location interval, 0.950 and
0.967 for the scale interval, against a nominal 0.95 at `k = 40`. Two things it
does not establish, and they are as important as what it does:

- it does not resolve the location shortfall's cause. The scale uncertainty is
  0.25 % of the standard error, so it is not that; the remaining candidates
  (estimated weights making the GLS variance too small, and the profile
  likelihood's own bias in :math:`\alpha`) were not separated;
- it does not establish the **level of the trend test**. The rejection rates at
  5 % over 30–40 replications were between 0.00 and 0.03 for every method under
  both nulls, which is what a calibrated test looks like at that sample size but
  is not evidence of calibration.

Three findings that changed the implementation:

- **`alpha0_sd = 1` was too narrow.** On the fixture it pulled
  :math:`\alpha_0` from the frequentist -2.61 to -2.31, a 35 % change in
  :math:`\tau^2`, because the data supply only :math:`\mathrm{se} = 0.38`.
  The default is now 2, and the docstring records why.
- **A small :math:`\tau^2` sends the log-linear scale to infinity.** On a
  simulation with a constant :math:`\tau^2 = 0.02` the fit returned
  :math:`\alpha_1 = 10.2` — a 27 000-fold change over the range — and
  `metafor` returns the same value, so it is the model and not the
  implementation. A fit whose :math:`\tau^2(t)` spans more than two orders of
  magnitude now warns; the estimate is not "corrected", because there is nothing
  to correct it to.
- **Omitting the mean trend is worse than it looks.** It does not add a trend
  to :math:`\tau^2(t)` on a balanced design — a linear mean misfit is symmetric
  about the mean time, so the *slope* stays near zero — it inflates the
  *level* and both tails, which is the shape distortion in the table above.

## Limits

- **The spline scale is a shape detector, not a variance curve.** On the bump
  fixture it finds the peak and the troughs, but the level at the peak is
  estimated from two bending coefficients and carries the width the posterior
  shows (0.106 with a 90 % interval of 0.051–0.194). Do not read a single
  :math:`\hat\tau^2(t)` off the curve without its interval.
- **No credible region for the curve.** `predict` returns pointwise intervals;
  the simultaneous band across all :math:`t` is wider and is not computed.
- **The Bayesian fit is a random walk** (`pimeta.bayes`), not NUTS. That is
  affordable only because the scale model has two or three parameters; the
  location coefficients are integrated out, not sampled.
- **The frequentist prediction interval ignores the uncertainty in
  :math:`\hat\tau^2(t)`**, as the formula above shows. The Bayesian one does not.
- **`se="profile"` is not a small-sample correction.** It differs from the GLS
  variance by 0.25 % on the fixture. Knapp–Hartung is the correction; this is a
  second-order refinement that happens to be free.
