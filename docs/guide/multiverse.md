# Multiverse analysis

A multiverse runs every analysis that was defensible *before* the results were
seen and asks how far the answer moves across them. It is not a search for the
result you like, and it is not model averaging: averaging over a multiverse would
be averaging over analyses whose weights nobody defined.

Prerequisites: the decisions have to be declared first — see
[Reproducible analyses](reproducibility.md#declaring-the-decisions-analysisspec).

```python
import pimeta as ma

spec = ma.AnalysisSpec(decisions=(
    ma.Decision(name="method", values=("REML", "DL", "FE"), default="REML",
                reasons={"REML": "the default here; less biased at small k",
                         "DL": "what most published analyses used",
                         "FE": "the assumption of no heterogeneity"}),
    ma.Decision(name="test", values=("z", "knha"), default="z",
                reasons={"z": "the nominal reference",
                         "knha": "wider when the studies disagree"}),
))

def run(choices):
    return ma.rma(yi, vi=vi, method=choices["method"], test=choices["test"])

mv = ma.multiverse(spec, run)
print(mv)
```

```text
Multiverse over 6 specification(s): 6 usable, 0 failed
  estimate range -0.0412 to +0.0518 (median +0.0104)
  sign agreement 0.67; 2 of 6 intervals exclude zero
  test: 0.71 of the spread
  method: 0.29 of the spread
```

## The five quantities, and what each one is not

| Quantity | Reading |
|---|---|
| counts: considered, run, usable, failed | the only numbers that are not a statistic |
| sign agreement | how often the analyses agree on the direction |
| intervals excluding zero | an **unweighted frequency over analyses** |
| estimate range and quantiles | how far the answer travels |
| variance by decision axis | where the spread sits, descriptively |

**The share of intervals that excludes zero is not a probability that the effect
is real.** `M` analyses are not `M` independent experiments: they share the same
studies, so they are not independent, and the grid is not a prior over
hypotheses. A multiverse where 80% of specifications are significant is not
evidence that is 80% likely to be true; it is evidence that survived four of five
sets of choices.

**The variance decomposition is descriptive.** It is a one-way decomposition, so
an interaction between two decisions is left unattributed and the shares sum to
less than one. An unbalanced grid moves the numbers too, and a decision can look
important simply because its levels happen to be paired with another decision's
favourable level. Read it as "where does the spread sit", never as "what caused
it".

## Failures are counted, not dropped

```python
mv.counts()        # {'considered': 6, 'run': 6, 'usable': 5, 'failed': 1}
mv.failed          # [{'id': 'sha256:...', 'choices': {...}, 'error': '...'}]
```

A specification that did not converge, or whose result could not be reduced to an
estimate, stays in the output with its id, its choices and its error. Dropping
those rows is the largest silent bias in a published multiverse: the analyses
that break are not a random subset of the analyses that work, and a table that
lists only the survivors reports a stability it never measured.

The same applies to a result whose estimate is not finite — it is a failure with
the reason attached, not a `NaN` quietly entering an average.

## Reading the curve

```python
for row in mv.curve():          # ordered by estimate
    print(f"{row['estimate']:+.4f}  {row['ci_lb']:+.4f} {row['ci_ub']:+.4f}"
          f"  method={row['method']} test={row['test']}")
```

Ordered by estimate rather than by grid position, because the question is where
the answer sits, not in what order the loop visited the cells. Each row keeps the
choices that produced it, so a cluster of similar estimates can be traced to the
decision that defines it.

### The figure

```python
fig = ma.plot_specification_curve(
    mv, mark_default=mv.curve()[0]["id"], filename="curve.png")
```

The upper panel is the curve: one interval per usable specification, ordered by
its estimate, with the median marked. The panels below are the decision margins —
for each decision, which value each specification took. Reading down a column
says "these analyses agreed on this decision"; reading across a row says "this
decision changed here".

The margins are the point of the figure. The curve alone invites the reader to
attribute the movement to whichever decision they already suspected; the values
underneath make a *cluster* visible instead, and a cluster is what an interaction
looks like. The decomposition's blind spot and the figure's useful feature are
the same thing seen from two sides.

`mark_default` puts the primary analysis inside its own multiverse. A figure of
this kind should always be able to say which of the specifications it draws was
the one pre-specified.

## Choosing an estimand

`multiverse` needs `(estimate, ci_lb, ci_ub)` per result. The default reads the
first coefficient, which is the pooled effect for an intercept-only model and the
intercept for a meta-regression — rarely what you want there. Pass `extract=`:

```python
def slope(result):
    return result.b[1], result.ci_lb[1], result.ci_ub[1]

mv = ma.multiverse(spec, run, extract=slope)
```

## See also

- [Sensitivity analyses](sensitivity.md) — one assumption at a time, no grid
- [Reproducible analyses](reproducibility.md) — declaring the decisions and recording the runs
- [Roadmap](roadmap.md) — what this deliberately does not do
