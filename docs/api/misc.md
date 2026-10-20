# Other

*2 public names, generated from `pimeta.__all__` and the docstrings in `src/pimeta/`. Do not edit this page by hand — run `python tools/gen_api_docs.py`.*

| Name | Mirrors in R |
|---|---|
| [`pairwise`](#pairwise) | — |
| [`PairwiseData`](#pairwisedata) | — |

## `pairwise`

```python
pairwise(treat=None, n=None, event=None, mean=None, sd=None, time=None, studlab=None, data=None, sm: 'str' = '', incr: 'float' = 0.5, allincr: 'bool' = False, addincr: 'bool' = False, allstudies: 'bool' = False) -> "'PairwiseData'"
```

Convert arm-level data to comparison-level data (netmeta::pairwise).

Supports binary (``event`` / ``n``; sm OR/RR/RD) and continuous
(``mean`` / ``sd`` / ``n``; sm MD/SMD) data. Each study with ``a`` arms
contributes ``a(a-1)/2`` pairwise comparisons (treat1 vs treat2).

``data`` may be a dict / ``DataFrame`` whose columns carry the same names
as the arguments (``treat``, ``n``, ``event``, ``mean``, ``sd``,
``studlab``); an explicitly passed argument wins over the column.

Not supported: R's ``TE``/``seTE`` arguments, which feed contrast-level data
straight through instead of deriving it from the arms.

## `PairwiseData`

```python
PairwiseData(sm: 'str', TE: 'list', seTE: 'list', treat1: 'list', treat2: 'list', studlab: 'list', narms: 'list') -> None
```

Comparison-level data produced by :func:`pairwise`.
