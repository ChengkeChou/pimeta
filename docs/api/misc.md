# Other

*4 public names, generated from `pimeta.__all__` and the docstrings in `src/pimeta/`. Do not edit this page by hand — run `python tools/gen_api_docs.py`.*

| Name | Mirrors in R |
|---|---|
| [`diagnose`](#diagnose) | — |
| [`Diagnostic`](#diagnostic) | — |
| [`Diagnosis`](#diagnosis) | — |
| [`STRATEGIES`](#strategies) | — |

## `diagnose`

```python
diagnose(fit) -> 'Diagnosis'
```

Run every applicable check against a fitted object.

**Args**

- `fit` — any fitted result. Checks that do not apply — because the object does not carry the state they read — are skipped rather than guessed at.

**Returns**

`Diagnosis`, worst severity first, empty when nothing fired.

**Example**

>>> fit = ma.rma(yi, vi)
- `>>> for d in ma.diagnose(fit)` — # doctest: +SKIP
...     print(d)

## `Diagnostic`

```python
Diagnostic(code: 'str', category: 'str', severity: 'str', message: 'str', action: 'str') -> None
```

One thing worth knowing about a fit.

**Attributes**

- `code` — a stable machine-readable identifier, e.g. ``tau2_zero``. Codes are the contract; messages may be reworded.
- `category` — one of ``identifiability``, ``optimisation``, ``boundary``, ``data`` or ``inference``.
- `severity` — ``error``, ``warning`` or ``info``.
- `message` — what was found, with the numbers that show it.
- `action` — what a user should do about it, or why there is nothing to do.

## `Diagnosis`

```python
Diagnosis(items: 'list[Diagnostic]' = <factory>) -> None
```

The diagnostics for one fit, with the worst severity first.

## `STRATEGIES`

*Mapping (`dict`) with 3 keys:* `complete_case`, `available_case`, `likelihood`

## Discovery

Five functions answer "what can this package do, and how far is each part
trusted" without reading the source.

```python
import pimeta as ma

ma.list_models()          # names accepted by describe()
ma.list_estimators()      # tau^2 estimators, by the name used in method=
ma.list_effect_sizes()    # effect-size families and the measures in each
ma.list_diagnostics()     # the diagnostics a fit can report
ma.describe("rma.mv")     # assumptions, methods, default, limitations, evidence
```

`describe()` returns `kind`, `entry_point`, `summary`, `assumptions`,
`methods`, `default`, `validation_status`, `limitations`, `references` and
`evidence`.  The status is one of:

| status | meaning |
|---|---|
| `validated` | the quantity was measured here and the claim held |
| `conditional` | measured, and it holds within the stated conditions |
| `measured-shortfall` | measured, and it does **not** meet the nominal claim; the numbers are quoted |
| `parity` | agrees with a reference implementation; says nothing about calibration |
| `unvalidated` | no measurement exists in this repository |

`evidence` names the file that backs the claim, and
`tests/test_catalog.py` asserts that the file exists and that the status is in
the vocabulary -- a catalogue of unbacked claims is worse than none.

::: pimeta.catalog
