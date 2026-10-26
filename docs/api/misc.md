# Other

*3 public names, generated from `pimeta.__all__` and the docstrings in `src/pimeta/`. Do not edit this page by hand — run `python tools/gen_api_docs.py`.*

| Name | Mirrors in R |
|---|---|
| [`diagnose`](#diagnose) | — |
| [`Diagnostic`](#diagnostic) | — |
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

## `STRATEGIES`

*Mapping (`dict`) with 3 keys:* `complete_case`, `available_case`, `likelihood`
