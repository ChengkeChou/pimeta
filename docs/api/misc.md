# Other

*9 public names, generated from `pimeta.__all__` and the docstrings in `src/pimeta/`. Do not edit this page by hand — run `python tools/gen_api_docs.py`.*

| Name | Mirrors in R |
|---|---|
| [`describe`](#describe) | — |
| [`list_models`](#list_models) | — |
| [`list_estimators`](#list_estimators) | — |
| [`list_effect_sizes`](#list_effect_sizes) | — |
| [`list_diagnostics`](#list_diagnostics) | — |
| [`diagnose`](#diagnose) | — |
| [`Diagnostic`](#diagnostic) | — |
| [`Diagnosis`](#diagnosis) | — |
| [`STRATEGIES`](#strategies) | — |

## `describe`

```python
describe(name: 'str') -> 'dict[str, Any]'
```

Everything this repository can say about one entry point.

Returns a dict with ``kind``, ``entry_point``, ``summary``,
``assumptions``, ``methods``, ``default``, ``validation_status``,
``limitations``, ``references`` and ``evidence``.  The status is one of
``validated``, ``conditional``, ``parity`` or ``unvalidated``; see the
module docstring for what each means, and ``evidence`` for the file that
backs it.

## `list_models`

```python
list_models() -> 'list[str]'
```

Names accepted by `describe`.

## `list_estimators`

```python
list_estimators() -> 'dict[str, str]'
```

The tau^2 estimators, by the name accepted in ``method=``.

## `list_effect_sizes`

```python
list_effect_sizes() -> 'dict[str, str]'
```

The effect-size families, and the measures in each.

## `list_diagnostics`

```python
list_diagnostics() -> 'dict[str, str]'
```

The diagnostics a fit can report, and what each answers.

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
