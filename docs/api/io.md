# Input, output and interoperability

*18 public names, generated from `pimeta.__all__` and the docstrings in `src/pimeta/`. Do not edit this page by hand — run `python tools/gen_api_docs.py`.*

| Name | Mirrors in R |
|---|---|
| [`read_rm5`](#read_rm5) | read.rm5 |
| [`read_cdir`](#read_cdir) | read.cdir |
| [`read_mtv`](#read_mtv) | read.mtv |
| [`ReviewDataResult`](#reviewdataresult) | — |
| [`REVIEW_COLUMNS`](#review_columns) | — |
| [`interop`](#interop) | — |
| [`as_columns`](#as_columns) | — |
| [`column_names`](#column_names) | — |
| [`lookup`](#lookup) | — |
| [`resolve`](#resolve) | — |
| [`resolve_mods`](#resolve_mods) | — |
| [`FormulaSpec`](#formulaspec) | — |
| [`parse_formula`](#parse_formula) | — |
| [`design_matrix`](#design_matrix) | — |
| [`model_frame`](#model_frame) | — |
| [`to_dataframe`](#to_dataframe) | — |
| [`to_html`](#to_html) | — |
| [`install_repr_html`](#install_repr_html) | — |

## `read_rm5`

```python
read_rm5(file, encoding='utf-8')
```

*Mirrors R's `read.rm5`.*

``meta::read.rm5`` — read a RevMan 5 ``.rm5`` (XML) review file.

RevMan 5 stores the review as XML (usually gzip-compressed)::

    <COCHRANE_REVIEW>
      <DATA_AND_ANALYSES>
        <COMPARISON>

**Args**

- `file` — path to the ``.rm5`` file, or an open binary file object.
- `encoding` — text encoding of the embedded XML.

**Returns**

`ReviewDataResult`.

## `read_cdir`

```python
read_cdir(file, encoding='utf-8', sep=None)
```

*Mirrors R's `read.cdir`.*

``meta::read.cdir`` — read a Cochrane ``.cdir`` review file.

``.cdir`` is a delimited text file.  Sections are introduced by lines such
as ``@comparison``, ``@outcome`` and ``@study``; the remaining lines are
tabular with a header row whose column names follow the RevMan naming
(``Event1``, ``Total1``, ``Event2``, ``Total2``, ``Mean1``, ``SD1``,
``Mean2``, ``SD2``, ``ES``, ``SE``, ``OE``, ``V``).

**Args**

- `file` — path or open text/binary file object.
- `encoding` — text encoding.
- `sep` — field separator (default: auto-detect tab / comma / semicolon).

**Returns**

`ReviewDataResult`.

## `read_mtv`

```python
read_mtv(file, encoding='utf-8')
```

*Mirrors R's `read.mtv`.*

``meta::read.mtv`` — read a ``.mtv`` (MetaView) data file.

The MetaView format is a whitespace/comma separated table with the columns
``id, comp, outcome, treat, n, event`` (binary) or
``id, comp, outcome, treat, n, mean, sd`` (continuous).  Each study
contributes one row per arm; the arms are paired within a study.

**Args**

- `file` — path or open file object.
- `encoding` — text encoding.

**Returns**

`ReviewDataResult`.

## `ReviewDataResult`

```python
ReviewDataResult(data, title='', label_e='', label_c='', comparisons=None, source='')
```

Result of `read_rm5` / `read_cdir` / `read_mtv`.

**Attributes**

- `data` — list of dicts with the columns in `REVIEW_COLUMNS`.
- `title` — review title.
- `label_e, label_c` — names of the experimental / control groups.
- `comparisons` — list of comparison labels.
- `source` — the file that was read.

## `REVIEW_COLUMNS`

*Constant (`list`):* `O.E`, `TE`, `V`, `comp.no`, `complab`, `event.c`, `event.e`, `group.no`, `label.c`, `label.e`, `mean.c`, `mean.e`, `method`, `n.c`, `n.e`, `outclab`, `outcome.no`, `overall`, `sd.c`, `sd.e`, `seTE`, `sm`, `studlab`, `type`

## `interop`

*Subpackage `pimeta.interop` — see the [interoperability guide](../guide/interop.md).*

## `as_columns`

```python
as_columns(data) -> 'dict'
```

Normalise any supported frame into ``{column name: ndarray}``.

**Args**

- `data` — ``dict``, pandas ``DataFrame``, polars ``DataFrame``, NumPy structured array, sequence of record dicts, or an object whose ``__dict__`` holds the columns.

**Returns**

A plain ``dict`` mapping column names to one-dimensional arrays.

**Raises**

- `TypeError` — when ``data`` is none of the supported types.

## `column_names`

```python
column_names(data) -> 'list'
```

The column names of any frame accepted by `as_columns`.

## `lookup`

```python
lookup(data, name: 'str') -> 'np.ndarray'
```

Fetch column ``name`` from ``data``, with a helpful error when missing.

## `resolve`

```python
resolve(spec, data=None, k=None, name: 'str' = 'value', dtype=<class 'float'>, strings_as_names=None)
```

Resolve one argument that may be a value, a column name or a name list.

**Args**

- `spec` — the value itself, a column name (requires ``data``), or a list of column names (returns a list of arrays).
- `data` — the ``data=`` frame.
- `k` — expected length; checked when given.
- `name` — argument name, used in error messages.
- `dtype` — target NumPy dtype.  ``None`` keeps the input dtype (needed for string arguments such as study labels or treatment names).
- `strings_as_names` — whether a list of strings should be read as column names.  ``None`` (default) decides automatically: only when ``data`` was supplied *and* every string is an existing column.  That keeps legitimate string vectors such as ``slab=["Study 1", "Study 2"]`` working, which a blanket "list of strings means column names" rule would break.

**Returns**

``None``, an ``ndarray``, or a ``list`` of ``ndarray``.

**Raises**

- `ValueError` — when a column name is used without ``data``, a column is missing, or the resolved length does not match ``k``.

## `resolve_mods`

```python
resolve_mods(mods, data=None, k=None)
```

Resolve the ``mods`` argument, accepting formulas and column names.

``rma()`` prepends its own intercept column, so the intercept encoded in a
formula is reported separately through the second return value instead of
being materialised here.

**Args**

- `mods` — ``None``; a 1-D/2-D array-like; a list of column names; or a formula string such as ``"~ dose + year"``.
- `data` — the ``data=`` frame.
- `k` — number of studies, used for length checking.

**Returns**

``(array_or_None, add_intercept)``.

**Raises**

- `ValueError` — when a formula has a left-hand side, or a column is missing.

## `FormulaSpec`

```python
FormulaSpec(response=None, terms=None, intercept=True)
```

A parsed formula: ``response ~ term1 + term2``.

**Attributes**

- `response` — name of the left-hand-side column, or ``None``.
- `terms` — list of term strings; an interaction is written ``"a:b"``.
- `intercept` — whether the model includes an intercept column.

## `parse_formula`

```python
parse_formula(formula) -> 'FormulaSpec'
```

Parse a formula string into a `FormulaSpec`.

Accepts ``"y ~ a + b"``, ``"~ a + b"``, ``"a + b"``, ``"~ a*b"``,
``"~ a:b"``, ``"~ 0 + a"``, ``"~ a - 1"`` and ``"~ ."``.

**Args**

- `formula` — the formula string (a `FormulaSpec` is returned as-is).

**Returns**

The parsed `FormulaSpec`.

**Raises**

- `ValueError` — for an empty or malformed formula.

## `design_matrix`

```python
design_matrix(formula, data, intercept=None)
```

Build a design matrix from a formula and a frame.

**Args**

- `formula` — formula string or `FormulaSpec`.
- `data` — any frame accepted by `as_columns`.
- `intercept` — force the intercept on/off; ``None`` uses the formula's own ``0`` / ``1`` marker.

**Returns**

``(X, names)`` — the ``k x p`` design matrix (``p == 0`` when the model has neither an intercept nor terms) and its column names.

**Raises**

- `ValueError` — when a term names a missing column, or a ``"."`` expansion leaves nothing.

## `model_frame`

```python
model_frame(formula, data)
```

Split a two-sided formula into ``(y, X, names)``.

**Args**

- `formula` — formula string such as ``"yi ~ dose + year"``.
- `data` — any frame accepted by `as_columns`.

**Returns**

``(y, X, names)`` where ``y`` is ``None`` for a one-sided formula.

**Raises**

- `ValueError` — when the formula has no terms, or a column is missing.

## `to_dataframe`

```python
to_dataframe(obj, rows=None)
```

Turn a result object into a pandas ``DataFrame`` (or a dict of columns).

Scalar fields become one row; when the object carries per-row vectors (a
``leave1out`` / ``influence`` / ``cumul`` style result) every row-level
field becomes a column instead.

**Args**

- `obj` — any pimeta result object.
- `rows` — optional row selection for list-like results.

**Returns**

A ``pandas.DataFrame`` when pandas is importable, otherwise a plain ``dict`` of columns (so the function never hard-requires pandas).

## `to_html`

```python
to_html(obj, max_rows: 'int' = 60) -> 'str'
```

Render a result object as a compact HTML table (Jupyter ``_repr_html_``).

**Args**

- `obj` — any pimeta result object.
- `max_rows` — maximum number of fields shown before truncating.

**Returns**

An HTML ``<table>`` string.

## `install_repr_html`

```python
install_repr_html(namespace) -> 'int'
```

Attach ``_repr_html_`` to every public class in ``namespace`` with ``to_dict``.

Called once from `__init__` so that ``repr()`` in Jupyter shows
a formatted table for every result type, without repeating the mixin in 40
class definitions.

**Args**

- `namespace` — a module ``globals()`` mapping.

**Returns**

The number of classes that received the method.
