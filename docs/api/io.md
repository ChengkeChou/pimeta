# Input, output and interoperability

*37 public names, generated from `pimeta.__all__` and the docstrings in `src/pimeta/`. Do not edit this page by hand — run `python tools/gen_api_docs.py`.*

| Name | Mirrors in R |
|---|---|
| [`read_rm5`](#read_rm5) | read.rm5 |
| [`read_cdir`](#read_cdir) | read.cdir |
| [`read_mtv`](#read_mtv) | read.mtv |
| [`ReviewDataResult`](#reviewdataresult) | — |
| [`REVIEW_COLUMNS`](#review_columns) | — |
| [`interop`](#interop) | — |
| [`record`](#record) | — |
| [`RECORD_SCHEMA`](#record_schema) | — |
| [`STORE_SCHEMA`](#store_schema) | — |
| [`AnalysisRecord`](#analysisrecord) | — |
| [`build_record`](#build_record) | — |
| [`save_record`](#save_record) | — |
| [`load_record`](#load_record) | — |
| [`compare_results`](#compare_results) | — |
| [`environment`](#environment) | — |
| [`RecordError`](#recorderror) | — |
| [`RecordMismatch`](#recordmismatch) | — |
| [`fingerprint`](#fingerprint) | — |
| [`source_fingerprint`](#source_fingerprint) | — |
| [`analysis_fingerprint`](#analysis_fingerprint) | — |
| [`EstimatesStore`](#estimatesstore) | — |
| [`StoredModel`](#storedmodel) | — |
| [`estimates_store`](#estimates_store) | — |
| [`read_table`](#read_table) | — |
| [`write_table`](#write_table) | — |
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

## `record`

*Subpackage `pimeta.record` — see the [interoperability guide](../guide/interop.md).*

## `RECORD_SCHEMA`

*Constant of type `str`.*

## `STORE_SCHEMA`

*Constant of type `str`.*

## `AnalysisRecord`

```python
AnalysisRecord(kind: 'str', inputs: 'dict' = <factory>, analysis: 'dict' = <factory>, process: 'dict' = <factory>, environment: 'dict' = <factory>, results: 'dict' = <factory>, schema: 'str' = 'pimeta.record/1', created_utc: 'str' = <factory>) -> None
```

One analysis, fully described.

**Attributes**

- `kind` — short label (``"rma"``, ``"netmeta"``, ...).
- `created_utc` — ISO-8601 timestamp of when the record was built.
- `inputs` — the ``输入数据`` block.
- `analysis` — the ``分析配置`` block.
- `process` — the ``运算过程`` block.
- `environment` — the ``运行环境`` block.
- `results` — the ``结果与诊断`` block.
- `schema` — layout identifier; see `RECORD_SCHEMA`.

## `build_record`

```python
build_record(result: 'Any', *, kind: 'str | None' = None, data: 'Any' = None, mode: 'str' = 'digest', columns: 'Sequence[str] | None' = None, analysis_columns: 'Sequence[str] | None' = None, settings: 'Mapping[str, Any] | None' = None, function: 'str | None' = None, call: 'Mapping[str, Any] | None' = None, pass_data: 'bool | None' = None, process: 'Mapping[str, Any] | None' = None, diagnostics: 'Mapping[str, Any] | None' = None, seed: 'int | None' = None) -> 'AnalysisRecord'
```

Wrap a fitted result in an `AnalysisRecord`.

**Args**

- `result` — anything the package fitted -- an ``RmaResult``, ``NetmetaResult``, ``MetagenResult``, ...
- `kind` — short label; defaults to ``function`` or the class name.
- `data` — the source table.  Supplying it is what makes the record verifiable; without it only the arrays pulled from ``result`` are fingerprinted.
- `mode` — ``"digest"`` (default) or ``"bundle"``.
- `columns` — source columns to fingerprint; defaults to every column of ``data``.
- `analysis_columns` — columns that entered the model; defaults to ``columns``.
- `settings` — values that can move a number.  Defaults to the result's own ``method``/``test``/``level``/... attributes.
- `function` — name of the pimeta function that was called, for `AnalysisRecord.replay`.
- `call` — the keyword arguments that were passed.  Column names are looked up in ``data`` again on replay.
- `pass_data` — force the bundled data to be injected as ``data=`` on replay.  ``None`` (default) infers it: the data is injected only when the recorded call refers to columns by name.  Set it to ``True`` for a function that takes a whole table -- ``reitsma``, for example, whose ``data=`` is the counts table itself rather than a set of column names.
- `process` — extra ``运算过程`` entries (iterations, convergence, timings).
- `diagnostics` — extra ``结果与诊断`` entries.
- `seed` — random seed in force, recorded in the environment block.

**Returns**

The populated record.

**Raises**

- `ValueError` — for an unknown ``mode``.

## `save_record`

```python
save_record(path: 'str | os.PathLike', result: 'Any', **kwargs: 'Any') -> 'AnalysisRecord'
```

Build a record and write it in one call.

``save_record(path, result, data=frame, mode="bundle", function="rma",
call={"yi": "TE", "vi": "seTE2", "method": "REML"})`` saves and returns
the record; ``load_record(path).replay()`` re-runs it.

**Args**

- `path` — destination JSON file.
- `result` — the fitted result.
- `**kwargs` — forwarded to `build_record`.

**Returns**

The record that was written.

## `load_record`

```python
load_record(path: 'str | os.PathLike') -> 'AnalysisRecord'
```

Read a record from disk.

## `compare_results`

```python
compare_results(recorded: 'Mapping[str, Any]', fresh: 'Mapping[str, Any]', *, rtol: 'float' = 1e-09, atol: 'float' = 1e-12) -> 'list[tuple[str, Any, Any]]'
```

Compare two result payloads field by field.

Only numbers are compared -- a changed ``class`` name or a longer
``slab`` is not a reproducibility failure.  NaN is considered equal to
NaN, since "undefined here, undefined there" is a match.

**Args**

- `recorded` — the ``results.values`` block from the record.
- `fresh` — the same block recomputed now.
- `rtol` — relative tolerance.
- `atol` — absolute tolerance.

**Returns**

A list of ``(path, recorded value, fresh value)`` for mismatches, in insertion order.  Empty means the analyses agree.

## `environment`

```python
environment(*, seed: 'int | None' = None) -> 'dict'
```

Describe the software and machine that produced a result.

Versions of everything that can change a floating-point answer are
recorded, because "reproducible" only means something when the reader can
see whether their NumPy differs from yours.  Thread counts matter for the
same reason: BLAS reductions are not always order-stable across thread
counts, so a 16-thread run can differ from a 1-thread run in the last bits.

**Args**

- `seed` — the random seed in force, when the analysis used one.

**Returns**

A JSON-able dict.  Missing optional packages are reported as ``None`` rather than omitted, so the shape of the block is stable.

## `RecordError`

A record cannot be verified or replayed.

Raised for a digest-only record asked to replay without data, a record
with no ``call`` block, or a ``function`` that is not a pimeta function.
The message always says what to do about it.

## `RecordMismatch`

```python
RecordMismatch(message: 'str', differences: 'list[tuple[str, Any, Any]]')
```

Recomputed numbers disagree with the values stored in the record.

## `fingerprint`

```python
fingerprint(payload: 'Any') -> 'str'
```

Digest any JSON-able structure.

**Args**

- `payload` — anything `record.jsonio.canonical` accepts.

**Returns**

- ```"sha256` — "`` followed by `DIGEST_CHARS` hex characters.

## `source_fingerprint`

```python
source_fingerprint(data: 'Any', *, columns: 'Iterable[str] | None' = None) -> 'str'
```

Digest a table exactly as supplied.

Includes column names *in the given order* and every value, so this is the
"same file?" check.  Use it in an audit trail next to the file name.

**Args**

- `data` — anything `as_columns` accepts -- dict, pandas or polars frame, structured array, list of records.
- `columns` — restrict to these columns; ``None`` uses all of them in frame order.

**Returns**

- ```"sha256` — ..."``.

## `analysis_fingerprint`

```python
analysis_fingerprint(data: 'Any' = None, *, columns: 'Iterable[str] | None' = None, settings: 'Mapping[str, Any] | None' = None, arrays: 'Mapping[str, Any] | None' = None) -> 'str'
```

Digest what actually determines the numbers.

Column *order* is ignored (names are sorted) and text columns are only
included when named explicitly, so a relabelled ``slab`` does not change
the result.  Anything in ``settings`` -- estimator, test, confidence
level, transformation -- does.

**Args**

- `data` — table to take the columns from; may be ``None`` when the analysis is fully described by ``arrays``/``settings``.
- `columns` — which columns enter the model.  ``None`` uses all of them.
- `settings` — scalars that affect the computation (``{"method": "REML"}``).
- `arrays` — extra named vectors not present in ``data`` (for example the ``yi``/``vi`` already filtered for missing values).

**Returns**

- ```"sha256` — ..."``.

## `EstimatesStore`

```python
EstimatesStore(name: 'str' = 'pimeta', *, path: 'str | os.PathLike | None' = None)
```

An ordered, named collection of fitted models.

**Args**

- `name` — label written into the saved file; purely descriptive.
- `path` — default file used by `save` and `load`.

## `StoredModel`

```python
StoredModel(key: 'str', record: 'AnalysisRecord', note: 'str | None' = None, live: 'Any' = None) -> None
```

One stored fit: its record plus a few read-only conveniences.

## `estimates_store`

*Constant of type `EstimatesStore`.*

## `read_table`

```python
read_table(path: 'str | os.PathLike', delimiter: 'str | None' = None, *, infer: 'bool' = False) -> 'dict'
```

Read a table into ``{column: list}``.

**Args**

- `path` — file to read.  The extension selects the parser: ``.csv`` (comma), ``.tsv``/``.tab``/``.txt`` (tab), ``.json`` (a list of records or a column dict), ``.jsonl`` (one record per line) or ``.npz``.
- `delimiter` — CSV/TSV delimiter; overrides the extension.
- `infer` — convert numeric cells to ``int``/``float``.  Off by default so the result matches the file byte for byte; the CLI does its own conversion, while a record wants numbers.

**Returns**

A dict of columns.  Blank cells and ``NA``/``NaN`` become ``None``.

**Raises**

- `FileNotFoundError` — when ``path`` does not exist.
- `ValueError` — when the file is not a recognised table.

## `write_table`

```python
write_table(path: 'str | os.PathLike', columns: 'Mapping[str, Sequence] | Any', *, delimiter: 'str | None' = None, float_format: 'str | None' = None) -> 'Path'
```

Write ``columns`` to ``path``, choosing the format from the extension.

**Args**

- `path` — destination; the extension selects the format (``.csv``, ``.tsv``/``.tab``/``.txt``, ``.json``, ``.jsonl``, ``.npz``); parent directories are created.
- `columns` — a mapping of ``{name: values}``, or any frame accepted by `as_columns`.
- `delimiter` — text delimiter; overrides the extension.
- `float_format` — a ``format`` spec such as ``"%.3f"``.  ``None`` (the default) writes the shortest string that round-trips exactly.

**Returns**

The destination path.

**Raises**

- `ValueError` — for an extension with no writer, or when the columns have different lengths.

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
