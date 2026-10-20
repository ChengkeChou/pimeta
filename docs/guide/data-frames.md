# Data frames and formulas

Every data-consuming entry point accepts a **frame plus column names**, and
`mods` additionally accepts an R-style **formula**. This is the closest Python
can get to R's non-standard evaluation, and it means R code translates without
unpacking columns by hand.

```python
import pimeta as ma

df = {
    "yi": [0.20, 0.40, 0.10, 0.50, 0.30, 0.15],
    "vi": [0.04, 0.05, 0.03, 0.06, 0.02, 0.05],
    "dose": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "year": [1990, 1995, 2000, 2005, 2010, 2015],
    "arm": ["A", "B", "A", "B", "A", "B"],
    "slab": ["S1", "S2", "S3", "S4", "S5", "S6"],
}

r = ma.rma("yi", "vi", mods="~ dose + year", slab="slab", data=df)
print(r.b)
```

## Accepted frames

| Type | Example |
|---|---|
| `dict` of sequences | `{"yi": [...], "vi": [...]}` |
| pandas `DataFrame` | `pd.read_csv("studies.csv")` |
| polars `DataFrame` | `pl.read_csv("studies.csv")` |
| NumPy structured array | `np.array(..., dtype=[("yi", "f8"), ...])` |
| list of records | `[{"yi": 0.2, "vi": 0.04}, ...]` |

Anything with `.columns` and `.to_dict()` is treated as a frame, so most
dataframe libraries work without an explicit adapter.

## Column names everywhere

Any argument that expects a vector also accepts a column name:

```python
import pimeta as ma

ma.rma("yi", "vi", data=df)                    # effect + variance
ma.rma("yi", sei="sei", data=df)               # standard error instead
ma.rma("yi", "vi", slab="slab", data=df)       # study labels
ma.escalc("RR", "ai", "bi", "ci", "di", data=df)
ma.metabin("e1", "n1", "e2", "n2", data=df)
ma.netmeta("TE", "seTE", "treat1", "treat2", "studlab", data=df)
```

A list of strings is treated as data unless `data=` is supplied **and** every
string names a real column — so `slab=["Study 1", "Study 2"]` keeps working.

## Formulas

`mods` accepts a one-sided formula; the response always travels separately as
`yi`.

| Formula | Design-matrix columns |
|---|---|
| `~ dose` | `dose` |
| `~ dose + year` | `dose`, `year` |
| `~ dose*year` | `dose`, `year`, `dose:year` |
| `~ dose:year` | `dose:year` only |
| `~ 0 + dose` | no intercept (`~ dose - 1` is equivalent) |
| `~ .` | every other column in `data` |
| `` ~ `dose.gy` `` | backticks protect unusual names |

```python
import pimeta as ma

r1 = ma.rma("yi", "vi", mods="~ dose", data=df)
r2 = ma.rma("yi", "vi", mods="~ dose * arm", data=df)   # main effects + interaction
r3 = ma.rma("yi", "vi", mods="~ 0 + dose", data=df)     # no intercept
```

## Categorical moderators

String columns are dummy-coded the way R's `model.matrix` does it: the first
level (alphabetically) is dropped and the remaining levels are named
`arm[B]`, `arm[C]`, …

```python
import pimeta as ma

r = ma.rma("yi", "vi", mods="~ arm", data=df)
print(r.b)      # intercept = level "A", then the shift for "B"
```

`rma` adds the intercept itself, so the design matrix from a formula never
duplicates it.

## Inspecting the pieces

```python
import pimeta as ma

spec = ma.parse_formula("~ dose * arm")
print(spec)                 # FormulaSpec(~ 'dose + arm + dose:arm')

X, names = ma.design_matrix("~ dose + arm", df)
print(names)                # ['(Intercept)', 'dose', 'arm[B]']

y, X, names = ma.model_frame("yi ~ dose", df)

cols = ma.as_columns(df)    # normalise any frame to {name: ndarray}
```

## Errors you might see

| Message | Cause |
|---|---|
| `looks like a column name but no 'data' was supplied` | a bare string was passed without `data=` |
| `term 'x' in the formula does not name a column` | typo, or the column lives in a different frame |
| `'mods' must be one-sided` | `mods="yi ~ dose"`; the response is `yi` itself |
| `column 'x' not found in 'data'` | `lookup` failed; the message lists the available columns |
