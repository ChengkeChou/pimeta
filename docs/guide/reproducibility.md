# Reproducible analyses: records, fingerprints and replay

A meta-analysis is a computation, and a computation that cannot be repeated is
an opinion. This page describes the machinery that makes a `pimeta` result
carry its own evidence: what data went in, what was asked of it, what the
software was, what came out, and whether re-running it gives the same answer.

Nothing here is required to fit a model. It is what you add when the number has
to survive a reviewer, a co-author, or yourself in two years.

**Every analysis in `pimeta` has to be recordable.** That is a design rule, not
an optional extra: if a function can produce a number, the number must be able
to leave the process with enough context to be rebuilt — the data fingerprint,
the arguments, the settings that can move it, and the values themselves. A new
model is not finished until `save_record` → `verify` → `replay` round-trips it,
and its test file proves it. The sections below describe that machinery; the
rest of the manual assumes it.

## The idea in one example

```python
import pandas as pd
import pimeta as ma
from pimeta.record import save_record, load_record

df = pd.read_csv("studies.csv")          # study, TE, seTE2

fit = ma.rma(df["TE"], vi=df["seTE2"], method="REML", slab=df["study"])

save_record(
    "m1.json", fit,
    data=df,
    mode="bundle",
    function="rma",
    call={"yi": "TE", "vi": "seTE2", "method": "REML", "slab": "study"},
)
```

Later — different machine, different year — the record is replayed:

```python
record = load_record("m1.json")
print(record.summary())
record.verify()     # the bundled data still hashes to the recorded digests
record.replay()     # re-fit; every number is compared against the record
```

If any number moved, `replay()` raises `RecordMismatch` and names the field.

## The five blocks

Every record is a single JSON file with five blocks. The split is deliberate:
each block answers a different question, and they are useful separately.

| Block | Question it answers | Chinese label |
|---|---|---|
| `inputs` | What data went in, and is it still the same data? | 输入数据 |
| `analysis` | What was asked of the data? | 分析配置 |
| `process` | What happened while it ran? | 运算过程 |
| `environment` | Which software and machine produced it? | 运行环境 |
| `results` | What came out? | 结果与诊断 |

```json
{
  "schema": "pimeta.record/1",
  "kind": "rma",
  "created_utc": "2026-10-21T15:40:02+00:00",
  "inputs": {
    "mode": "bundle",
    "n_rows": 20,
    "columns": ["study", "TE", "seTE2"],
    "analysis_columns": ["study", "TE", "seTE2"],
    "arrays": {"yi": [...], "vi": [...]},
    "fingerprint": {
      "algorithm": "sha256",
      "version": 1,
      "basis": "data",
      "source": "sha256:fac1d318e3a34ff1235daca8b586e867",
      "analysis": "sha256:e277e72c6f6f0ecc7b7a00b28ddabbc1"
    },
    "data": {"study": ["S1", "..."], "TE": [0.2, "..."]}
  },
  "analysis": {
    "settings": {"method": "REML", "test": "z", "level": 0.95, "tau2_fix": false},
    "function": "rma",
    "call": {"yi": "TE", "vi": "seTE2", "method": "REML", "slab": "study"},
    "pass_data": true
  },
  "process": {},
  "environment": {"pimeta": "1.1.0", "numpy": "2.5.4", "python": "3.12.3", "...": "..."},
  "results": {"class": "RmaResult", "values": {"b": [0.2596], "tau2": 0.0, "...": "..."}}
}
```

## digest or bundle

| | `mode="digest"` | `mode="bundle"` |
|---|---|---|
| Data copied into the record | no | yes |
| Record can verify itself | no — needs the original file | yes |
| Record can replay itself | no | yes |
| Safe to share publicly | typically yes | only if the data is |

`digest` is the default because it never copies participant-level data. Use
`bundle` for anything that has to be reproducible without the original file.

Both modes record the same fingerprints, so a bundle can always be checked
against the file it came from:

```python
record.verify(df)      # raises RecordMismatch if df is not the file used
```

## Two fingerprints, because they answer two questions

One hash cannot serve both "is this the same file?" and "would the same numbers
come out?", so there are two.

```python
from pimeta.record import source_fingerprint, analysis_fingerprint

source_fingerprint(df)                     # every column, in file order, verbatim
analysis_fingerprint(df, columns=["TE", "seTE2"], settings={"method": "REML"})
```

The difference matters:

| Change | `source` | `analysis` |
|---|---|---|
| Study label `"S3"` → `"Smith 2019"` | changes | unchanged |
| Column order in the file | changes | unchanged |
| Unused column added | changes | unchanged |
| `method="REML"` → `"DL"` | unchanged | changes |
| One `yi` value nudged | changes | changes |

The rule is that anything which cannot move a coefficient is not part of
*"the same analysis"*. `slab`, `digits` and `measure` fall on that side;
`method`, `test`, `level` and the numbers themselves do not.

Both digests are SHA-256 over a canonical form: sorted keys, no whitespace,
floats rendered with Python's shortest round-trip `repr`. A digest computed on
Windows equals one computed on Linux for the same data.

## NaN and infinity survive the file

Plain `json.dumps` writes `NaN` and `Infinity`, which are not JSON — every
browser, PostgreSQL's `jsonb` and most strict readers reject them. Writing
`null` instead would be worse: in meta-analysis `tau2 = 0` is a real estimate,
`I2 = nan` with `k = 1` means *undefined*, and `H2 = inf` means the pooled
variance collapsed. Those are three different facts.

`pimeta` writes them as tagged objects:

```json
{"$pimeta": "nan"}
{"$pimeta": "inf"}
{"$pimeta": "-inf"}
```

so the file is valid JSON *and* lossless:

```python
from pimeta.record.jsonio import dumps, loads
import math

back = loads(dumps({"tau2": 0.0, "I2": float("nan"), "H2": float("inf")}))
math.isnan(back["I2"])          # True
back["H2"] == math.inf          # True
```

Records are written atomically — to a temporary file, then renamed — so an
interrupted save cannot truncate an existing record. Line endings are forced to
`\n`, which keeps records byte-comparable across platforms.

## Saving and reading tables

The reader and the writer accept each other's output, so a round trip through a
file is a normal workflow rather than a one-way pipe:

```python
from pimeta.record import read_table, write_table

write_table("studies.csv", df)
write_table("studies.npz", df)        # numbers + labels, no pickle
back = read_table("studies.npz", infer=True)
```

| Format | Notes |
|---|---|
| `.csv`, `.tsv`, `.tab` | text; floats written at full round-trip precision by default |
| `.json` | a column dict; NaN/infinity preserved |
| `.jsonl` | one record per row |
| `.npz` | numeric and string columns; object columns are refused rather than pickled |

Text output defaults to `repr(float)`, the shortest string that parses back to
the identical double. Pass `float_format="%.3f"` when a rounded table is wanted
for reading — and accept that it is then a report, not a record.

## Storing several fits: the `estimates store` equivalent

Fitting one model is rarely the end. A paper compares a common-effects and a
random-effects fit, or three estimators:

```python
from pimeta.record import EstimatesStore

models = EstimatesStore("primary")
models.store("fe", ma.rma(df["TE"], vi=df["seTE2"], method="FE"))
models.store("re", fit)

print(models.summary())
models.compare()["model"]        # ['fe', 're']
models.save("estimates.json")
```

Each entry keeps a full record, so a stored model is traceable on its own. It
also keeps the live result object, so `restore()` hands back the real thing
while the process runs:

```python
models.restore("re").tau2
```

After `save()`/`load()` the live object is gone, and `restore()` raises rather
than inventing one. A record can prove what the numbers were; it cannot pretend
to be the fitted model. What still works on a loaded entry:

```python
reloaded = EstimatesStore.load("estimates.json")
reloaded["re"].coef_table()
reloaded["re"].record.replay()
```

## Recording a network meta-analysis

`netmeta()` takes column names, so nothing extra is needed — the same pattern
as `rma()`:

```python
import pimeta as ma
from pimeta.record import save_record

nm = ma.netmeta("TE", "seTE", "treat1", "treat2", data=df, sm="MD")

save_record(
    "nma.json", nm,
    data=df,
    mode="bundle",
    function="netmeta",
    call={"TE": "TE", "seTE": "seTE",
          "treat1": "treat1", "treat2": "treat2", "sm": "MD"},
)
```

`replay()` re-fits the network and compares every numeric field of the fitted
object — the contrast matrices, `TE.random`, the decomposition `Q` terms and
the P-scores — so a drift anywhere in the network is caught, not just in the
headline estimate.

The τ² estimator is part of the analysis fingerprint: a DL fit and a REML fit
of the same table get different `analysis` digests, because they are different
analyses. `reference_group` and `baseline_reference` are recorded for the same
reason — they change which contrasts are reported.

A network has no coefficient vector, so `EstimatesStore.coef_table()` returns
an empty table for one. That is deliberate: `compare()` simply contributes no
rows rather than inventing a coefficient that does not exist.

## What replay actually proves

`replay()` re-runs the recorded call with the recorded data and compares every
numeric field. It is a strong check, not a total one:

- **Proves** the same inputs and settings give the same numbers *on the
  installed version*.
- **Does not prove** the analysis was right. A record of a misspecified model
  is a faithful record of a misspecified model.
- **Does not prove** bit-for-bit equality with an older version. Floating-point
  results can shift in the last bits between NumPy/BLAS versions, which is why
  `environment` records them and `replay()` compares with a tolerance
  (`rtol=1e-9`, `atol=1e-12` by default).

### When `call=` uses column names, and when it does not

Most functions take column *names* plus a `data=` frame, so the record infers
that the bundled table should be passed back as `data=` on replay:

```python
save_record("m1.json", fit, data=df,
            function="rma", call={"yi": "TE", "vi": "seTE2", "method": "REML"})
```

Some functions take the whole table as their `data=` argument — `reitsma()` is
the clearest case. The inference cannot tell, because the call contains no
column names, so say so explicitly:

```python
save_record("dta.json", dta_fit, data=counts,
            mode="bundle", function="reitsma",
            call={"method": "reml"}, pass_data=True)
```

`pass_data=True` always injects the bundled data. `pass_data=False` never
does, which is what you want for a function that takes no data at all.

## Declaring the decisions: `AnalysisSpec`

A record says what was run. A specification says what *could* have been run, and
why each alternative was defensible. It is the object you write before you look
at the results, and the one that makes a multiverse or a sensitivity table a
declared set of analyses rather than an improvised one.

```python
import pimeta as ma

spec = ma.AnalysisSpec(decisions=(
    ma.Decision(
        name="method", values=("REML", "DL", "FE"), default="REML",
        reasons={
            "REML": "the default here; less biased than DL at small k",
            "DL": "what most published meta-analyses used, so it aids comparison",
            "FE": "the assumption that there is no heterogeneity at all",
        },
    ),
    ma.Decision(
        name="test", values=("z", "knha"), default="z",
        reasons={
            "z": "the nominal reference; slightly liberal at small k",
            "knha": "wider when the studies disagree; better coverage at small k",
        },
    ),
))
spec.size()          # 6
spec.default()       # the specification made of the declared defaults
one = spec.expand()[3]
print(one)           # 4f9c1a0b2d3e: method=DL, test=knha
```

**Two rules keep this from becoming a combinatorics toy.** A decision enters
only if more than one value is statistically defensible — a one-valued
"decision" raises, because that is the method, not a choice. And the number of
specifications is a *consequence* of what was declared; `expand()` is a Cartesian
product, not a search, and nothing here rewards a bigger grid.

**Ids come from the choices and nothing else.** Each specification's id is a
`sha256:` digest of its own choices, so it does not change when a justification
is reworded, when the decisions are declared in a different order, or when the
grid is expanded twice. Two runs that made the same decisions collide on
purpose: that is how they are recognised as the same analysis, and how a result
is traced back to the specification that produced it.

A specification round-trips through a file, and the file records its own size so
a hand edit is caught rather than half-trusted:

```python
spec.save("spec.json")
again = ma.AnalysisSpec.load("spec.json")
assert [s.id for s in again.expand()] == [s.id for s in spec.expand()]
```

### Running the grid

```python
def run(choices):
    return ma.rma(yi, vi=vi, method=choices["method"], test=choices["test"])

outcomes = ma.execute(spec, run)
usable = [o for o in outcomes if o.ok]
failed = [o for o in outcomes if not o.ok]
print(f"{len(usable)} usable of {len(outcomes)}; {len(failed)} failed")
for o in failed:
    print(o.id, o.error)
```

**A failed specification is an outcome, not an exception.** It keeps its id, its
choices and its error message, and it stays in the list. Dropping the runs that
did not converge is the largest silent bias in a published multiverse, and it
starts with a bare `try: ... except: pass` in the loop. `on_error="raise"` is
there for scripts that genuinely cannot continue.

Each outcome also carries the diagnostic codes of its fit
(`outcome.diagnostics`), so "usable" and "warned" are different columns rather
than the same one.

### Recording which specification produced a fit

A saved record can name the specification behind it, which is what makes a
grid's results traceable one fit at a time:

```python
from pimeta.record import save_record, load_record

outcome = outcomes[3]
save_record("m3.json", outcome.result, data=df, mode="bundle",
            function="rma", specification=outcome.id)

load_record("m3.json").specification
# {'id': 'sha256:4f9c...', 'choices': {'method': 'DL', 'test': 'knha'}}
```

The object, its id string and a plain `{"id": ..., "choices": ...}` mapping are
all accepted, so the record layer does not depend on `pimeta.spec` and a script
that only kept the id can still write a correct record. Attaching a
specification **does not change the analysis fingerprint** — the choices are
already in `settings` whenever they can move a number — so records that verified
before still verify afterwards.

## Practical advice

1. **Save one record per model, in the repository that holds the paper.** A
   record is small; a lost record is not recoverable.
2. **Use `mode="bundle"` for the final analysis** and `digest` for everything
   exploratory.
3. **Quote the `analysis` fingerprint in the manuscript or supplement.** It is
   short enough to print and it pins the exact numbers used.
4. **Keep `call=` accurate.** Replay is only as good as the recorded call; if it
   drifts from what was actually run, the record is fiction.

## Parallel execution, and why it is not everywhere

`validation.run_scenario` and `DynamicHetResult.test_trend` accept `n_jobs`.
Those are the only places in the package where work is spread over processes,
and the reason is not timidity. **A parallelism that changes a number is not a
speed-up, it is a different analysis.**

Every repetition in those two loops is a pure function of data drawn *before*
the loop starts. `run_scenario` simulates all `reps` datasets first, serially,
from one seeded generator; the workers then only fit them. Nothing is
accumulated across workers, results are collected in submission order, and the
answer is identical to the serial loop **to the last bit** — `tests/test_parallel.py`
compares `float.hex()`, not `pytest.approx`.

What is deliberately *not* parallelised:

| Candidate | Why not |
|---|---|
| the REML optimiser | BLAS already threads the linear algebra; what is left is per-study Python, which has to be vectorised rather than multiplied |
| a cumulative / running meta-analysis | the answer is a sum, and floating-point addition is not associative, so splitting it across workers moves the last digits |
| leave-one-out sweeps | the same, and each refit is cheap next to the pickling |

The callable handed to a worker must be picklable — a module-level function or a
`functools.partial` of one — because a lambda or a closure cannot be sent to
another process. `pimeta.parallel.check_picklable` reports that where the caller
can fix it instead of inside a child process. The short version: parallel over
independent repetitions, never over an accumulation.

## See also

- [Data frames and formulas](data-frames.md) — how `data=` resolves column names
- [Interoperability](interop.md) — pandas, polars, sklearn, statsmodels
- [Command line](cli.md) — the same table formats from a shell
