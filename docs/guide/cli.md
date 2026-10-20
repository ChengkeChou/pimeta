# Command line

The package installs a `pimeta` command; `python -m pimeta` is equivalent, so
nothing needs a script.

```bash
pip install pimeta
pimeta --version
pimeta --help
```

Input is CSV, TSV or JSON: one row per study, with a header row naming the
columns. Numbers may be quoted; blank cells, `NA`, `NaN` and `nan` are read as
missing.

## `pimeta fit` — pool effect sizes

```bash
pimeta fit studies.csv --yi TE --vi seTE2
```

```
term     estimate  std.error  statistic  p.value   conf.low  conf.high
intrcpt  -0.71453  0.15423    4.6331     3.6e-06   -1.01682  -0.41224
```

| Option | Meaning |
|---|---|
| `--yi COL` | effect-size column (required) |
| `--vi COL` | variance column |
| `--sei COL` | standard-error column, instead of `--vi` |
| `--mods "~ a + b"` | moderator formula; string columns become R-style dummies |
| `--slab COL` | study-label column |
| `--method` | τ² estimator (default `REML`) |
| `--test` | `z`, `knha`, `t` or `adhoc` |
| `--level` | confidence level (default `0.95`) |
| `--forest FILE` | also write a forest plot |
| `-o FILE` | write the table to a file (adds a `*_glance` file with model-level statistics) |
| `--format` | `tsv` (default), `csv` or `json` |
| `--no-warn-missing` | drop incomplete rows without warning |

Moderators work exactly as in Python — the CLI passes column names straight to
`mods=`, so a formula expands the same way:

```bash
pimeta fit studies.csv --yi TE --vi seTE2 --mods "~ dose * arm"
```

## `pimeta escalc` — effect sizes from raw data

```bash
pimeta escalc raw.csv --measure RR --ai ai --bi bi --ci ci --di di
```

```
yi        vi
0.28768   0.012345
...
```

Continuous data uses `--n1i --m1i --sd1i --n2i --m2i --sd2i`; ratios use
`--xi --ni` or `--x1i --t1i --x2i --t2i`. `--add` sets the continuity
correction, `--keep00` keeps studies with zero events in both arms.

## `pimeta network` — network meta-analysis

```bash
pimeta network nma.csv --te TE --se seTE --treat1 t1 --treat2 t2 \
    --studlab studlab --sm MD -o league.csv --graph network.png
```

The output is the treatment-by-treatment estimate matrix (rows and columns are
the sorted treatment names). `--random`/`--common` pick the effects model
(random is the default).

## Pipes and scripts

Output goes to stdout as TSV unless `-o` is given, so it composes:

```bash
pimeta fit studies.csv --yi TE --vi seTE2 --format csv | column -t -s,
pimeta fit studies.csv --yi TE --vi seTE2 --format json | jq '.estimate'
```

Exit codes: `0` success, `2` bad arguments or unreadable data (the message goes
to stderr), so a Makefile or CI step can rely on them.

## Missing values

Rows with a missing effect, variance or moderator are dropped — the way R's
`na.action` does — but pimeta **says so**:

```
pimeta: warning: dropped 2 of 13 studies with missing values in 'yi', 'vi'
        (rows 3, 7); pass warn_missing=False to silence this
```

That is the same behaviour as `rma(..., warn_missing=True)` in Python; use
`--no-warn-missing` to drop silently.
