# pimeta

[![PyPI](https://img.shields.io/pypi/v/pimeta.svg)](https://pypi.org/project/pimeta/)
[![Python versions](https://img.shields.io/pypi/pyversions/pimeta.svg)](https://pypi.org/project/pimeta/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-203%20passing-brightgreen.svg)](https://github.com/ChengkeChou/pimeta/actions/workflows/publish.yml)

**Pure-Python meta-analysis — a faithful, numerically verified port of R's
[`metafor`](https://cran.r-project.org/package=metafor),
[`netmeta`](https://cran.r-project.org/package=netmeta) and
[`meta`](https://cran.r-project.org/package=meta).**

No R installation. No rpy2. No third-party meta-analysis library under the hood —
the numeric engine is a direct NumPy/SciPy transcription of the R reference
implementations, so results reproduce R to floating-point tolerance.

```bash
pip install pimeta
```

```python
import pimeta as ma

r = ma.rma(yi=[0.20, 0.40, 0.10, 0.50], vi=[0.04, 0.05, 0.03, 0.06], method="REML")
print(r.b, r.se, r.tau2, r.I2)     # pooled estimate, SE, tau^2, I^2
```

---

## Why this package

| | R (`metafor`/`netmeta`/`meta`) | **pimeta** |
|---|---|---|
| Runtime | R + C/C++ toolchain | Python + NumPy/SciPy |
| Integration | subprocess / rpy2 | native Python objects |
| Output | R list-like S3 objects | dataclasses with the **same field names** |
| Plots | base graphics / grid | matplotlib `Axes` you can further style |
| Reproducibility | reference implementation | 351 functions verified against R |

Every result object exposes the **same attribute names as the R object's `$`
fields**, so code and documentation written for R translate almost one-to-one:

```python
r = ma.rma(yi, vi, method="REML")
r.b, r.se, r.zval, r.pval, r.ci_lb, r.ci_ub   # R: r$b, r$se, r$zval, r$pval, r$ci.lb, r$ci.ub
r.tau2, r.I2, r.H2, r.QE, r.QEp               # R: r$tau2, r$I2, r$H2, r$QE, r$QEp
r.weights, r.fitted, r.residuals              # R: r$weights, r$fitted, r$residuals
```

---

## What is covered

### Effect sizes — `escalc()`
`RR`, `OR`, `RD`, `AS`, `PETO`, `IRR`, `IRS`, `IRSD`, `MN`, `SMN`,
`MRAW`, `SMCC`, `MD`, `SMD`, `SMDH`, `ROM`, `RPB`, `RBIS`, `D2OR`, `COR`, `UCOR`,
`ZCOR`, `ZPCOR`, `PR`, `PLN`, `PLO`, `PAS`, `PFT`, `PRAW`, `PD` — plus cell
continuity corrections (`add`, `to = "only0" | "all" | "IF0"`) and `drop00`.

### Pooling and heterogeneity
- Fixed-effect and random-effects (`common` / `random`) pooling
- τ² estimators: `DL`, `HE`, `HS`, `SJ`, `ML`, `REML`, `PM`, `EB`, `GENQ`,
  `GENQM`, `QGENQ`, `QGENQM`
- Hartung–Knapp adjustment, Knapp–Hartung (`test = "knha"`, `"adhoc"`)
- Prediction intervals, `QE`/`I²`/`H²`/`R²`, `QE`-based CIs, profile-likelihood τ² CI

### High-level drivers (R-function parity)
`rma()`, `rma.uni` (`rma_uni`), `rma.mv` (`rma_mv`), `rma.glmm` (`rma_glmm`),
`rma.peto`, `rma.mh`, `rma.uni.selmodel`, `metagen`, `metabin`, `metacont`,
`metacor`, `metainc`, `metaprop`, `metarate`, `metamean`, `metacum`, `metainf`,
`metareg`, `trimfill`, `regtest`, `begg_test`, `funnel`, `forest`, `labbe`,
`radial`, `baujat`, `leave1out`, `influence`, `cooks_distance`, `rstudent`,
`hatvalues`, `gosh`, `fsn`, `permutest`, `tes`, `vif`, `profile_rma`.

### Network meta-analysis (`netmeta`)
`netmeta`, `netmetabin`, `netcomb`, `discomb`, `netcomplex`, `netmetareg`,
`netmeasures`, `netrank` (P-scores), `netimpact`, `netpath`,
`netsplit`, `netmatrix`, `netcomparison`, `netbind`, `netcontrib`,
`netdistance`, `netheat`, `nettable`, `netgraph`, `netposet`, `hasse`,
`rankogram`, `decomp_design`, `invmat`, `hatmatrix`, `comps`, `combinations`,
`treats`, `vikor`, `subgroup`, `crossnma2netmeta`, `gemtc2netmeta`, `createC`.

### Publication bias & sensitivity
`regtest` (Egger), `begg_test` (Begg rank test), `trimfill` (L0 / R0 / Q0), `fsn`,
`funnel`, `gosh`, `influence`, `baujat`, `leave1out`, `permutest`.

### Reporting helpers (`meta` package parity)
`cidprop`, `cilayout`, `format_ci`, `forest_dims`, `longarm`, `addpoly`,
`reporter`, `rob`, `aggregate_escalc`, `cbind_escalc`, `rbind_escalc`,
`estimates`, `emmprep`, `points_regplot`.

### S3-style generics
`coef`, `vcov`, `fitted`, `residuals`, `logLik`, `deviance`, `nobs`, `formula`,
`AIC`, `BIC`, `weights`, `sigma`, `df_residual`, `model_matrix`, `simulate`,
`confint`, `predict`, `hatvalues`, `hatvalues_rma`, `dfbetas_rma`,
`rstandard_rma`, `ranef`, `summary`, `summary_text`, `print_summary`, `plot`.

### Readers
`read_rm5`, `read_cdir`, `read_mtv` — parse Cochrane RevMan `.rm5` files and
Cochrane `.cdir` / `.mtv` exports directly into analysis-ready objects.

### Transforms (all 58 of `metafor::transf.*`)
`transf_logit`, `transf_ilogit`, `transf_probit`, `transf_arcsin`,
`transf_ztor`, `transf_ztor2`, `transf_dtocliffd`,
`transf_pft`, `transf_ipft`, `transf_ahw`, `transf_iahw`, `transf_abt`,
`transf_iabt`, `transf_rpbtod`, `transf_dtou1`, `transf_dtou2`,
`transf_dtou3`, and the `_int` / `_mode` variants.

---

## Quick start

### 1. Effect sizes from raw 2×2 data

```python
import pimeta as ma

d = ma.escalc(
    measure="RR",
    ai=[10, 20, 15], bi=[90, 80, 85],     # events / non-events, treatment
    ci=[12, 18, 14], di=[88, 82, 86],     # events / non-events, control
)
print(d.yi)   # log risk ratios
print(d.vi)   # sampling variances
```

### 2. Random-effects pooling + forest plot

```python
r = ma.rma(d.yi, d.vi, method="REML")
print(r.b, r.ci_lb, r.ci_ub)   # 0.0266  -0.3557  0.4090   (log RR; exp() -> 1.027)
print(r.tau2, r.I2, r.QE, r.QEp)  # 0.0  0.0  0.355  0.837

ma.forest(d.yi, d.vi, studlab=["Study A", "Study B", "Study C"],
          pooled=r, effect_type="RR", filename="forest.png")
```

### 3. Meta-regression

```python
# yi, vi come from your study table; dose / baseline_risk are the moderators
meta_reg = ma.rma(yi, vi, mods=[dose, baseline_risk], method="REML")
print(meta_reg.b)          # intercept + slopes
print(meta_reg.QM, meta_reg.QMp)
```

### 4. Network meta-analysis

```python
n = ma.netmeta(
    TE, seTE, treat1, treat2, studlab,
    sm="MD", reference_group="placebo",
)
print(n.TE_random)                       # league-style contrast matrix
print(ma.netrank(n).ranking_random)      # P-scores per treatment
print(ma.netmeasures(n).proportion)      # direct-evidence proportions
ma.netgraph(n, filename="network.png")   # evidence graph
```

### 5. Publication bias

```python
print(ma.regtest(d.yi, d.vi).p)          # Egger's test p-value
print(ma.trimfill(d.yi, d.vi).k0)        # trim-and-fill, L0 estimator
print(ma.fsn(d.yi, d.vi, type="Rosenthal").fsnum)
```

---

## Requirements

- Python ≥ 3.10 (tested on 3.10 – 3.13)
- `numpy` ≥ 1.24, `scipy` ≥ 1.10, `matplotlib` ≥ 3.7
- `pandas` is optional — only needed for DataFrame-shaped output
  (`pip install pimeta[pandas]`)

CI runs the suite on Python 3.12 with `numpy` 2.5, `scipy` 1.18 and
`matplotlib` 3.11; the same example script passes locally with `numpy` 1.26 /
`scipy` 1.16, so both sides of the NumPy 2.0 boundary are covered. Figures use
the headless `Agg` backend and never need a display.

---

## Numerical validation

The library is developed against a differential test suite: identical datasets
are fed to R (`metafor` 5.2-1, `netmeta` 3.7-0, `meta` 8.5-0) and to `pimeta`,
and every reported quantity is compared. Typical agreement:

| Family | Max abs. deviation from R |
|---|---|
| `escalc` / effect sizes | 0 (bit-identical cells) |
| `rma`, `metagen`, τ² estimators | < 1e-12 |
| `rma.mv` / multilevel | < 1e-6 |
| `netmeta` NMA (Q, τ², contrasts) | < 1e-6 |
| `netmetareg`, `netmeasures`, `netimpact` | < 1e-8 |
| `transf.*`, `fmt.*`, measures | < 1e-10 |

Where R has quirks (non-convergence, `NA` subsets, sign conventions of
`contrmat`), the behaviour is reproduced rather than "fixed".

Coverage: `meta` 59/59, `netmeta` 38/38 and `metafor` 278/279 exported
functions. The single exception is `metafor.news()`, a changelog viewer with no
analytic content.

---

## API conventions worth knowing

A few things differ from what a Python-first user might assume. Each one is
deliberate: it matches the R object being mirrored.

| Convention | Detail |
|---|---|
| `result.I2` is a **percentage** | `92.22`, not `0.9222` — identical to R's `r$I2`. Same for `H2` and `R2`. |
| `confint(result)` returns a **dict** | `{"tau2", "ci_lb", "ci_ub", "level"}` — the QP/profile confidence interval for τ², mirroring `confint.rma()`. |
| `result.b` is always an **array** | Even intercept-only models give `np.ndarray` of length 1, like R's `r$b`. Use `float(r.b[0])`. |
| `escalc()` returns **log-scale** `yi` for ratio measures | `RR` / `OR` / `IRR` are log-transformed; use `math.exp()` to get the reported ratio. |
| `netrank(...).ranking_random` holds **P-scores**, not ranks | netmeta 3.7-0 names the field `ranking.random` even though it contains P-scores. `Pmatrix_random` holds the pairwise probability matrix. |
| Plots return a matplotlib `Axes` | Pass `filename="x.png"` to also write the figure, otherwise it stays in memory. The backend is forced to `Agg`, so nothing blocks a headless script. |

`pimeta.__version__` gives the installed version.

---

## Examples

`examples/bcg_meta_analysis.py` is a complete runnable script. It writes the 13
BCG vaccine trials to a CSV, recomputes the effect sizes with `escalc()`, pools
them with `rma()`, draws a forest plot, runs Egger's test and trim-and-fill, fits
a meta-regression on publication year, and finishes with a small network
meta-analysis — checking the numbers against the values R returns for
`metafor::dat.bcg` at every step.

```bash
python examples/bcg_meta_analysis.py
```

---

## Documentation

```python
import pimeta as ma

print(ma.__version__)
help(ma.rma)
help(ma.netmeta)
help(ma.escalc)
```

Every public symbol carries a docstring that names the R function it mirrors.

- Source and issues: <https://github.com/ChengkeChou/pimeta>
- Package index: <https://pypi.org/project/pimeta/>
- Changelog: <https://github.com/ChengkeChou/pimeta/blob/main/CHANGELOG.md>

---

## License

MIT. This package contains an independent Python implementation; it does not
contain or link against R or any GPL R package source code.

## Citation

If you use this package in academic work, please also cite the original R
packages and methods:

- Viechtbauer, W. (2010). Conducting meta-analyses in R with the metafor
  package. *Journal of Statistical Software*, 36(3), 1–48.
- Balduzzi, S., Rücker, G., Nikolakopoulou, A., Papakonstantinou, T.,
  Salanti, G., Efthimiou, O., Schwarzer, G. (2023). netmeta: An R package for
  network meta-analysis using frequentist methods. *Journal of Statistical
  Software*, 106(5), 1–40.
- Balduzzi, S., Rücker, G., Schwarzer, G. (2019). How to perform a
  meta-analysis with R: a practical tutorial. *Evidence-Based Mental Health*,
  22, 153–160.

---

## 中文说明

**pimeta 是 R 生态里 `metafor` + `netmeta` + `meta` 三个包的纯 Python 复现。**
不装 R、不用 rpy2、不依赖任何第三方 meta 分析库，数值引擎是用 NumPy/SciPy 直接转写的，
结果与 R 逐位对齐（多数函数误差 < 1e-12）。

```python
import pimeta as ma

# 效应量
d = ma.escalc(measure="RR", ai=[10, 20, 15], bi=[90, 80, 85],
              ci=[12, 18, 14], di=[88, 82, 86])

# 合并 + 异质性
r = ma.rma(d.yi, d.vi, method="REML")
print(r.b, r.se, r.tau2, r.I2, r.QE, r.QEp)

# 森林图
ma.forest(d.yi, d.vi, studlab=["A", "B", "C"], pooled=r,
          effect_type="RR", filename="forest.png")

# 网状 meta 分析
n = ma.netmeta(TE, seTE, treat1, treat2, studlab, sm="MD")
print(ma.netrank(n).ranking_random)   # 各处理 P-score

# 发表偏倚
print(ma.regtest(d.yi, d.vi).p)     # Egger
print(ma.trimfill(d.yi, d.vi).k0)   # 剪补法
```

**结果对象的字段名和 R 完全一致**，R 代码里的 `r$b`、`r$tau2`、`r$I2`、
`r$ci.lb` 在 Python 里就是 `r.b`、`r.tau2`、`r.I2`、`r.ci_lb`，
已有的 R 脚本和教材可以近乎一对一翻译。

**几个需要知道的约定**（都是为了和 R 保持一致）：

| 约定 | 说明 |
|---|---|
| `r.I2` 是**百分数** | `92.22` 而不是 `0.9222`，和 R 的 `r$I2` 相同；`H2`、`R2` 同理 |
| `confint(r)` 返回 **dict** | `{"tau2", "ci_lb", "ci_ub", "level"}`，对应 R 的 `confint.rma()` 给出的 τ² 置信区间 |
| `r.b` 永远是**数组** | 即使只有截距也是长度为 1 的 `np.ndarray`，取值用 `float(r.b[0])` |
| `escalc()` 对 RR/OR/IRR 返回**对数尺度** | 要报告比值本身请用 `math.exp()` |
| 画图返回 matplotlib `Axes` | 加 `filename="x.png"` 才会落盘；后端强制 `Agg`，脚本里不会弹窗卡住 |

完整可运行示例见 [`examples/bcg_meta_analysis.py`](examples/bcg_meta_analysis.py)：
从 CSV 读入 13 个卡介苗试验 → 算效应量 → 合并 → 森林图 → Egger / 剪补法 →
元回归 → 网状 meta 分析，每一步都和 R 的 `metafor::dat.bcg` 结果对拍。

覆盖范围：`meta` 59/59、`netmeta` 38/38、`metafor` 278/279 个导出函数
（唯一未实现的是 `metafor.news()`，它只是查看更新日志，不含分析功能）。
