# pimeta 使用手册

这份手册是"照着敲就能跑"的版本。**每一个 Python 代码块下一段的 `text` 块，都是真实运行输出**，
由 `tools/run_docs_examples.py` 生成并校验；`tests/test_docs_examples.py` 会在 CI 里重跑，
所以示例不会过期。

!!! note "示例是按顺序执行的"
    下面所有示例共享同一个 Python 命名空间，后面的例子可以直接用前面定义的数据和结果对象。
    想单独跑某一段，把前面初始化 `import pimeta as ma` 和数据的几行一起复制即可。

!!! tip "想直接看结果？"
    [`examples/`](https://github.com/ChengkeChou/pimeta/tree/main/examples) 里有 4 个**已经跑好**的
    notebook，输出和图都嵌在里面，在 GitHub 上直接可读：
    `01-quickstart`（合并到森林图）、`02-meta-regression`（元回归与发表偏倚）、
    `03-network-meta-analysis`（网状 meta）、`04-ecosystem`（pandas / scikit-learn / xarray）。

!!! abstract "English"
    This manual is also available in [English](../usage.md).

## 1. 第一次合并

最常用的入口是 `rma()`：给它效应量 `yi` 和方差 `vi`，它按随机效应模型合并。

```python
import pimeta as ma

yi = [0.10, 0.65, 0.20, 0.90, 0.15, 1.10, 0.05, 0.75, 0.40, 1.00]
vi = [0.04, 0.06, 0.05, 0.08, 0.04, 0.09, 0.05, 0.07, 0.06, 0.08]

r = ma.rma(yi, vi, method="REML")

print(f"k            = {r.k}")
print(f"合并效应 b    = {r.b[0]:.6f}")
print(f"标准误 se     = {r.se[0]:.6f}")
print(f"95% CI       = [{r.ci_lb[0]:.6f}, {r.ci_ub[0]:.6f}]")
print(f"tau^2        = {r.tau2:.6f}")
print(f"I^2 (%)      = {r.I2:.2f}")
print(f"Q            = {r.QE:.4f} (df={r.k - r.p}, p={r.QEp:.4f})")
```

```text
k            = 10
合并效应 b    = 0.491362
标准误 se     = 0.123249
95% CI       = [0.249799, 0.732926]
tau^2        = 0.091675
I^2 (%)      = 61.23
Q            = 22.7202 (df=9, p=0.0069)
```

`r.b` 永远是数组（即使只有截距），取数用 `float(r.b[0])`。字段名和 R 的 `$` 完全对应：
`r$b` → `r.b`，`r$ci.lb` → `r.ci_lb`，`r$tau2` → `r.tau2`。

## 2. 数据从哪来

三种输入方式完全等价，选你手头有的那种。

```python
import numpy as np

# (a) 直接把数组传进去——上面已经这样做了

# (b) 传列名 + 一个字典，避免长列表对不齐
dat = {
    "study": ["Adams 2019", "Baker 2020", "Chen 2021", "Diaz 2022", "Evans 2022",
              "Ferrari 2023", "Garcia 2023", "Huang 2024", "Ito 2024", "Jensen 2025"],
    "yi": yi,
    "vi": vi,
    "dose": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
}
r2 = ma.rma("yi", "vi", data=dat)

# (c) 公式接口：mods 写成 "~ 1 + dose"
r3 = ma.rma("yi", "vi", mods="~ dose", data=dat)

print("(b) 与 (a) 结果一致:", np.allclose(r2.b, r.b))
print("(c) 系数名:", r3.coefnames)
print("(c) 系数:", [round(float(z), 6) for z in r3.b])
```

```text
(b) 与 (a) 结果一致: True
(c) 系数名: ['intrcpt', 'dose']
(c) 系数: [0.247874, 0.004535]
```

有 pandas 就直接丢 DataFrame，列名一样用：

```python
import pandas as pd

df = pd.DataFrame(dat)
r4 = ma.rma("yi", "vi", data=df)
print(r4.k, round(float(r4.b[0]), 6))
```

```text
10 0.491362
```

## 3. 效应量：`escalc()`

先从最常见的 2×2 表开始。`escalc()` 对**比值类**指标（OR/RR/IRR）返回的是**对数尺度**，
要报告数值本身需要 `exp()`。

```python
ai = [12, 15, 9, 20]      # 试验组事件数
bi = [88, 85, 91, 80]     # 试验组非事件数
ci = [20, 22, 14, 25]     # 对照组事件数
di = [80, 78, 86, 75]     # 对照组非事件数

es_or = ma.escalc("OR", ai=ai, bi=bi, ci=ci, di=di)

print("measure:", es_or.measure, "| k =", es_or.k)
print("log OR:", [round(float(v), 6) for v in es_or.yi])
print("OR    :", [round(float(np.exp(v)), 4) for v in es_or.yi])
print("方差  :", [round(float(v), 6) for v in es_or.vi])
```

```text
measure: OR | k = 4
log OR: [-0.606136, -0.468935, -0.498345, -0.287682]
OR    : [0.5455, 0.6257, 0.6075, 0.75]
方差  : [0.157197, 0.136706, 0.205157, 0.115833]
```

连续型与单组比例：

```python
es_md = ma.escalc("MD",
                  n1i=[30, 25, 40], m1i=[5.2, 4.9, 6.1], sd1i=[1.1, 1.3, 0.9],
                  n2i=[28, 26, 38], m2i=[4.6, 4.4, 5.3], sd2i=[1.0, 1.2, 1.1])
es_pr = ma.escalc("PLO", xi=[3, 7, 12], ni=[40, 50, 60])

print("MD  :", [round(float(v), 6) for v in es_md.yi])
print("log-odds(比例):", [round(float(v), 6) for v in es_pr.yi])
print("还原成比例:", [round(float(v), 4) for v in 1 / (1 + np.exp(-np.asarray(es_pr.yi)))])
```

```text
MD  : [0.6, 0.5, 0.8]
log-odds(比例): [-2.512306, -1.81529, -1.386294]
还原成比例: [0.075, 0.14, 0.2]
```

`escalc()` 一共支持 metafor 的主要指标：二分类 `OR`/`RR`/`RD`/`AS`/`PETO`，连续型
`MD`/`SMD`/`SMDH`/`ROM`/`CVR`，单组比例 `PLN`/`PLO`/`PAS`/`PFT`/`PR`，相关系数
`ZCOR`/`COR`，以及率比 `IRR`/`IRLN`/`IRS`。完整签名见 API 文档。

## 4. `rma()` 的选项

τ² 估计量、检验方法、置信区间这三件事是最常改的。

```python
for method in ("DL", "REML", "PM", "HS", "EB", "ML"):
    fit = ma.rma(yi, vi, method=method)
    print(f"{method:5s} tau2={fit.tau2:.6f}  I2={fit.I2:6.2f}%  b={fit.b[0]:.6f}")

# Knapp-Hartung 调整会让 CI 变宽，小样本时更稳妥
kh = ma.rma(yi, vi, test="knha")
print(f"\nknha: t={kh.zval[0]:.4f} p={kh.pval[0]:.4f} CI=[{kh.ci_lb[0]:.4f}, {kh.ci_ub[0]:.4f}]")
```

```text
DL    tau2=0.088490  I2= 60.39%  b=0.490553
REML  tau2=0.091675  I2= 61.23%  b=0.491362
PM    tau2=0.094703  I2= 62.00%  b=0.492102
HS    tau2=0.073218  I2= 55.78%  b=0.486144
EB    tau2=0.094704  I2= 62.00%  b=0.492102
ML    tau2=0.075659  I2= 56.59%  b=0.486914

knha: t=3.9484 p=0.0034 CI=[0.2098, 0.7729]
```

## 5. 异质性与预测区间

```python
print(f"H^2   = {r.H2:.4f}")
print(f"I^2   = {r.I2:.2f}%")
print(f"tau   = {np.sqrt(r.tau2):.6f}")

# 未来研究的预测区间（metafor 的 predict()）
pred = ma.predict(r)
print("预测值:", f"{pred.pred:.4f}", " 95% CI:", f"[{pred.ci_lb:.4f}, {pred.ci_ub:.4f}]")
print("预测区间:", f"[{pred.pi_lb:.4f}, {pred.pi_ub:.4f}]")

# tau^2 自己的置信区间
ci_tau2 = ma.confint(r)
print("tau^2 CI:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in ci_tau2.items()})
```

```text
H^2   = 2.5793
I^2   = 61.23%
tau   = 0.302779
预测值: 0.4914  95% CI: [0.2498, 0.7329]
预测区间: [-0.2481, 1.2309]
tau^2 CI: {'tau2': 0.091675, 'ci_lb': 0.011901, 'ci_ub': 0.465019, 'level': 0.95}
```

## 6. 亚组与元回归

```python
reg = ma.rma(yi, vi, mods="~ dose", data=dat)
print("截距  :", round(float(reg.b[0]), 6))
print("dose  :", round(float(reg.b[1]), 6))
print("QM    :", f"{reg.QM:.4f}", "df =", reg.QMdf[0], "p =", f"{reg.QMp:.4f}")
print("tau^2 :", f"{r.tau2:.6f}", "->", f"{reg.tau2:.6f}")

# 分类变量会被自动哑变量化，系数名沿用 R 的 "arm[B]" 风格
dat2 = dict(dat, arm=["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"])
reg2 = ma.rma("yi", "vi", mods="~ arm", data=dat2)
print("系数名:", reg2.coefnames)
```

```text
截距  : 0.247874
dose  : 0.004535
QM    : 1.1431 df = 1 p = 0.2850
tau^2 : 0.091675 -> 0.089292
系数名: ['intrcpt', 'arm[B]']
```

## 7. 发表偏倚

```python
rt = ma.regtest(yi, vi)          # Egger 回归检验
tf = ma.trimfill(yi, vi)         # trim-and-fill
mb = ma.metabias(r)              # 在 rma 结果上做检验

print("regtest   p =", f"{rt.p:.4f}", " 斜率 =", f"{rt.slope:.4f}")
print("trimfill  补齐研究数 =", tf.k0, " 调整后效应 =", f"{tf.filled_effect:.6f}")
print("metabias  p =", f"{mb.pval:.4g}" if mb.pval is not None else f"k={mb.k} < k_min={mb.k_min}，按惯例不报告")
```

```text
regtest   p = 0.0000  斜率 = 10.4959
trimfill  补齐研究数 = 1  调整后效应 = 0.438025
metabias  p = 3.12e-05
```

真实的发表偏倚检验要求 `k >= 10`，上面的 k=4 只是演示调用方式。

## 8. 多水平 / 多元模型：`rma_mv()`

同一个研究报了多个效应量时，需要把研究内的相关性建进去。

```python
study = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E", "F", "F"]
y_mv = [0.10, 0.35, 0.55, 0.20, 0.80, 0.65,
        0.15, 0.45, 0.95, 0.70, 0.30, 0.60]
v_mv = [0.05, 0.04, 0.06, 0.05, 0.07, 0.06,
        0.04, 0.05, 0.08, 0.06, 0.05, 0.05]

mv = ma.rma_mv(y_mv, v_mv, random="~ 1 | study", data={"study": study})

print("k =", mv.k, "个效应量,", len(set(study)), "个研究")
print("beta =", f"{float(np.atleast_1d(mv.beta)[0]):.6f}")
print("sigma^2 =", [round(float(t), 6) for t in np.atleast_1d(mv.sigma2)])
print("随机效应公式:", mv.random, "| 结构:", mv.struct)
```

```text
k = 12 个效应量, 6 个研究
beta = 0.458086
sigma^2 = [0.023807]
随机效应公式: ~ 1 | study | 结构: CS
```

!!! warning "`rma_mv` 的方差分量叫 `sigma2`"
    这是照抄 R 的 `rma.mv`：多元模型的方差分量存在 `$sigma2` 里（长度等于随机效应项个数），
    `$tau2` 不使用。一元 `rma()` 才是 `tau2`。

## 9. 网络 meta 分析：`netmeta()`

```python
nm = ma.netmeta(
    TE=[0.30, -0.20, 0.50, 0.10, -0.15, 0.25],
    seTE=[0.12, 0.15, 0.18, 0.14, 0.11, 0.13],
    treat1=["A", "B", "A", "C", "B", "C"],
    treat2=["B", "A", "C", "A", "C", "B"],
    studlab=["S1", "S2", "S3", "S4", "S5", "S6"],
    sm="RR",
)

print("处理组:", list(nm.trts))
print("比较数:", nm.m, " 治疗数:", nm.n)
print("随机效应 tau^2 =", f"{nm.tau2:.6f}")
print("总 Q =", f"{nm.Q:.4f}", " df =", nm.df_Q, " p =", f"{nm.pval_Q:.4f}")
print("Q 分解：异质性", f"{nm.Q_heterogeneity:.4f}", "+ 不一致性", f"{nm.Q_inconsistency:.4f}")
print("TE_random 形状:", np.shape(nm.TE_random), "（治疗 × 治疗矩阵）")
print("A vs B:", f"{float(nm.TE_random[0, 1]):.4f}",
      f"[{float(nm.lower_random[0, 1]):.4f}, {float(nm.upper_random[0, 1]):.4f}]")
```

```text
处理组: ['A', 'B', 'C']
比较数: 6  治疗数: 3
随机效应 tau^2 = 0.016798
总 Q = 7.6544  df = 4.0  p = 0.1051
Q 分解：异质性 7.5389 + 不一致性 0.1155
TE_random 形状: (3, 3) （治疗 × 治疗矩阵）
A vs B: 0.2866 [0.0739, 0.4993]
```

排名、联赛表、节点拆分：

```python
rk = ma.netrank(nm)
print("P-score（按治疗顺序）:", [round(float(v), 4) for v in rk.pscore_random])

lg = ma.netleague(nm)
print("league 治疗数:", len(lg.trts), "| 随机效应矩阵形状:", np.asarray(lg.random).shape)

sp = ma.netsplit(nm)
print("拆分比较:", list(sp.comparison))
print("compare_random 字段:", list(sp.compare_random))
print("直接/间接差异:", [round(float(v), 4) for v in sp.compare_random["TE"]])
```

```text
P-score（按治疗顺序）: [0.0763, 0.9707, 0.4531]
league 治疗数: 3 | 随机效应矩阵形状: (3, 3)
拆分比较: ['A:B', 'A:C', 'B:C']
compare_random 字段: ['TE', 'seTE', 'lower', 'upper', 'statistic', 'p', 'z']
直接/间接差异: [-0.0956, 0.0956, -0.0956]
```

## 10. 二分类网络的原始数据：`netmetabin()`

```python
nb = ma.netmetabin(
    event1=[12, 15, 9, 20], n1=[100, 100, 100, 140],
    event2=[20, 22, 14, 25], n2=[100, 100, 100, 150],
    treat1=["A", "B", "A", "C"], treat2=["B", "A", "C", "A"],
    studlab=["S1", "S2", "S3", "S4"], sm="OR",
)
print("方法:", nb.method, "| 设计数:", nb.d)
print("前两个比较的 log OR:", np.round(np.asarray(nb.TE, dtype=float)[:2], 4))
print("总 Q =", f"{nb.Q:.4f}", "p =", f"{nb.pval_Q:.4f}", "| I^2 =", f"{nb.I2:.2f}%")
```

```text
方法: MH | 设计数: 2
前两个比较的 log OR: [-0.6061  0.4689]
总 Q = nan p = nan | I^2 = nan%
```

## 11. 结果怎么取出来

三种粒度：单行摘要（`tidy`）、整体指标（`glance`）、研究级明细（`to_dataframe`）。

```python
t = ma.interop.tidy(r)
print("tidy 的类型:", type(t).__name__, "| 列:", list(t))
print(t.to_string(index=False))

g = ma.interop.glance(r)
print("\nglance 的键:", list(g)[:8], "...")
print("k =", int(g["k"].iloc[0]),
      "| tau2 =", round(float(g["tau2"].iloc[0]), 6),
      "| I2 =", round(float(g["I2"].iloc[0]), 2),
      "| QEp =", round(float(g["QEp"].iloc[0]), 6))

print("\nto_dataframe(r) 的列:", list(ma.to_dataframe(r)))
```

```text
tidy 的类型: DataFrame | 列: ['term', 'estimate', 'std.error', 'statistic', 'p.value', 'conf.low', 'conf.high']
   term  estimate  std.error  statistic  p.value  conf.low  conf.high
intrcpt  0.491362   0.123249   3.986743 0.000067  0.249799   0.732926

glance 的键: ['k', 'p', 'tau2', 'I2', 'H2', 'R2', 'QE', 'QEp'] ...
k = 10 | tau2 = 0.091675 | I2 = 61.23 | QEp = 0.006856

to_dataframe(r) 的列: ['yi', 'vi', 'X', 'weights', 'weights_pct', 'fitted', 'residuals', 'slab', 'not_na']
```

`to_table()` 默认给你一个 pandas `DataFrame`，也能直接要 `dict` / `records` / `polars` / `arrow`：

```python
tbl = r.to_table()
print("默认类型:", type(tbl).__name__, "| 形状:", getattr(tbl, "shape", None))
print("dict backend 的键:", list(r.to_table(backend="dict"))[:6])
print("Markdown 输出可用:", hasattr(tbl, "to_markdown"))
```

```text
默认类型: DataFrame | 形状: (10, 9)
dict backend 的键: ['yi', 'vi', 'X', 'weights', 'weights_pct', 'fitted']
Markdown 输出可用: True
```

## 12. 和 Python 生态系统对接

`pimeta` 不 import sklearn / statsmodels，但**实现了它们的协议**，所以能直接塞进 Pipeline：

```python
from sklearn.base import clone, is_regressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

est = ma.interop.MetaRegressor(method="REML")
X = np.asarray(dat["dose"], dtype=float).reshape(-1, 1)
w = 1.0 / np.asarray(vi)
est.fit(X, np.array(yi), sample_weight=w)

print("is_regressor:", is_regressor(est))
print("预测 dose=55:", round(float(est.predict(np.array([[55.0]]))[0]), 6))
print("clone 可用:", clone(est).get_params()["method"])

pipe = Pipeline([("scale", StandardScaler()),
                 ("meta", ma.interop.MetaRegressor(method="DL"))])
pipe.fit(X, np.array(yi), meta__sample_weight=w)
print("Pipeline 预测:", round(float(pipe.predict(np.array([[55.0]]))[0]), 6))
```

```text
is_regressor: True
预测 dose=55: 0.497301
clone 可用: REML
Pipeline 预测: 0.49666
```

导出成 xarray / networkx / statsmodels 结果对象：

```python
ds = ma.interop.to_xarray(nm)
print("xarray dims:", dict(ds.dims))

g = ma.interop.to_networkx(nm)
print("networkx:", g.number_of_nodes(), "节点", g.number_of_edges(), "条边")

sm_res = ma.interop.as_results(r)
print("statsmodels 风格:", ["params", "bse", "pvalues", "aic"])
print("  params =", sm_res.params.round(6), " aic =", round(sm_res.aic, 4))
```

```text
xarray dims: {'treat1': 3, 'treat2': 3, 'treatment': 3}
networkx: 3 节点 3 条边
statsmodels 风格: ['params', 'bse', 'pvalues', 'aic']
  params = [0.491362]  aic = 12.8138
```

## 13. 画图

所有绘图既能直接调（`ma.forest(...)` 返回 Matplotlib 的 `Axes`），也能走统一入口
`ma.plot(result, which=...)`；传 `filename=` 就顺便写盘。

```python
import tempfile
from pathlib import Path

out = Path(tempfile.mkdtemp())

ax = ma.plot(r, which="forest", filename=str(out / "forest.png"))
print("forest  ->", type(ax).__name__, "| 写盘:", (out / "forest.png").stat().st_size > 0)

for which in ("funnel", "baujat", "radial", "qqnorm"):
    obj = ma.plot(r, which=which, filename=str(out / f"{which}.png"))
    print(f"{which:7s} ->", type(obj).__name__,
          "| 写盘:", (out / f"{which}.png").stat().st_size > 0)

ax2 = ma.forest(yi, vi, studlab=dat["study"])
print("ma.forest 返回:", type(ax2).__name__,
      "| 画布尺寸:", tuple(float(v) for v in ax2.figure.get_size_inches()))
```

```text
forest  -> Axes | 写盘: True
funnel  -> Axes | 写盘: True
baujat  -> Axes | 写盘: True
radial  -> PlotResult | 写盘: True
qqnorm  -> PlotResult | 写盘: True
ma.forest 返回: Axes | 画布尺寸: (7.0, 6.5)
```

## 14. 命令行

装好之后有 `pimeta` 命令，适合放进 shell 脚本或 Makefile。

```bash
pimeta fit results.csv --yi yi --vi vi --method REML -o pooled.tsv
pimeta fit results.csv --yi yi --vi vi --mods "~ dose" --forest forest.png
pimeta escalc results.csv --measure OR --ai ai --bi bi --ci ci --di di
pimeta network net.csv --TE TE --seTE seTE --treat1 treat1 --treat2 treat2
```

```text
```

`-o` 除了写主表，还会在同一目录写一个 `<名字>_glance.tsv`（k、τ²、I²、Q、p 一行摘要）。
`python -m pimeta` 与 `pimeta` 等价，没装脚本时也能用。

## 15. 常见坑

| 现象 | 原因 | 处理 |
|---|---|---|
| 合并值看着不对 | `escalc` 对 OR/RR/IRR 返回对数尺度 | 用 `math.exp()` 还原 |
| `r.b` 打印出 `[0.29]` | `b` 永远是数组 | `float(r.b[0])` |
| `I2` 是 92.2 而不是 0.922 | 有意为之，和 R 一致 | 按百分数用 |
| `rma_mv(level=95)` 报错 | 这个参数要百分数 | `level=95`；`rma()` 才是 `0.95` |
| 传了 `slab=["Study 1", ...]` 却被当成列名 | 只有在 `data` 里确实存在同名列时才当列名 | 一般不会误判；必要时用 `strings_as_names=False` |
| `netmeta` 说比较被丢弃 | 有空值 | 加 `warn_missing=False` 静默，或先清洗数据 |
| 元回归系数名是 `V1` | 用的是数组接口 | 改用 `mods="~ dose", data=...` |
