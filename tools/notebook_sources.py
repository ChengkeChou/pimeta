"""Content of the example notebooks (executed by build_example_notebooks.py).

Cells are plain Python: no IPython magics, so the same source runs under the
builder, under ``tests/test_example_notebooks.py`` and in a real Jupyter kernel.
"""

from __future__ import annotations

#: The classic BCG vaccine trials (``metafor::dat.bcg``): 13 studies, raw 2x2
#: counts.  Every number these notebooks print matches R on the same data.
BCG = '''study = ["Aronson 1948", "Ferguson & Simes 1949", "Rosenthal et al 1960",
         "Hart & Sutherland 1977", "Frimodt-Moller et al 1973",
         "Stein & Aronson 1953", "Vandiviere et al 1973", "TPT Madras 1980",
         "Coetzee & Berjak 1968", "Rosenthal et al 1961", "Comstock et al 1974",
         "Comstock & Webster 1969", "Comstock et al 1976"]
year = [1948, 1949, 1960, 1977, 1973, 1953, 1973, 1980, 1968, 1961, 1974, 1969, 1976]
tpos = [4, 6, 3, 62, 33, 180, 8, 505, 29, 17, 186, 5, 27]
tneg = [119, 300, 228, 13536, 5036, 1361, 2537, 87886, 7470, 1699, 50448, 2493, 16886]
cpos = [11, 29, 11, 248, 47, 372, 10, 499, 45, 65, 141, 3, 29]
cneg = [128, 274, 209, 12619, 5761, 1079, 619, 87892, 7232, 1600, 27197, 2338, 17825]
print(f"{len(study)} 项试验，共 {sum(tpos) + sum(tneg) + sum(cpos) + sum(cneg):,} 名受试者")'''


_NB01 = [
    ("markdown", """# 1 · 第一次合并：BCG 疫苗数据

这个 notebook 用 metafor 里的经典 **BCG 疫苗试验**（13 项研究）走一遍完整流程：

> 2×2 计数 → 效应量 → 随机效应合并 → 异质性 → 预测区间 → 森林图 → 结果表

所有数字都能和 R 对上：`metafor::dat.bcg` + `escalc(measure="RR")` + `rma(yi, vi)`。
"""),
    ("code", """import math

import numpy as np

import pimeta as ma"""),
    ("markdown", "## 数据：13 项试验的原始 2×2 计数\n\n直接给出 `tpos/tneg/cpos/cneg`，让 `escalc()` 去做效应量换算 —— 手工算比值比很容易出错。"),
    ("code", BCG),
    ("markdown", "## 效应量\n\n`escalc()` 对**比值类**指标返回的是**对数尺度**：`yi` 是 log RR，要报告 RR 本身需要 `exp()`。"),
    ("code", """es = ma.escalc("RR", ai=tpos, bi=tneg, ci=cpos, di=cneg)

print("前 4 个 log RR :", np.round(es.yi[:4], 4))
print("还原成 RR      :", np.round(np.exp(es.yi[:4]), 4))
print("方差 vi        :", np.round(es.vi[:4], 4))"""),
    ("markdown", '## 合并\n\n默认就是随机效应 + REML 估计 τ²，与 `rma(yi, vi, method="REML")` 一致。'),
    ("code", """res = ma.rma(es.yi, es.vi, method="REML", slab=study)

print(f"k = {res.k},  tau^2 = {res.tau2:.5f},  I^2 = {res.I2:.2f}%")
print(f"Q = {res.QE:.3f} (df={res.k - res.p}, p={res.QEp:.4f})")
print()
print(f"合并 RR = {math.exp(float(res.b[0])):.4f} "
      f"[{math.exp(float(res.ci_lb[0])):.4f}, {math.exp(float(res.ci_ub[0])):.4f}]")"""),
    ("markdown", "## 森林图\n\n绘图函数返回 Matplotlib 的 `Axes`，在 notebook 里直接内联显示；也可以传 `filename=` 落盘。"),
    ("code", """ax = ma.forest(es.yi, es.vi, studlab=study)
_ = ax.set_title("BCG vaccine trials — RR (random effects, REML)")"""),
    ("markdown", "## 异质性与预测区间\n\n`predict()` 给的是**未来研究**的预测区间，通常比合并效应的置信区间宽得多 —— 这在报告里是个重要区别。"),
    ("code", """pred = ma.predict(res)
ci_tau2 = ma.confint(res)

print(f"合并效应 95% CI : [{pred.ci_lb:.4f}, {pred.ci_ub:.4f}]  (log scale)")
print(f"预测区间        : [{pred.pi_lb:.4f}, {pred.pi_ub:.4f}]  (log scale)")
print(f"tau^2 的 95% CI : [{ci_tau2['ci_lb']:.4f}, {ci_tau2['ci_ub']:.4f}]")"""),
    ("markdown", "## 结果表\n\n`tidy()` 给出 broom 风格的一行摘要（每行一个系数），`glance()` 给出模型级指标。"),
    ("code", """tidy = ma.interop.tidy(res)
glance = ma.interop.glance(res)

print(tidy.to_string(index=False))
print()
print(f"k = {int(glance['k'].iloc[0])}, AIC = {float(glance['AIC'].iloc[0]):.3f}, "
      f"BIC = {float(glance['BIC'].iloc[0]):.3f}")"""),
    ("markdown", """## 小结

- `escalc()` 算效应量（比值类返回**对数尺度**）
- `rma()` 合并，`tau2` / `I2` / `Q` 都在结果对象上
- `predict()` 是预测区间，不是置信区间
- `tidy()` / `glance()` 把结果摊平成表，能直接进 pandas

下一步：在 [`02-meta-regression.ipynb`](02-meta-regression.ipynb) 里用调节变量解释这份异质性。"""),
]

_NB02 = [
    ("markdown", """# 2 · 元回归与发表偏倚

同一个 BCG 数据集，这次问两个问题：

1. 试验**年代**能不能解释掉这部分异质性？（元回归 / 亚组）
2. 有没有发表偏倚的迹象？（Egger 检验、trim-and-fill、漏斗图、Baujat 图）
"""),
    ("code", """import math

import numpy as np

import pimeta as ma"""),
    ("code", BCG),
    ("code", """es = ma.escalc("RR", ai=tpos, bi=tneg, ci=cpos, di=cneg)

# 用列名 + 字典喂数据，公式接口写 mods，变量名不用抄错
dat = {"study": study, "year": year, "yi": es.yi, "vi": es.vi,
       "tpos": tpos, "tneg": tneg, "cpos": cpos, "cneg": cneg}

null = ma.rma("yi", "vi", method="REML", data=dat)
print(f"无调节变量：tau^2 = {null.tau2:.5f}, I^2 = {null.I2:.2f}%")"""),
    ("markdown", "## 元回归：`mods=\"~ year\"`\n\n返回的系数名沿用 R：`intrcpt` 是截距，`year` 是斜率。"),
    ("code", """reg = ma.rma("yi", "vi", mods="~ year", data=dat)

print("系数名 :", reg.coefnames)
print("截距   :", f"{float(reg.b[0]):.6f}")
print("year   :", f"{float(reg.b[1]):.6f}")
print(f"QM     : {reg.QM:.4f} (df={reg.QMdf[0]}, p={reg.QMp:.4f})")
print(f"tau^2  : {null.tau2:.5f} -> {reg.tau2:.5f}")"""),
    ("markdown", "## 亚组：按开始年代拆开\n\n分别合并两组，对比组内 τ² 和组间差异。"),
    ("code", """early = np.asarray(year) < 1970
yi_arr = np.asarray(es.yi, dtype=float)
vi_arr = np.asarray(es.vi, dtype=float)

fit_early = ma.rma(yi_arr[early], vi_arr[early], method="REML")
fit_late = ma.rma(yi_arr[~early], vi_arr[~early], method="REML")

for label, fit in (("<= 1969", fit_early), (">= 1970", fit_late)):
    rr = math.exp(float(fit.b[0]))
    lo = math.exp(float(fit.ci_lb[0]))
    hi = math.exp(float(fit.ci_ub[0]))
    print(f"{label:8s} k={fit.k}  RR={rr:.4f} [{lo:.4f}, {hi:.4f}]  "
          f"tau^2={fit.tau2:.5f}")"""),
    ("markdown", """## 发表偏倚

`regtest()` 是 Egger 的回归检验，`trimfill()` 是 trim-and-fill。传统上要求 **k ≥ 10**，
这里 k = 13，刚好够用。"""),
    ("code", """rt = ma.regtest(es.yi, es.vi)
tf = ma.trimfill(es.yi, es.vi)
mb = ma.metabias(null)

print(f"regtest   p = {rt.p:.4f}（斜率 {rt.slope:.4f}）")
print(f"trimfill  补齐 {tf.k0} 项研究，调整后效应 {tf.filled_effect:.4f}")
print(f"metabias  p = {mb.pval:.4g}（{mb.method_bias}）" if mb.pval is not None
      else f"metabias  k={mb.k} < {mb.k_min}，按惯例不报告")"""),
    ("markdown", "## 漏斗图"),
    ("code", """ax = ma.plot(null, which="funnel")
_ = ax.set_title("BCG trials — funnel plot")"""),
    ("markdown", "## Baujat 图：谁在贡献异质性，谁在拉偏结果\n\n横轴是每个研究对总体异质性的贡献，纵轴是它对合并效应的影响。"),
    ("code", """ax = ma.plot(null, which="baujat")
_ = ax.set_title("BCG trials — Baujat plot")"""),
    ("markdown", """## 怎么看这些结果

- 元回归里 `year` 的 p 值要结合 **QM 的自由度**（这里 1）一起读；
- 亚组的组内 τ² 都比总体小，说明年代确实解释了一部分异质性；
- Baujat 图右上角的研究同时「贡献异质性」又「影响结论」，值得单独检查。

下一步：[`03-network-meta-analysis.ipynb`](03-network-meta-analysis.ipynb) 换成网状结构。"""),
]

_NB03 = [
    ("markdown", """# 3 · 网络 meta 分析（NMA）

!!! note "示例数据"
    这里用一个小型的**教学用网络**（4 个干预、6 个直接比较），目的是演示 API，
    不是真实临床结论。把 `TE/seTE` 换成你自己的数据即可。

NMA 要回答的是：**A、B、C、D 两两之间谁更好**，即使某些对子从来没有被直接比较过。
"""),
    ("code", """import numpy as np

import pimeta as ma

# 6 个直接比较。比值类指标要先取对数，再作为 TE 传进去。
TE = [0.30, -0.20, 0.50, 0.10, -0.15, 0.25]
seTE = [0.12, 0.15, 0.18, 0.14, 0.11, 0.13]
treat1 = ["A", "B", "A", "C", "B", "C"]
treat2 = ["B", "A", "C", "A", "C", "B"]
studlab = ["S1", "S2", "S3", "S4", "S5", "S6"]

print(f"{len(TE)} 个比较，涉及 {len(set(treat1) | set(treat2))} 个干预")"""),
    ("markdown", "## 拟合网络\n\n`netmeta()` 默认同时给 frequentist 的固定效应与随机效应结果。"),
    ("code", """nm = ma.netmeta(TE, seTE, treat1, treat2, studlab, sm="RR")

print("干预       :", list(nm.trts))
print("比较数 m   :", nm.m)
print(f"tau^2      : {nm.tau2:.5f}")
print(f"总 Q       : {nm.Q:.4f} (df={nm.df_Q:g}, p={nm.pval_Q:.4f})")
print(f"  └ 异质性 : {nm.Q_heterogeneity:.4f}")
print(f"  └ 不一致 : {nm.Q_inconsistency:.4f}")"""),
    ("markdown", "## 相对效应是一个 n×n 矩阵\n\n和 R 的 `$TE.random` 一样，矩阵第 `[i, j]` 元就是干预 `j` 相对干预 `i` 的效应（对角线上是 `nan`）。"),
    ("code", """import pandas as pd

n = len(nm.trts)
mat = pd.DataFrame(np.asarray(nm.TE_random, dtype=float),
                   index=[f"vs {t}" for t in nm.trts], columns=nm.trts)
print(mat.round(4).to_string())"""),
    ("markdown", "## 排名：P-score"),
    ("code", """rk = ma.netrank(nm)

score = pd.Series(np.asarray(rk.pscore_random, dtype=float),
                  index=nm.trts).sort_values(ascending=False)
print("P-score（1 = 最好，0 = 最差）")
print(score.round(4).to_string())"""),
    ("markdown", "## 联赛表（league table）\n\n`netleague()` 返回的是 R 里那种**已经排好版的字符串矩阵**：对角线是干预名，\n**下三角是网络估计**，**上三角是直接比较**（metafor/netmeta 的约定）。"),
    ("code", """lg = ma.netleague(nm)
print("干预顺序:", list(lg.trts))
print()
print(pd.DataFrame(lg.random, index=nm.trts, columns=nm.trts).to_string())
print()
print(lg.details)"""),
    ("markdown", "## 一致性：直接证据 vs 间接证据\n\n每个比较拆成直接部分和间接部分，两者不一致说明网络内部有矛盾。"),
    ("code", """sp = ma.netsplit(nm)
print(pd.DataFrame({
    "比较": sp.comparison,
    "k": sp.k,
    "直接": np.round(sp.direct_random["TE"], 4),
    "间接": np.round(sp.indirect_random["TE"], 4),
}).to_string(index=False))"""),
    ("markdown", """## 排名图：P-score

（`netheat()` 需要 design-by-treatment 分解，`netmeta()` 目前还没算这个矩阵，
所以这里换成更直接的排名图；不一致性请看上面的 `netsplit()`。）"""),
    ("code", """ax = score.sort_values().plot.barh(color="#4c78a8")
ax.set_xlabel("P-score (higher is better)")
_ = ax.set_title("Treatment ranking — P-score")"""),
    ("markdown", """## 小结

| 想知道 | 用 |
|---|---|
| 两两相对效应 | `nm.TE_random[i, j]`（n×n 矩阵） |
| 谁最好 | `ma.netrank(nm).pscore_random` |
| 贴论文的联赛表 | `ma.netleague(nm).random` |
| 直接/间接是否一致 | `ma.netsplit(nm)` |
| 不一致性来源 | `ma.netheat(nm)` |

原始 2×2 数据可以用 `ma.netmetabin(event1, n1, event2, n2, treat1, treat2)` 直接喂，
它会在内部做 Mantel-Haenszel / 逆方差等合并。"""),
]

_NB04 = [
    ("markdown", """# 4 · 和 Python 生态一起用

`pimeta` 不 import pandas / scikit-learn / xarray / networkx —— 它只**实现它们的协议**。
所以你想用哪个就用哪个，不装也能跑。

这个 notebook 演示五件事：

1. pandas DataFrame 直接喂进 `rma()`
2. `tidy()` / `glance()` 的结果进 pandas
3. `rma()` 的结果塞进 scikit-learn 的 `Pipeline` 和 `GridSearchCV`
4. 网络 meta 结果导出成 xarray `Dataset` 和 networkx `Graph`
"""),
    ("code", """import numpy as np
import pandas as pd

import pimeta as ma"""),
    ("code", """study = ["Aronson 1948", "Ferguson & Simes 1949", "Rosenthal et al 1960",
         "Hart & Sutherland 1977", "Frimodt-Moller et al 1973",
         "Stein & Aronson 1953", "Vandiviere et al 1973", "TPT Madras 1980",
         "Coetzee & Berjak 1968", "Rosenthal et al 1961", "Comstock et al 1974",
         "Comstock & Webster 1969", "Comstock et al 1976"]
year = [1948, 1949, 1960, 1977, 1973, 1953, 1973, 1980, 1968, 1961, 1974, 1969, 1976]
tpos = [4, 6, 3, 62, 33, 180, 8, 505, 29, 17, 186, 5, 27]
tneg = [119, 300, 228, 13536, 5036, 1361, 2537, 87886, 7470, 1699, 50448, 2493, 16886]
cpos = [11, 29, 11, 248, 47, 372, 10, 499, 45, 65, 141, 3, 29]
cneg = [128, 274, 209, 12619, 5761, 1079, 619, 87892, 7232, 1600, 27197, 2338, 17825]

es = ma.escalc("RR", ai=tpos, bi=tneg, ci=cpos, di=cneg)
frame = pd.DataFrame({"study": study, "year": year, "yi": es.yi, "vi": es.vi})
frame.head()"""),
    ("markdown", "## 1–2 · DataFrame 进，DataFrame 出"),
    ("code", """res = ma.rma("yi", "vi", mods="~ year", data=frame)

print("系数名:", res.coefnames)
print(ma.interop.tidy(res).to_string(index=False))"""),
    ("code", """glance = ma.interop.glance(res)
row = pd.DataFrame({k: [v.iloc[0] if hasattr(v, "iloc") else v]
                    for k, v in glance.items()})
row[["k", "tau2", "I2", "QE", "QEp", "AIC"]].round(4)"""),
    ("markdown", """## 3 · 塞进 scikit-learn

`MetaRegressor` 实现了 `get_params` / `set_params` / `fit` / `predict` / `score`，
还带 `__sklearn_tags__`，所以 `is_regressor`、`clone`、`Pipeline`、`GridSearchCV`、
`cross_val_score` 全都能用。样本权重就是 `1/vi`（逆方差加权）。"""),
    ("code", """from sklearn.base import clone, is_regressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X = frame[["year"]].to_numpy(dtype=float)
y = frame["yi"].to_numpy(dtype=float)
w = 1.0 / frame["vi"].to_numpy(dtype=float)

est = ma.interop.MetaRegressor(method="REML")
print("is_regressor:", is_regressor(est))

pipe = Pipeline([("scale", StandardScaler()),
                 ("meta", ma.interop.MetaRegressor())])
grid = GridSearchCV(pipe, {"meta__method": ["DL", "REML", "PM"]},
                    cv=3, scoring="neg_mean_absolute_error")
grid.fit(X, y, meta__sample_weight=w)

print("最优参数:", grid.best_params_)
print("最优 MAE:", round(-grid.best_score_, 6))"""),
    ("markdown", "## 4 · 网络结果导出成 xarray / networkx\n\n（这里用一个 3 干预的小网络演示。）"),
    ("code", """nm = ma.netmeta(
    TE=[0.30, -0.20, 0.50, 0.10, -0.15, 0.25],
    seTE=[0.12, 0.15, 0.18, 0.14, 0.11, 0.13],
    treat1=["A", "B", "A", "C", "B", "C"],
    treat2=["B", "A", "C", "A", "C", "B"],
    studlab=["S1", "S2", "S3", "S4", "S5", "S6"], sm="RR",
)

ds = ma.interop.to_xarray(nm)
print(ds)

graph = ma.interop.to_networkx(nm)
print("节点:", list(graph.nodes))
print("边:", list(graph.edges(data=True))[:2])"""),
    ("markdown", """## 小结

| 你手上的东西 | 怎么接 |
|---|---|
| pandas / polars / arrow | 直接传给 `data=`，或用 `to_table(backend=...)` 拿走 |
| scikit-learn 工作流 | `ma.interop.MetaRegressor`（含 `__sklearn_tags__`） |
| statsmodels 报告 | `ma.interop.as_results(res)`，配 `summary_col` |
| xarray | `ma.interop.to_xarray(netmeta_result)` |
| networkx | `ma.interop.to_networkx(netmeta_result)` |

这些是**协议**不是适配器：没有装 pandas 也能用 pimeta，装了就能互操作。"""),
]


NOTEBOOKS: list[tuple[str, list[tuple[str, str]]]] = [
    ("01-quickstart.ipynb", _NB01),
    ("02-meta-regression.ipynb", _NB02),
    ("03-network-meta-analysis.ipynb", _NB03),
    ("04-ecosystem.ipynb", _NB04),
]
