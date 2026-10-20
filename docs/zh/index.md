# pimeta 中文说明

**pimeta 是 R 生态里 `metafor` + `netmeta` + `meta` 三个包的纯 Python 复现。**
不装 R、不用 rpy2、不依赖任何第三方 meta 分析库；数值引擎是用 NumPy/SciPy
直接转写的，结果与 R 逐位对齐（多数函数误差 < 1e-12）。

```bash
pip install pimeta
```

```python
import pimeta as ma
import math

yi = [0.20, 0.40, 0.10, 0.50]
vi = [0.04, 0.05, 0.03, 0.06]

r = ma.rma(yi, vi, method="REML")

print(r.b[0], r.ci_lb[0], r.ci_ub[0])   # 合并效应与 95% 区间
print(math.exp(r.b[0]))                 # 若为对数尺度，还原成比值
print(r.tau2, r.I2, r.QE, r.QEp)        # 异质性
```

## 结果对象的字段名和 R 一致

R 代码里的 `r$b`、`r$tau2`、`r$I2`、`r$ci.lb` 在 Python 里就是 `r.b`、
`r.tau2`、`r.I2`、`r.ci_lb`。已有的 R 脚本和教材可以近乎一对一翻译。

## 几个必须知道的约定

| 约定 | 说明 |
|---|---|
| `r.I2` 是**百分数** | `92.22` 而不是 `0.9222`；`H2`、`R2` 同理 |
| `confint(r)` 返回 **dict** | `{"tau2", "ci_lb", "ci_ub", "level"}`，是 τ² 的区间 |
| `r.b` 永远是**数组** | 即使只有截距也是长度 1 的 `np.ndarray`，取值用 `float(r.b[0])` |
| `escalc()` 对 RR/OR/IRR 返回**对数尺度** | 要报告比值本身请用 `math.exp()` |
| `rma_mv(level=95)` 用百分数 | `rma(level=0.95)` 用小数；这是 R 自己的不一致，我们照抄 |
| `rma_ls` 的 `tau2`/`I2`/`H2` 是**向量** | 位置-尺度模型里 τ² 逐研究不同 |
| `btt` / `att` 是 **1-based** | 截距是第 1 个系数 |
| 画图返回 matplotlib `Axes` | 加 `filename="x.png"` 才会落盘 |

## 用数据框和公式

所有吃数据的入口都支持 **frame + 列名**，`mods` 还支持 R 风格公式：

```python
import pimeta as ma

df = {
    "yi": [0.20, 0.40, 0.10, 0.50, 0.30, 0.15],
    "vi": [0.04, 0.05, 0.03, 0.06, 0.02, 0.05],
    "dose": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "arm": ["A", "B", "A", "B", "A", "B"],
    "slab": ["S1", "S2", "S3", "S4", "S5", "S6"],
}

r = ma.rma("yi", "vi", mods="~ dose + arm", slab="slab", data=df)
print(r.b)
```

支持的 frame：`dict`、pandas、polars、NumPy 结构化数组、记录字典列表。
公式支持 `a + b`、`a*b`、`a:b`、`0 + a` / `- 1`、`~ .`；字符串列按 R 的方式做
哑变量编码（丢弃第一层，列名 `arm[B]`）。

## 网状 meta 分析

```python
import numpy as np
import pimeta as ma

TE = np.array([0.10, -0.20, 0.30, 0.15, -0.05, 0.25])
seTE = np.array([0.10, 0.12, 0.11, 0.09, 0.13, 0.10])
treat1 = np.array(["A", "A", "B", "B", "C", "A"])
treat2 = np.array(["B", "C", "C", "D", "D", "D"])
studlab = np.array([f"S{i + 1}" for i in range(TE.size)])

n = ma.netmeta(TE, seTE, treat1, treat2, studlab, sm="MD")

print(n.TE_random)                    # 对比矩阵
print(ma.netrank(n).ranking_random)   # P-score（注意：字段名叫 ranking，内容是 P-score）
print(ma.netmeasures(n).proportion)   # 直接证据占比
ma.netgraph(n, filename="network.png")
```

## 命令行

安装后自带 `pimeta` 命令（`python -m pimeta` 等价），常见分析不用写脚本：

```bash
pimeta fit studies.csv --yi TE --vi seTE2 --method REML
pimeta fit studies.csv --yi TE --vi seTE2 --mods "~ dose + year" -o coef.csv
pimeta escalc raw.csv --measure RR --ai ai --bi bi --ci ci --di di
pimeta network nma.csv --te TE --se seTE --treat1 t1 --treat2 t2 --sm MD
```

输出默认 TSV 到 stdout，可用 `--format csv|json` 或 `-o file.csv`。
详见 [命令行指南](../guide/cli.md)。

## 缺失值

含缺失值的行会被剔除（等同 R 的 `na.action`），但 pimeta **会明确告知**：

```
pimeta: warning: dropped 2 of 13 studies with missing values in 'yi', 'vi'
        (rows 3, 7); pass warn_missing=False to silence this
```

Python 里同理：`rma(..., warn_missing=True)` 是默认；命令行用 `--no-warn-missing`
静默。`metagen()` 同样处理。

## 生态对接
协议层在 `pimeta.interop`，所有可选依赖都是懒加载：

```python
import pimeta as ma
from pimeta.interop import MetaRegressor, as_results

r = ma.rma(yi, vi)
r.tidy()      # 每个系数一行的 tidy 表
r.glance()    # 模型级统计量
r.augment()   # 每个研究一行 + 诊断量

print(as_results(r).conf_int())          # statsmodels 风格结果

est = MetaRegressor().fit(dose, yi, sample_weight=1 / vi)   # sklearn 协议
```

装了对应库之后，`Pipeline` / `GridSearchCV` / `cross_val_score` / `summary_col` /
`xarray` / `networkx` 都能直接用。

## 和 R 不一样的地方

完整列表见 [Differences from R](../reference/differences.md)。摘要：

- **命名**：R 的 `generic.class`（`print.rma.uni`）不是合法 Python 标识符，
  但 `getattr(ma, "cooks.distance.rma.mv")` 可直接解析；`x[i]` / `head` / `tail`
  在列表型结果上是原生的。
- **数值**：`escalc` 逐位相同；`rma` < 1e-12；`rma.mv` < 1e-6；
  `rma.ls` 的 α 差到 ~1e-5 —— 那是因为 metafor 提前停止，**我们的解更收敛**，
  对数似然恒 ≥ R。
- **故意不移植**：`metabias(plotit=)` 不画图、`pairwise(TE=, seTE=)` 未实现、
  `print.*` 不复刻 R 的多段式排版。

## 下一步

- [快速上手](../quickstart.md)
- [从 R 迁移](../migration-from-r.md)
- [API 参考](../api/index.md)
