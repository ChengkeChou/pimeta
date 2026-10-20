# R method-name map

R writes its methods as `generic.class` (`print.rma.uni`,
`cooks.distance.rma.mv`, `[.list.rma`).  Those strings are not valid
Python identifiers, so pimeta exposes R-independent names and bridges the
two with `pimeta.r_methods()` plus a module-level `__getattr__`:

```python
import pimeta as ma
getattr(ma, "cooks.distance.rma.mv")   # -> ma.cooks_distance
getattr(ma, "[.list.rma")              # -> ma.slice_rma
```

66 entries:

| R | pimeta |
|---|---|
| `$<-.escalc` | [`update_rma`](../api/index.md#update_rma) |
| `$<-.list.rma` | [`update_rma`](../api/index.md#update_rma) |
| `[.escalc` | [`slice_rma`](../api/index.md#slice_rma) |
| `[.list.rma` | [`slice_rma`](../api/index.md#slice_rma) |
| `anova.rma.mv` | [`anova_rma`](../api/index.md#anova_rma) |
| `anova.rma.uni` | [`anova_rma`](../api/index.md#anova_rma) |
| `as.data.frame.confint.rma` | [`as_data_frame_confint`](../api/index.md#as_data_frame_confint) |
| `as.data.frame.list.anova.rma` | [`as_data_frame_list_anova`](../api/index.md#as_data_frame_list_anova) |
| `as.data.frame.list.confint.rma` | [`as_data_frame_list_confint`](../api/index.md#as_data_frame_list_confint) |
| `as.data.frame.list.rma` | [`as_data_frame_list_rma`](../api/index.md#as_data_frame_list_rma) |
| `as.data.frame.vif.rma` | [`as_data_frame_vif`](../api/index.md#as_data_frame_vif) |
| `blup.rma.uni` | [`blup`](../api/index.md#blup) |
| `confint.rma.ls` | [`confint_rma_ls`](../api/index.md#confint_rma_ls) |
| `confint.rma.mh` | [`confint`](../api/index.md#confint) |
| `confint.rma.mv` | [`confint`](../api/index.md#confint) |
| `confint.rma.peto` | [`confint`](../api/index.md#confint) |
| `confint.rma.uni` | [`confint`](../api/index.md#confint) |
| `cooks.distance.rma.mv` | [`cooks_distance`](../api/index.md#cooks_distance) |
| `hc.rma.uni` | [`hc_rma`](../api/index.md#hc_rma) |
| `head.list.rma` | [`head_rma`](../api/index.md#head_rma) |
| `permutest.rma.ls` | [`permutest`](../api/index.md#permutest) |
| `permutest.rma.uni` | [`permutest`](../api/index.md#permutest) |
| `predict.rma.ls` | [`predict_rma`](../api/index.md#predict_rma) |
| `predict.rma.uni` | [`predict_rma`](../api/index.md#predict_rma) |
| `print.anova.rma` | [`print_summary`](../api/index.md#print_summary) |
| `print.confint.matreg` | [`as_data_frame_confint`](../api/index.md#as_data_frame_confint) |
| `print.confint.rma` | [`as_data_frame_confint`](../api/index.md#as_data_frame_confint) |
| `print.deltamethod` | [`print_summary`](../api/index.md#print_summary) |
| `print.escalc` | [`print_summary`](../api/index.md#print_summary) |
| `print.fsn` | [`print_summary`](../api/index.md#print_summary) |
| `print.gosh.rma` | [`print_summary`](../api/index.md#print_summary) |
| `print.hc.rma.uni` | [`print_summary`](../api/index.md#print_summary) |
| `print.hettest` | [`print_summary`](../api/index.md#print_summary) |
| `print.infl.rma.uni` | [`print_summary`](../api/index.md#print_summary) |
| `print.list.anova.rma` | [`as_data_frame_list_anova`](../api/index.md#as_data_frame_list_anova) |
| `print.list.confint.rma` | [`as_data_frame_list_confint`](../api/index.md#as_data_frame_list_confint) |
| `print.list.rma` | [`print_summary`](../api/index.md#print_summary) |
| `print.matreg` | [`summary`](../api/index.md#summary) |
| `print.permutest.rma.uni` | [`print_summary`](../api/index.md#print_summary) |
| `print.profile.rma` | [`print_summary`](../api/index.md#print_summary) |
| `print.ranktest` | [`print_summary`](../api/index.md#print_summary) |
| `print.regtest` | [`print_summary`](../api/index.md#print_summary) |
| `print.rma.glmm` | [`summary`](../api/index.md#summary) |
| `print.rma.ls` | [`summary`](../api/index.md#summary) |
| `print.rma.mh` | [`summary`](../api/index.md#summary) |
| `print.rma.mv` | [`summary`](../api/index.md#summary) |
| `print.rma.peto` | [`summary`](../api/index.md#summary) |
| `print.rma.uni` | [`summary`](../api/index.md#summary) |
| `print.summary.rma` | [`print_summary`](../api/index.md#print_summary) |
| `print.tes` | [`print_summary`](../api/index.md#print_summary) |
| `print.vcovmat` | [`print_summary`](../api/index.md#print_summary) |
| `print.vif.rma` | [`as_data_frame_vif`](../api/index.md#as_data_frame_vif) |
| `profile.rma.ls` | [`profile_rma_ls`](../api/index.md#profile_rma_ls) |
| `profile.rma.mv` | [`profile_rmamv`](../api/index.md#profile_rmamv) |
| `profile.rma.uni` | [`profile_rma`](../api/index.md#profile_rma) |
| `profile.rma.uni.selmodel` | [`profile_selmodel`](../api/index.md#profile_selmodel) |
| `qqnorm.rma.glmm` | [`qqnorm_rma`](../api/index.md#qqnorm_rma) |
| `qqnorm.rma.mh` | [`qqnorm_rma`](../api/index.md#qqnorm_rma) |
| `qqnorm.rma.mv` | [`qqnorm_rma`](../api/index.md#qqnorm_rma) |
| `qqnorm.rma.peto` | [`qqnorm_rma`](../api/index.md#qqnorm_rma) |
| `qqnorm.rma.uni` | [`qqnorm_rma`](../api/index.md#qqnorm_rma) |
| `ranef.rma.uni` | [`ranef`](../api/index.md#ranef) |
| `ranktest.rma` | [`begg_test`](../api/index.md#begg_test) |
| `regtest.rma` | [`regtest`](../api/index.md#regtest) |
| `robust.rma.mv` | [`robust`](../api/index.md#robust) |
| `tail.list.rma` | [`tail_rma`](../api/index.md#tail_rma) |
