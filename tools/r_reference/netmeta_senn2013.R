#!/usr/bin/env Rscript
# R reference values for the Senn2013 network, used by
# tests/test_r_parity_senn2013.py.  Run with:
#
#   Rscript tools/r_reference/netmeta_senn2013.R
#
# Requires netmeta (3.7-0 was used) and writes tests/data/_r_senn2013.tsv.

suppressPackageStartupMessages(library(netmeta))

root <- "D:/autometa/pimeta"
data(Senn2013, package = "netmeta")

net <- netmeta(TE = Senn2013$TE, seTE = Senn2013$seTE,
               treat1 = Senn2013$treat1, treat2 = Senn2013$treat2,
               studlab = Senn2013$studlab, sm = "MD",
               reference.group = "plac", method.tau = "REML")

con <- file(file.path(root, "tests", "data", "_r_senn2013.tsv"), open = "wt")

fmt <- function(x) sprintf("%.17g", as.numeric(x))

emit1 <- function(key, value) {
  cat(key, "\t", fmt(value), "\n", sep = "", file = con)
}

emitv <- function(prefix, values, labels) {
  for (i in seq_along(values)) {
    cat(prefix, "|", labels[i], "\t", fmt(values[i]), "\n",
        sep = "", file = con)
  }
}

cat("# netmeta", as.character(packageVersion("netmeta")),
    "| metafor", as.character(packageVersion("metafor")),
    "| R", as.character(getRversion()), "\n", sep = " ", file = con)

emit1("core.k", net$k)
emit1("core.m", net$m)
emit1("core.n", net$n)
emit1("core.Q", net$Q)
emit1("core.df.Q", net$df.Q)
emit1("core.pval.Q", net$pval.Q)
emit1("core.tau2", net$tau2)
emit1("core.I2", net$I2)

cat("trts\t", paste(net$trts, collapse = ","), "\n", sep = "", file = con)

emitv("TE.random", net$TE.random, outer(net$trts, net$trts, paste, sep = ":"))
emitv("seTE.random", net$seTE.random, outer(net$trts, net$trts, paste, sep = ":"))
emitv("TE.common", net$TE.common, outer(net$trts, net$trts, paste, sep = ":"))
emitv("seTE.common", net$seTE.common, outer(net$trts, net$trts, paste, sep = ":"))

dd <- decomp.design(net)
for (row in rownames(dd$Q.decomp)) {
  emit1(paste0("decomp.Q.", row, ".Q"), dd$Q.decomp[row, "Q"])
  emit1(paste0("decomp.Q.", row, ".df"), dd$Q.decomp[row, "df"])
  emit1(paste0("decomp.Q.", row, ".pval"), dd$Q.decomp[row, "pval"])
}
emit1("decomp.Q.inc.random.Q", dd$Q.inc.random$Q)
emit1("decomp.Q.inc.random.df", dd$Q.inc.random$df)
emit1("decomp.Q.inc.random.pval", dd$Q.inc.random$pval)
emit1("decomp.Q.inc.random.tau.within", dd$Q.inc.random$tau.within)

emitv("decomp.Q.het.design.Q", dd$Q.het.design$Q, dd$Q.het.design$design)
emitv("decomp.Q.het.design.df", dd$Q.het.design$df, dd$Q.het.design$design)
emitv("decomp.Q.inc.detach.Q", dd$Q.inc.detach$Q, dd$Q.inc.detach$design)
emitv("decomp.Q.inc.detach.df", dd$Q.inc.detach$df, dd$Q.inc.detach$design)
emitv("decomp.Q.inc.design", dd$Q.inc.design,
      paste0(seq_along(dd$Q.inc.design), ":", names(dd$Q.inc.design)))

nr <- netrank(net, small.values = "desirable")
emitv("netrank.random", as.numeric(nr$ranking.random), names(nr$ranking.random))
emitv("netrank.common", as.numeric(nr$ranking.common), names(nr$ranking.common))

nms <- netmeasures(net)
emitv("netmeasures.proportion", nms$proportion, names(nms$proportion))
emitv("netmeasures.meanpath", nms$meanpath, names(nms$meanpath))
emitv("netmeasures.minpar", nms$minpar, names(nms$minpar))

ns <- netsplit(net)
emitv("netsplit.TE.direct.random", ns$direct.random$TE, ns$comparison)
emitv("netsplit.seTE.direct.random", ns$direct.random$seTE, ns$comparison)
emitv("netsplit.TE.indirect.random", ns$indirect.random$TE, ns$comparison)
emitv("netsplit.seTE.indirect.random", ns$indirect.random$seTE, ns$comparison)
emitv("netsplit.TE.random", ns$compare.random$TE, ns$comparison)
emitv("netsplit.prop.random", ns$prop.random, ns$comparison)

close(con)
cat("wrote tests/data/_r_senn2013.tsv\n")
