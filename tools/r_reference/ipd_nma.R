#!/usr/bin/env Rscript
# Reference values for tests/test_ipd_nma.py, from lme4.
#
#   Rscript tools/r_reference/ipd_nma.R
#
# The Gaussian one-stage IPD network model is a linear mixed model, so lme4 is a
# reference for the likelihood rather than an analogy of it:
#
#   y ~ 0 + study + trt + x + (0 + trt | study)
#
# with `study` a factor (one intercept per study) and `trt` a factor whose first
# level is the reference treatment.  The fixed treatment coefficients are the
# network's basic parameters and `(0 + trt | study)` is their between-study
# covariance, which is what the pinned `Sigma` in the test is.

suppressPackageStartupMessages(library(lme4))

this <- sub("^--file=", "",
            grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
root <- normalizePath(file.path(dirname(this), "..", ".."))
d <- read.csv(file.path(root, "tests", "data", "ipd_nma.csv"))

d$study <- factor(d$study)
d$trt <- factor(ifelse(d$dB > 0.5, "B", ifelse(d$dC > 0.5, "C", "A")))
cat("treatments:", paste(levels(d$trt), collapse = ", "), "\n")
cat("rows:", nrow(d), "| studies:", nlevels(d$study), "\n")

ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
m <- lmer(y ~ 0 + study + trt + x + (0 + trt | study), data = d,
          REML = TRUE, control = ctrl)

p <- function(x) sprintf("%.10f", as.numeric(x))
fe <- fixef(m)
se <- sqrt(diag(as.matrix(vcov(m))))
cat("## fixed effects (study intercepts omitted)\n")
for (nm in names(fe)) {
  if (startsWith(nm, "study")) next
  cat("b   ", nm, p(fe[nm]), p(se[nm]), "\n")
}
vc <- as.data.frame(VarCorr(m))
cat("## variance components\n")
for (i in seq_len(nrow(vc))) {
  cat("vc  ", as.character(vc$grp[i]), as.character(vc$var1[i]),
      as.character(vc$var2[i]), p(vc$vcov[i]), "\n")
}
# The random-effect block above is parameterised over (A, B, C).  The model in
# pimeta carries the *contrasts* against the reference treatment, so the
# between-study covariance the test pins is
#   Sigma = L' V L,  L = [B - A, C - A],
# and the two parameterisations are the same matrix, just re-expressed.
S <- as.matrix(VarCorr(m)$study)
lam <- cbind(c(-1, 1, 0), c(-1, 0, 1))
Sigma <- t(lam) %*% S %*% lam
cat("## Sigma in the contrast parameterisation (B-A, C-A)\n")
cat("Sigma", p(Sigma[1, 1]), p(Sigma[1, 2]), p(Sigma[2, 2]), "\n")
cat("sigma2", p(sigma(m)^2), "\n")
cat("loglik", sprintf("%.10f", as.numeric(logLik(m))), "\n")
cat("## versions\n")
cat("lme4", as.character(packageVersion("lme4")), "| R",
    as.character(getRversion()), "\n")
