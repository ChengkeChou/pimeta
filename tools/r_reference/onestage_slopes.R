#!/usr/bin/env Rscript
# Reference values for tests/test_onestage_slopes.py, from lme4.
#
#   Rscript tools/r_reference/onestage_slopes.R
#
# The one-stage IPD model with a random treatment slope *is* a linear mixed
# model, so lme4 is a reference for the likelihood rather than an analogy of it.
# The numbers pinned in the test came from lme4 2.0.6 under R 4.6.1; this script
# exists so they can be recomputed instead of quoted.

suppressPackageStartupMessages(library(lme4))

# Resolve the repository root from this script's own path, so the script works
# under Windows R and under WSL R without editing a drive letter.
this <- sub("^--file=", "",
            grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
root <- normalizePath(file.path(dirname(this), "..", ".."))
d <- read.csv(file.path(root, "tests", "data", "onestage_slopes.csv"))
d$study <- factor(d$study)
d$agec <- d$age - 55
d$treat <- as.numeric(d$treat)

f_int <- y ~ treat + agec + treat:agec + (1 | study)
f_un <- y ~ treat + agec + treat:agec + (1 + treat | study)
f_diag <- y ~ treat + agec + treat:agec + (1 | study) + (0 + treat | study)
ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

p <- function(x) sprintf("%.12f", as.numeric(x))

report <- function(tag, m) {
  cat("##", tag, "\n")
  cat("b   ", paste(p(fixef(m)), collapse = " "), "\n")
  cat("se  ", paste(p(sqrt(diag(as.matrix(vcov(m))))), collapse = " "), "\n")
  vc <- as.data.frame(VarCorr(m))
  for (i in seq_len(nrow(vc))) {
    cat("vc  ", as.character(vc$grp[i]), as.character(vc$var1[i]),
        as.character(vc$var2[i]), p(vc$vcov[i]), "\n")
  }
  cat("sigma2", p(sigma(m)^2), "\n")
  cat("loglik", sprintf("%.12f", as.numeric(logLik(m))), "\n")
  cat("deviance", sprintf("%.12f", deviance(m)), "\n")
}

report("intercept only (REML)", lmer(f_int, data = d, REML = TRUE,
                                     control = ctrl))
report("random slope UN (REML)", lmer(f_un, data = d, REML = TRUE,
                                      control = ctrl))
report("random slope DIAG (REML)", lmer(f_diag, data = d, REML = TRUE,
                                        control = ctrl))
report("random slope UN (ML)", lmer(f_un, data = d, REML = FALSE,
                                    control = ctrl))
cat("## versions\n")
cat("lme4", as.character(packageVersion("lme4")), "| R",
    as.character(getRversion()), "\n")
