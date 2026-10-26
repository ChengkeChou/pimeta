# The reference numbers in tests/test_prediction_joint.py, recomputed from the
# fixture. Run from the repository root:
#
#   Rscript tools/r_reference/prediction_joint.R tests/data/prediction_joint.csv
#
# The model is pimeta's `prediction_joint`: a bivariate random-effects model on
# (logit AUC, calibration slope), with the Hanley-McNeil variance for the AUC on
# the logit scale, a within-study correlation `rho`, and an unstructured
# between-study covariance (`struct = "UN"`).
#
# This script exists so the fixture can be *regenerated*: a synthetic fixture is
# only as good as its ability to be rebuilt, and rebuilding it invalidates every
# number that was pinned on the old bytes. Recomputing them from the same script
# is the difference between a regenerated fixture and a broken test.

suppressMessages(library(metafor))

args <- commandArgs(trailingOnly = TRUE)
path <- if (length(args) >= 1) args[[1]] else "tests/data/prediction_joint.csv"
d <- read.csv(path)

# Hanley-McNeil variance of the C-statistic, then the delta-method variance of
# its logit. This is the same expression pimeta::auc_logit evaluates.
hm_var <- function(A, n1, n0) {
  Q1 <- A / (2 - A)
  Q2 <- 2 * A^2 / (1 + A)
  (A * (1 - A) + (n1 - 1) * (Q1 - A^2) + (n0 - 1) * (Q2 - A^2)) / (n1 * n0)
}

rho <- 0.4
va_auc <- hm_var(d$auc, d$n_event, d$n_nonevent)
# pimeta reports the variance on the *logit* scale: delta method from the AUC
# scale, va / (A (1 - A))^2. Using the AUC-scale variance in V gives beta that is
# close but wrong in the third decimal, which is how this line was found.
va <- va_auc / (d$auc * (1 - d$auc))^2
y1 <- qlogis(d$auc)
y2 <- d$slope
vs <- d$slope_se^2
n <- nrow(d)

cat(sprintf("R_STUDY1_LOGIT = %.10f\n", y1[1]))
cat(sprintf("R_STUDY1_VAR = %.10f\n", va[1]))

yi <- as.numeric(rbind(y1, y2))     # interleaved: (auc1, slope1, auc2, slope2, ...)
V <- bdiag(lapply(seq_len(n), function(i) {
  cov12 <- rho * sqrt(va[i]) * sqrt(vs[i])
  matrix(c(va[i], cov12, cov12, vs[i]), nrow = 2, ncol = 2)
}))
dat <- data.frame(study = rep(seq_len(n), each = 2),
                  metric = factor(rep(c("auc", "slope"), times = n)))
res <- rma.mv(yi, V, mods = ~ metric - 1, random = ~ metric | study,
              struct = "UN", data = dat, method = "REML", sparse = TRUE)

b <- as.numeric(res$b)
se <- as.numeric(res$se)
cat(sprintf("R_RHO40_BETA = [%.10f, %.10f]\n", b[1], b[2]))
cat(sprintf("R_RHO40_SE = [%.10f, %.10f]\n", se[1], se[2]))
tau2_mat <- if (is.matrix(res$tau2)) res$tau2 else diag(as.numeric(res$tau2), 2)
cat(sprintf("R_RHO40_TAU2 = [%.10f, %.10f]\n", tau2_mat[1, 1], tau2_mat[2, 2]))
rho_between <- if (is.matrix(res$tau2)) {
  res$tau2[1, 2] / sqrt(res$tau2[1, 1] * res$tau2[2, 2])
} else {
  as.numeric(res$rho)[1]
}
cat(sprintf("R_RHO40_SIGMA_RHO = %.10f\n", rho_between))
cat(sprintf("R_RHO40_AUC = %.10f\n", plogis(b[1])))
cat(sprintf("R_RHO40_SLOPE_Z = %.10f\n", (b[2] - 1) / se[2]))

# Joint test of (mu_auc = 0, mu_slope = 1) against the full coefficient
# covariance: the delta-free quadratic form the package reports as test_both().
diff <- c(b[1] - 0, b[2] - 1)
joint <- as.numeric(t(diff) %*% solve(res$vb) %*% diff)
cat(sprintf("R_RHO40_JOINT = %.10f\n", joint))

# The rho = 0 fit: zero *within*-study covariance, but the default `struct="UN"`
# still estimates the between-study correlation. That is the test's point --
# zeroing the assumption is not the same as removing the parameter -- so the
# between-study structure must stay unstructured here.
V0 <- bdiag(lapply(seq_len(n), function(i) {
  matrix(c(va[i], 0, 0, vs[i]), nrow = 2, ncol = 2)
}))
res0 <- rma.mv(yi, V0, mods = ~ metric - 1, random = ~ metric | study,
               struct = "UN", data = dat, method = "REML", sparse = TRUE)
b0 <- as.numeric(res0$b)
cat(sprintf("R_RHO0_BETA = [%.10f, %.10f]\n", b0[1], b0[2]))
cat(sprintf("R_RHO0_TAU2 = [%.10f, %.10f]\n", as.numeric(res0$tau2)[1],
            as.numeric(res0$tau2)[2]))

# The two marginal pools, which the DIAG fit must reproduce.
m1 <- rma(y1, vi = va, method = "REML")
m2 <- rma(y2, vi = vs, method = "REML")
cat(sprintf("R_SEPARATE = auc (%.10f, %.10f, %.10f) slope (%.10f, %.10f, %.10f)\n",
            as.numeric(m1$b), as.numeric(m1$se), m1$tau2,
            as.numeric(m2$b), as.numeric(m2$se), m2$tau2))
