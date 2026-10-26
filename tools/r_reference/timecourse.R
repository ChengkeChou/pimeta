# The reference numbers in tests/test_timecourse.py, recomputed from the fixture.
#
#   Rscript tools/r_reference/timecourse.R tests/data/timecourse.csv
#
# The model is pimeta's `timecourse` with a random intercept and a random linear
# slope (`struct = "GEN"`), moderators `~ 1 + tc` with `tc = time - 6`, and a
# within-study AR(1) covariance with `phi = 0.6` on the diagonal-free times.

suppressMessages(library(metafor))

args <- commandArgs(trailingOnly = TRUE)
path <- if (length(args) >= 1) args[[1]] else "tests/data/timecourse.csv"
d <- read.csv(path)

phi <- 0.6
d$tc <- d$time - 6

# Block-diagonal within-study covariance: se^2 * phi^|t_i - t_j|.
blocks <- lapply(split(d, d$study), function(s) {
  s <- s[order(s$time), ]
  dt <- abs(outer(s$time, s$time, "-"))
  s$se[1]^2 * phi^dt
})
V <- bdiag(blocks)
# `split` sorts study labels, so the rows must be ordered the same way.
d <- d[order(d$study, d$time), ]

res <- rma.mv(yi, V, mods = ~ tc, random = ~ 1 + tc | study,
              struct = "GEN", data = d, method = "REML", sparse = TRUE)

b <- as.numeric(res$b)
se <- as.numeric(res$se)
cat(sprintf("R_BETA = [%.10f, %.10f]\n", b[1], b[2]))
cat(sprintf("R_SE = [%.10f, %.10f]\n", se[1], se[2]))

tau2 <- as.numeric(res$tau2)
cat(sprintf("R_TAU2 = [%.10f, %.10f]\n", tau2[1], tau2[2]))
# `struct = "GEN"` reports the correlation of the random effects separately.
cat(sprintf("R_RHO = %.10f\n", as.numeric(res$rho)[1]))

# The omnibus test of the slope, which is QM with one degree of freedom.
cat(sprintf("R_TREND = %.10f\n", (b[2] / se[2])^2))

# The fitted curve at the three measured times.
t <- c(3, 6, 12)
est <- b[1] + b[2] * (t - 6)
vb <- as.matrix(res$vb)
se_curve <- sqrt(vb[1, 1] + (t - 6)^2 * vb[2, 2] + 2 * (t - 6) * vb[1, 2])
cat(sprintf("R_CURVE_EST = [%s]\n", paste(sprintf("%.10f", est), collapse = ", ")))
cat(sprintf("R_CURVE_SE = [%s]\n", paste(sprintf("%.10f", se_curve), collapse = ", ")))
