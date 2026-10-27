# tau2(t) with a spline scale -- the Bayesian reference for
# pimeta's tau_trend(tau="spline"), on the bytes of tests/data/dynhet_spline.csv
#
# Fits the same model as pimeta, with the same basis: the restricted cubic
# spline's non-linear columns on centred time, projected off [1, t] and
# Gram-Schmidt'ed to B'B = I.  The transform (C for the projection, R for the
# triangular factor) is kept so the basis can be evaluated at times that are not
# in the data, exactly as pimeta's _SplineBasis does.
#
# Versions used: R 4.6.1, cmdstanr 0.9.0.9002, CmdStan 2.40.0.
#
#   wsl.exe -d Ubuntu -e bash -lc \
#     "cd /mnt/d/autometa/pimeta && Rscript tools/r_reference/tau2t_spline.R \
#      > /mnt/d/autometa/_tau2t_stan.log 2>&1"

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[[1]] else {
  f <- sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))
  if (length(f) == 0) "." else dirname(dirname(dirname(f)))
}
fixture <- file.path(root, "tests", "data", "dynhet_spline.csv")
stopifnot(file.exists(fixture), requireNamespace("cmdstanr", quietly = TRUE))

d <- read.csv(fixture)
k <- nrow(d)
y <- d$yi
v <- d$vi
centre <- mean(d$year)
tc <- d$year - centre

# --- Harrell's default knots on the centred time (nk = 4) ------------------
# pimeta.basis.default_knots uses the type-7 quantiles at 5/35/65/95 %
knots <- as.numeric(quantile(tc, c(0.05, 0.35, 0.65, 0.95)))
K <- length(knots)
denom <- (knots[K] - knots[1])^2

rcs <- function(x) {
  cols <- matrix(0, length(x), K - 2)
  for (j in seq_len(K - 2)) {
    hi <- (knots[K] - knots[j]) / (knots[K] - knots[K - 1])
    lo <- (knots[K - 1] - knots[j]) / (knots[K] - knots[K - 1])
    cols[, j] <- (pmax(x - knots[j], 0)^3 - hi * pmax(x - knots[K - 1], 0)^3 +
                    lo * pmax(x - knots[K], 0)^3) / denom
  }
  cols
}

U <- rcs(tc)
linear <- cbind(1, tc)
C <- qr.solve(linear, U)              # projection coefficients
resid <- U - linear %*% C
qr_res <- qr(resid)
R <- qr.R(qr_res)
B <- resid %*% solve(R)               # orthonormal bending columns
sgn <- ifelse(diag(R) < 0, -1, 1)
B <- sweep(B, 2, sgn, "*")
R <- diag(sgn) %*% R

bending_at <- function(time) {
  x <- time - centre
  (rcs(x) - cbind(1, x) %*% C) %*% solve(R)
}

cat("## basis checks\n")
cat("B'B\n"); print(crossprod(B), digits = 12)
cat("B'[1,t]", sprintf("%.10g", crossprod(B, linear)), "\n")
cat("knots", sprintf("%.10f", knots), "\n")
cat("B col norms", sprintf("%.10f", sqrt(colSums(B^2))), "\n")

mod <- cmdstanr::cmdstan_model(file.path(root, "tools", "r_reference",
                                         "tau2t_spline.stan"))
fit <- mod$sample(
  data = list(k = k, y = y, v = v, t = tc, m = ncol(B), B = B),
  chains = 4, iter_warmup = 2000, iter_sampling = 2000, seed = 1234,
  adapt_delta = 0.99, max_treedepth = 12,
  refresh = 0, show_messages = FALSE, show_exceptions = FALSE
)
s <- fit$summary()
cat("## posterior (mean sd q5 q95 rhat ess_bulk)\n")
for (nm in c("b0", "b1", "a0", "a1", "sigma_s")) {
  row <- s[s$variable == nm, ]
  cat(sprintf("%-8s %12.10f %12.10f %12.10f %12.10f %8.4f %8.1f\n", nm,
              row$mean, row$sd, row$q5, row$q95, row$rhat, row$ess_bulk))
}
for (j in seq_len(ncol(B))) {
  nm <- sprintf("gamma[%d]", j)
  row <- s[s$variable == nm, ]
  cat(sprintf("%-8s %12.10f %12.10f %12.10f %12.10f %8.4f %8.1f\n", nm,
              row$mean, row$sd, row$q5, row$q95, row$rhat, row$ess_bulk))
}

draws <- fit$draws(format = "matrix")
gm <- draws[, grep("^gamma\\[", colnames(draws)), drop = FALSE]
for (tt in c(0, min(tc), max(tc))) {
  Bt <- as.numeric(bending_at(tt + centre))
  tau2 <- exp(draws[, "a0"] + draws[, "a1"] * tt + as.numeric(gm %*% Bt))
  cat(sprintf("tau2(%8.4f) mean %12.10f q5 %12.10f q95 %12.10f\n",
              tt + centre, mean(tau2), quantile(tau2, 0.05),
              quantile(tau2, 0.95)))
}
cat(sprintf("## versions cmdstanr %s | CmdStan %s | R %s\n",
            as.character(packageVersion("cmdstanr")),
            as.character(cmdstanr::cmdstan_version()),
            as.character(getRversion())))
