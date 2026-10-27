#!/usr/bin/env Rscript
# Reference values for tests/test_tvhr_joint.py, from glm and metafor.
#
#   Rscript tools/r_reference/tvhr_piecewise.R
#
# The pipeline is three steps and every one of them is mirrored here rather than
# re-invented:
#
# 1. split each participant's follow-up at the break points, recording the
#    exposure and whether the event fell inside the interval (pimeta's
#    `split_followup`);
# 2. fit each study by Poisson regression with the interval as a stratifying
#    factor and an interval-by-treatment interaction, offset log(exposure), and
#    read the log hazard ratios and their covariance out of the interaction
#    block (pimeta's `_fit_study_blocks`).  The design matrix is built here
#    explicitly so the R and Python designs cannot drift apart through a
#    formula's contrast coding;
# 3. pool with `metafor::rma.mv`, `mods = ~ interval - 1`,
#    `random = ~ interval | study`, `struct = "UN"` (and `"DIAG"`).

suppressPackageStartupMessages(library(metafor))

this <- sub("^--file=", "",
            grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
root <- normalizePath(file.path(dirname(this), "..", ".."))
d <- read.csv(file.path(root, "tests", "data", "tvhr_piecewise.csv"))
breaks <- c(0, 6, 12, 24)
k_int <- length(breaks) - 1

split_followup <- function(time, event) {
  cap <- length(time) * k_int
  rows <- integer(cap); ints <- integer(cap)
  expo <- numeric(cap); ev <- integer(cap)
  n <- 0L
  for (i in seq_along(time)) {
    t_end <- time[i]
    hit <- event[i] == 1
    for (j in seq_len(k_int)) {
      lo <- breaks[j]; hi <- breaks[j + 1]
      if (t_end <= lo) break
      n <- n + 1L
      rows[n] <- i; ints[n] <- j
      expo[n] <- min(t_end, hi) - lo
      ev[n] <- if (hit && t_end <= hi) 1L else 0L
    }
  }
  keep <- seq_len(n)
  list(rows = rows[keep], ints = ints[keep], exposure = expo[keep],
       events = ev[keep])
}

studies <- sort(unique(d$study))
yi <- list(); Vi <- list(); labels <- list(); study_of <- list()
for (s in studies) {
  sub <- d[d$study == s, ]
  sp <- split_followup(sub$time, sub$event)
  treat <- sub$treat[sp$rows]
  codes <- sp$ints
  X <- matrix(0, length(codes), 2L * k_int)
  X[cbind(seq_along(codes), codes)] <- 1
  X[cbind(seq_along(codes), k_int + codes)] <- treat
  fit <- glm.fit(X, sp$events, family = poisson(),
                 offset = log(sp$exposure),
                 control = glm.control(epsilon = 1e-14, maxit = 100))
  if (!fit$converged) stop("study ", s, " did not converge")
  # Poisson dispersion is 1, so the covariance is the inverse of X'WX with the
  # IRLS weights of the last iteration
  XtWX <- crossprod(X * sqrt(fit$weights))
  V <- chol2inv(chol(XtWX))
  cols <- (k_int + 1L):(2L * k_int)
  yi[[length(yi) + 1L]] <- fit$coefficients[cols]
  Vi[[length(Vi) + 1L]] <- V[cols, cols]
  study_of[[length(study_of) + 1L]] <- rep(s, k_int)
}
cat("## study 1\n")
cat("log_hr", sprintf("%.10f", yi[[1]]), "\n")
cat("var   ", sprintf("%.10f", diag(Vi[[1]])), "\n")
cat("offdiag", sprintf("%.10f", Vi[[1]][1, 2]), "\n")

Y <- unlist(yi)
V <- Matrix::bdiag(Vi)
data <- data.frame(y = Y,
                   interval = factor(rep(seq_len(k_int), length(studies)),
                                     levels = seq_len(k_int)),
                   study = factor(unlist(study_of)))

pooled <- function(struct) {
  rma.mv(y ~ interval - 1, V = as.matrix(V), data = data,
         random = ~ interval | study, struct = struct, method = "REML")
}

diag_fit <- pooled("DIAG")
cat("## struct = DIAG\n")
cat("beta  ", sprintf("%.10f", as.numeric(diag_fit$beta)), "\n")
cat("se    ", sprintf("%.10f", as.numeric(diag_fit$se)), "\n")
cat("tau2  ", sprintf("%.10f", diag(as.matrix(diag_fit$G))), "\n")
cat("loglik", sprintf("%.10f", as.numeric(diag_fit$ll)), "\n")

un_fit <- pooled("UN")
cat("## struct = UN\n")
cat("beta  ", sprintf("%.10f", as.numeric(un_fit$beta)), "\n")
cat("se    ", sprintf("%.10f", as.numeric(un_fit$se)), "\n")
cat("tau2  ", sprintf("%.10f", diag(as.matrix(un_fit$G))), "\n")
sd <- sqrt(diag(as.matrix(un_fit$G)))
G <- as.matrix(un_fit$G)
cat("rho   ", sprintf("%.10f", c(G[2, 1], G[3, 1], G[3, 2]) /
                        c(sd[2] * sd[1], sd[3] * sd[1], sd[3] * sd[2])), "\n")
w <- c(-1, 0, 1)
b <- as.numeric(un_fit$beta)
vb <- as.matrix(un_fit$vb)
cat("linear est", sprintf("%.10f", sum(w * b)),
    " se", sprintf("%.10f", sqrt(as.numeric(t(w) %*% vb %*% w))), "\n")
cat("## versions\n")
cat("metafor", as.character(packageVersion("metafor")), "| R",
    as.character(getRversion()), "\n")

