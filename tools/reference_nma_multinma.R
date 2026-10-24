# Reference posterior for the arm-level Bayesian NMA, from multinma.
#
# Run from the repository root:
#
#   Rscript tools/reference_nma_multinma.R
#
# Writes tests/reference/nma_bayes_multinma.csv with the posterior mean, SD and
# a between-chain Monte Carlo standard error for every parameter of interest.
#
# The model is the aggregate-data arm-level NMA:
#
#   y[jk] ~ N(mu[j] + d[t_jk] - d[A] + delta[jk], se[jk]^2),
#   delta[jk] ~ N(0, tau^2) independent across arms except the study's
#   reference-treatment arm, which is pinned at zero.
#
# The priors are set explicitly so that they match what pimeta uses; leaving
# multinma's own defaults (prior_het = half_normal(5), prior_intercept =
# normal(100)) would compare two different models and call the difference a
# bug.  prior_het is half_normal(0.5) here because that is pimeta's default and
# the test should measure what ships.
#
# The Monte Carlo standard error is the SD of the four chain means divided by
# sqrt(4).  It is computed here rather than taken from an effective-sample-size
# estimate so that the tolerance in the test does not depend on either
# package's diagnostics agreeing about anything.
#
# adapt_delta is raised to 0.95 on purpose.  At Stan's default the first attempt
# at this reference produced divergent transitions and a posterior mean of d[B]
# 0.022 away from the exact quadrature answer for a three-study sub-network --
# a biased reference would have been read as a pimeta bug.  Divergences are now
# counted and reported rather than suppressed.

suppressMessages(library(multinma))

root <- if (file.exists("tests/data/nma_bayes.csv")) "." else ".."
d <- read.csv(file.path(root, "tests/data/nma_bayes.csv"), stringsAsFactors = FALSE)

net <- set_agd_arm(d, study = study, trt = treat, y = yi, se = sei, trt_ref = "A")

iter <- 10000
warmup <- 5000
chains <- 4
seed <- 20240617

fit <- nma(
  net,
  trt_effects = "random",
  prior_intercept = normal(0, 100),
  prior_trt = normal(0, 10),
  prior_het = half_normal(0.5),
  adapt_delta = 0.95,
  chains = chains, iter = iter, warmup = warmup, seed = seed, refresh = 0
)

n_divergent <- sum(rstan::get_divergent_iterations(fit$stanfit))
cat("divergent transitions:", n_divergent, "of", chains * iter, "\n")
if (n_divergent > 0) {
  warning("divergent transitions present: the reference posterior is biased")
}

draws <- as.matrix(fit)
studies <- sort(unique(d$study))
pars <- c(paste0("mu[", studies, "]"), "d[B]", "d[C]", "d[D]", "tau")
missing <- setdiff(pars, colnames(draws))
if (length(missing) > 0) {
  stop("multinma did not report: ", paste(missing, collapse = ", "))
}

# as.matrix() stacks chains, so split them back out for a between-chain MCSE
X <- draws[, pars, drop = FALSE]
n_per_chain <- nrow(X) / chains
chain_means <- sapply(seq_len(chains), function(k) {
  colMeans(X[((k - 1) * n_per_chain + 1):(k * n_per_chain), , drop = FALSE])
})

out <- data.frame(
  par = pars,
  mean = colMeans(X),
  sd = apply(X, 2, sd),
  mcse = apply(chain_means, 1, sd) / sqrt(chains),
  stringsAsFactors = FALSE
)

path <- file.path(root, "tests/reference/nma_bayes_multinma.csv")
write.csv(out, path, row.names = FALSE)

cat("multinma", as.character(packageVersion("multinma")), "\n")
cat("chains", chains, "iter", iter, "warmup", warmup, "seed", seed, "\n")
cat("wrote", path, "\n")
print(out, digits = 6)
