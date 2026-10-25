## Rung 2: a selection model with FREE weights on the reference's cumulative
## Dirichlet prior, so the comparison tests the kernel and the prior together.
##
##   wsl -d Ubuntu -e bash -lc 'cd /mnt/d/autometa/pimeta && Rscript tools/reference_robma_free.R'

suppressPackageStartupMessages(library(RoBMA))

d <- read.csv("tests/data/robma.csv")

fit <- bselmodel(
  yi = d$yi, sei = d$sei, ni = d$ni, measure = "SMD",
  prior_bias = prior_weightfunction("one-sided", steps = 0.05),
  sample = 20000, burnin = 8000, adapt = 2000,
  chains = 4, parallel = FALSE, silent = TRUE, seed = 202
)

cat("=== one-sided, steps 0.05, free weights ===\n")
print(summary(fit)$estimates)
cat("\n--- bias prior ---\n")
print(fit$priors$outcome$bias[[1]])
cat("\n--- omega posterior ---\n")
print(summary(fit)$estimates_bias)

est <- summary(fit)$estimates
write.csv(data.frame(term = rownames(est), est, check.names = FALSE),
          "tests/reference/robma_bsel_free.csv", row.names = FALSE)
cat("\nwrote tests/reference/robma_bsel_free.csv\n")
