## Null check: omega = (1, 1) is no selection at all, so the selection model
## collapses to the plain random-effects model. If pimeta agrees with the
## reference here, the base model is shared and the difference at (1, 0.5) is in
## how the weights enter. If it does not, the difference is upstream of them.

suppressPackageStartupMessages(library(RoBMA))

d <- read.csv("tests/data/robma.csv")

fit <- bselmodel(
  yi = d$yi, sei = d$sei, ni = d$ni, measure = "SMD",
  prior_bias = prior_weightfunction("one-sided", steps = 0.05,
                                    weights = wf_fixed(c(1, 1))),
  sample = 20000, burnin = 8000, adapt = 2000,
  chains = 4, parallel = FALSE, silent = TRUE, seed = 303
)
cat("=== one-sided, omega fixed at (1, 1) ===\n")
print(summary(fit)$estimates)

## and the plain model for comparison, with no bias component at all
fit2 <- brma(yi = d$yi, sei = d$sei, ni = d$ni, measure = "SMD",
             sample = 20000, burnin = 8000, adapt = 2000,
             chains = 4, parallel = FALSE, silent = TRUE, seed = 304)
cat("\n=== brma() with no bias component ===\n")
print(summary(fit2)$estimates)
