## Rung 1: a selection model with a FIXED step weight function.
##
## With omega fixed at (1, 0.5) the only free parameters are mu and tau, so the
## comparison against pimeta isolates the selection kernel: there is no Dirichlet
## prior and no weight parameterisation to disagree about.
##
##   wsl -d Ubuntu -e bash -lc 'cd /mnt/d/autometa/pimeta && Rscript tools/reference_robma_bsel.R'

suppressPackageStartupMessages(library(RoBMA))

d <- read.csv("tests/data/robma.csv")

fit_one <- function(side, steps, weights, seed) {
  bselmodel(
    yi = d$yi, sei = d$sei, ni = d$ni, measure = "SMD",
    prior_bias = prior_weightfunction(side, steps = steps, weights = weights),
    sample = 10000, burnin = 4000, adapt = 1000,
    chains = 3, parallel = FALSE, silent = TRUE, seed = seed
  )
}

cat("=== rung 1a: one-sided, steps 0.05, omega fixed at (1, 0.5) ===\n")
f1 <- fit_one("one-sided", 0.05, wf_fixed(c(1, 0.5)), 101)
print(summary(f1)$estimates)
cat("\nweight function actually used:\n")
print(f1$priors$outcome$bias[[1]])

cat("\n=== rung 1b: two-sided, steps 0.05, omega fixed at (1, 0.5) ===\n")
f2 <- fit_one("two-sided", 0.05, wf_fixed(c(1, 0.5)), 102)
print(summary(f2)$estimates)

out1 <- data.frame(term = rownames(summary(f1)$estimates),
                   summary(f1)$estimates, check.names = FALSE)
out2 <- data.frame(term = rownames(summary(f2)$estimates),
                   summary(f2)$estimates, check.names = FALSE)
dir.create("tests/reference", showWarnings = FALSE, recursive = TRUE)
write.csv(out1, "tests/reference/robma_bsel_onesided.csv", row.names = FALSE)
write.csv(out2, "tests/reference/robma_bsel_twosided.csv", row.names = FALSE)
cat("\nwrote tests/reference/robma_bsel_onesided.csv and _twosided.csv\n")
