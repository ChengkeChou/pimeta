## Reference fit for the RoBMA ensemble, restricted to the three bias models
## that pimeta implements: no bias, PET, PEESE.
##
## Run inside the WSL reference environment:
##   wsl -d Ubuntu -e bash -lc 'cd /mnt/d/autometa/pimeta && Rscript tools/reference_robma.R'
## Output: tests/reference/robma_robma.csv (a posterior summary table).

suppressPackageStartupMessages(library(RoBMA))

d <- read.csv("tests/data/robma.csv")

set.seed(20261025)
fit <- RoBMA(
  yi = d$yi, sei = d$sei, ni = d$ni,
  measure = "SMD",
  ## The bias mixture always carries its own null component with weight one;
  ## prior_bias supplies the *alternatives* only. Adding prior_none() here
  ## inserts a second null and changes P(bias) from 1/3 to 3/5.
  prior_bias = list(
    prior_PET("cauchy", list(0, 1), prior_weights = 1 / 4),
    prior_PEESE("cauchy", list(0, 5), prior_weights = 1 / 4)
  ),
  sample = 10000, burnin = 4000, adapt = 1000,
  chains = 3, parallel = FALSE, silent = TRUE, seed = 20261025
)

cat("=== inclusion ===\n")
print(summary(fit)$inclusion_components)

cat("\n=== estimates ===\n")
est <- summary(fit)$estimates
print(est)

out <- data.frame(term = rownames(est), est, check.names = FALSE)
dir.create("tests/reference", showWarnings = FALSE, recursive = TRUE)
write.csv(out, "tests/reference/robma_robma.csv", row.names = FALSE)

inc <- summary(fit)$inclusion_components
write.csv(data.frame(component = rownames(inc), inc, check.names = FALSE),
          "tests/reference/robma_robma_inclusion.csv", row.names = FALSE)

cat("\n=== convergence ===\n")
print(max(est[, "R-hat"], na.rm = TRUE))
print(min(est[, "ESS"], na.rm = TRUE))
