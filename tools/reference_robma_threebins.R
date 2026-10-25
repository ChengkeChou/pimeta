## Three bins, where orientation becomes identifiable.
##
## With two bins there are only two weight configurations, so matching one of
## them says nothing about which end of the p-value scale the vector starts at.
## With three bins the two ends are distinguishable, and an asymmetric fixed
## vector reads off the order directly.

suppressPackageStartupMessages(library(RoBMA))

d <- read.csv("tests/data/robma.csv")

fit <- function(pb, seed) {
  bselmodel(yi = d$yi, sei = d$sei, ni = d$ni, measure = "SMD",
            prior_bias = pb, sample = 20000, burnin = 8000, adapt = 2000,
            chains = 4, parallel = FALSE, silent = TRUE, seed = seed)
}

show <- function(label, f) {
  est <- summary(f)$estimates
  cat(sprintf("%-22s mu=%.4f (%.4f)  tau=%.4f (%.4f)\n", label,
              est["mu", "Mean"], est["mu", "SD"],
              est["tau", "Mean"], est["tau", "SD"]))
}

steps <- c(0.025, 0.05)

show("fixed (1,1,0.5)",
     fit(prior_weightfunction("one-sided", steps = steps,
                              weights = wf_fixed(c(1, 1, 0.5))), 401))
show("fixed (0.5,1,1)",
     fit(prior_weightfunction("one-sided", steps = steps,
                              weights = wf_fixed(c(0.5, 1, 1))), 402))

free <- fit(prior_weightfunction("one-sided", steps = steps), 403)
show("free (cumulative)", free)
cat("\nomega posterior (free):\n")
print(summary(free)$estimates_bias)
