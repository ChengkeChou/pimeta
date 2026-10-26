#!/usr/bin/env Rscript
# R reference for tests/data/_nc_common_ref.csv and _nc_random_ref.csv, used by
# tests/test_pimeta.py::test_netcontrib_senn.
#
#   Rscript tools/r_reference/netcontrib_senn2013.R           # verify
#   Rscript tools/r_reference/netcontrib_senn2013.R --write   # rewrite
#
# `netcontrib` returns a 45 x 15 matrix: one row per network estimate (all pairs
# of the ten treatments) and one column per *direct* comparison, cell = the
# weight that direct comparison carries in that estimate.  The common-effects
# matrix does not depend on tau^2 and reproduces to 1e-13; the random-effects
# one does, which is what pins the estimator.
#
# The estimator is netmeta's default, DerSimonian-Laird.  That is not incidental:
# with method.tau = "REML" tau^2 is 0.10105 and the random-effects matrix is off
# by 3.2e-3, while with DL tau^2 is 0.10872 and it agrees to 4e-15.  pimeta's
# netmeta() also defaults to DL, so the fixture and the code agree for the same
# reason rather than by coincidence.

suppressPackageStartupMessages(library(netmeta))

args <- commandArgs(trailingOnly = TRUE)
write_fixture <- "--write" %in% args

this <- sub("^--file=", "",
            grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
root <- normalizePath(file.path(dirname(this), "..", ".."))
datadir <- file.path(root, "tests", "data")

data(Senn2013, package = "netmeta")
net <- netmeta(TE = Senn2013$TE, seTE = Senn2013$seTE,
               treat1 = Senn2013$treat1, treat2 = Senn2013$treat2,
               studlab = Senn2013$studlab, sm = "MD",
               reference.group = "plac")
nc <- netcontrib(net)

fmt <- function(m) {
  out <- m
  storage.mode(out) <- "character"
  out[] <- sprintf("%.16g", as.numeric(m))
  out <- cbind(rownames(m), out)
  colnames(out) <- c("", colnames(m))
  out
}

status <- 0L
for (nm in c("common", "random")) {
  ref_file <- file.path(datadir, paste0("_nc_", nm, "_ref.csv"))
  m <- fmt(nc[[nm]])
  if (write_fixture) {
    utils::write.table(m, file = ref_file, sep = ",", quote = TRUE,
                       row.names = FALSE, col.names = TRUE)
    cat("wrote", basename(ref_file), "\n")
    next
  }
  ref <- as.matrix(read.csv(ref_file, row.names = 1, check.names = FALSE))
  got <- nc[[nm]][rownames(ref), colnames(ref)]
  worst <- max(abs(got - ref))
  cat(sprintf("%-6s %3d x %2d  max abs diff %.3e\n", nm, nrow(ref), ncol(ref),
              worst))
  if (worst > 1e-10) status <- 1L
}
cat("netmeta", as.character(packageVersion("netmeta")), "| metafor",
    as.character(packageVersion("metafor")), "| R",
    as.character(getRversion()), "| tau2 =",
    format(net$tau2, digits = 10), "\n")
if (status != 0L) cat("MISMATCH\n")
quit(status = status)
