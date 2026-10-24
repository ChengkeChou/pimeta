suppressMessages(library(metafor))

root <- if (file.exists("tests/data/dynhet.csv")) "." else ".."
d <- read.csv(file.path(root, "tests/data/dynhet.csv"), stringsAsFactors = FALSE)
d$year_c <- d$year - mean(d$year)
d$year_c2 <- d$year_c^2

out <- list()
add <- function(tag, fit) {
  out[[length(out) + 1]] <<- data.frame(
    model = tag, part = "location", term = rownames(fit$beta),
    estimate = as.numeric(fit$beta), se = as.numeric(fit$se),
    statistic = as.numeric(fit$zval), p_value = as.numeric(fit$pval),
    ci_low = as.numeric(fit$ci.lb), ci_high = as.numeric(fit$ci.ub),
    loglik = as.numeric(logLik(fit)), stringsAsFactors = FALSE)
  out[[length(out) + 1]] <<- data.frame(
    model = tag, part = "scale", term = rownames(fit$alpha),
    estimate = as.numeric(fit$alpha), se = as.numeric(fit$se.alpha),
    statistic = as.numeric(fit$zval.alpha), p_value = as.numeric(fit$pval.alpha),
    ci_low = as.numeric(fit$ci.lb.alpha), ci_high = as.numeric(fit$ci.ub.alpha),
    loglik = as.numeric(logLik(fit)), stringsAsFactors = FALSE)
}

fit_A_REML <- rma.uni(d$yi, d$vi, mods = ~ year_c, scale = ~ year_c, data = d, method = "REML")
fit_A_ML   <- rma.uni(d$yi, d$vi, mods = ~ year_c, scale = ~ year_c, data = d, method = "ML")
fit_B_REML <- rma.uni(d$yi, d$vi, mods = ~ year_c, scale = ~ 1, data = d, method = "REML")
fit_B_ML   <- rma.uni(d$yi, d$vi, mods = ~ year_c, scale = ~ 1, data = d, method = "ML")
fit_C_REML <- rma.uni(d$yi, d$vi, mods = ~ 1, scale = ~ 1, data = d, method = "REML")
fit_C_ML   <- rma.uni(d$yi, d$vi, mods = ~ 1, scale = ~ 1, data = d, method = "ML")
fit_D_REML <- rma.uni(d$yi, d$vi, mods = ~ year_c + year_c2, scale = ~ year_c, data = d, method = "REML")
fit_D_ML   <- rma.uni(d$yi, d$vi, mods = ~ year_c + year_c2, scale = ~ year_c, data = d, method = "ML")
fit_A_knha <- rma.uni(d$yi, d$vi, mods = ~ year_c, scale = ~ year_c, data = d, method = "REML", test = "knha")

add("A_REML", fit_A_REML); add("A_ML", fit_A_ML)
add("B_REML", fit_B_REML); add("B_ML", fit_B_ML)
add("C_REML", fit_C_REML); add("C_ML", fit_C_ML)
add("D_REML", fit_D_REML); add("D_ML", fit_D_ML)
add("A_knha", fit_A_knha)

res <- do.call(rbind, out)
res$s2w <- NA_real_
res$s2w[res$model == "A_knha"] <- fit_A_knha$s2w
res$QS <- NA_real_; res$QSdf <- NA_real_; res$QSp <- NA_real_
rows <- res$model == "A_REML"
res$QS[rows] <- fit_A_REML$QS
res$QSdf[rows] <- fit_A_REML$QSdf
res$QSp[rows] <- fit_A_REML$QSp

path <- file.path(root, "tests/reference/dynhet_metafor.csv")
write.csv(res, path, row.names = FALSE)
cat("metafor", as.character(packageVersion("metafor")), "\n")
cat("wrote", path, "\n")
print(res[, c("model","part","term","estimate","se","statistic","p_value","loglik")], digits = 7, row.names = FALSE)