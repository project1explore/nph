#!/usr/bin/env Rscript
resolve_repo_root <- function() {
  file_arg <- "--file="
  cmd_args <- commandArgs(trailingOnly = FALSE)
  script_arg <- cmd_args[grep(paste0("^", file_arg), cmd_args)]
  if (length(script_arg) > 0) {
    script_path <- normalizePath(sub(file_arg, "", script_arg[1]), winslash = "/", mustWork = TRUE)
    return(normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = TRUE))
  }

  frame_files <- vapply(sys.frames(), function(x) if (!is.null(x$ofile)) x$ofile else "", character(1))
  frame_files <- frame_files[nzchar(frame_files)]
  if (length(frame_files) > 0) {
    script_path <- normalizePath(frame_files[1], winslash = "/", mustWork = TRUE)
    return(normalizePath(file.path(dirname(script_path), ".."), winslash = "/", mustWork = TRUE))
  }

  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

repo_root <- resolve_repo_root()
setwd(repo_root)

suppressPackageStartupMessages({
  library(survival)
  library(parallel)
})

p_from_chi1 <- function(x) pchisq(x, df = 1, lower.tail = FALSE)

sim_data <- function(n, prev, lambda0, lambdaC) {
  arm <- rbinom(n, 1, 0.5)
  biom <- rbinom(n, 1, prev)
  haz <- rep(lambda0, n) # strict null: hr_pos = hr_neg = 1
  te <- rexp(n, rate = haz)
  tc <- if (lambdaC > 0) rexp(n, rate = lambdaC) else rep(Inf, n)
  t <- pmin(te, tc)
  d <- as.integer(te <= tc)
  data.frame(time = t, status = d, arm = arm, biom = biom)
}

weighted_logrank_p <- function(d, rho = 0, gamma = 0) {
  sf <- survfit(Surv(time, status) ~ 1, data = d)
  et <- sort(unique(d$time[d$status == 1]))
  if (length(et) == 0) return(1)
  ss <- summary(sf, times = et, extend = TRUE)
  S <- ss$surv

  O <- E <- V <- 0
  for (i in seq_along(et)) {
    tt <- et[i]
    w <- (S[i]^rho) * ((1 - S[i])^gamma)
    risk <- d$time >= tt
    n1 <- sum(d$arm == 1 & risk)
    n0 <- sum(d$arm == 0 & risk)
    n <- n1 + n0
    evt <- d$time == tt & d$status == 1
    d1 <- sum(d$arm == 1 & evt)
    dd <- sum(evt)
    if (n <= 1 || dd == 0) next

    O <- O + w * d1
    E <- E + w * dd * (n1 / n)
    V <- V + w^2 * (n1 * n0 * dd * (n - dd)) / ((n^2) * (n - 1))
  }
  if (V <= 0) return(1)
  p_from_chi1((O - E)^2 / V)
}

analyze_one <- function(d) {
  out <- list()
  out$logrank_p <- tryCatch(p_from_chi1(survdiff(Surv(time, status) ~ arm, data = d)$chisq), error = function(e) 1)
  out$wlr_fh00_p <- weighted_logrank_p(d, 0, 0)
  out$wlr_fh01_p <- weighted_logrank_p(d, 0, 1)
  out$wlr_fh11_p <- weighted_logrank_p(d, 1, 1)
  out$wlr_fh10_p <- weighted_logrank_p(d, 1, 0)

  if (requireNamespace('nph', quietly = TRUE)) {
    mc <- try(nph::logrank.maxtest(
      time = d$time,
      event = d$status,
      group = d$arm,
      alternative = 'two.sided',
      rho = c(0, 0, 1, 1),
      gamma = c(0, 1, 1, 0)
    ), silent = TRUE)
    out$maxcombo_p <- if (!inherits(mc, 'try-error')) as.numeric(mc$pmult) else NA_real_
    out$maxcombo_bonf_p <- if (!inherits(mc, 'try-error')) as.numeric(mc$p.Bonf) else NA_real_
  } else {
    out$maxcombo_p <- NA_real_
    out$maxcombo_bonf_p <- NA_real_
  }

  cf <- try(coxph(Surv(time, status) ~ arm, data = d), silent = TRUE)
  out$cox_p <- if (!inherits(cf, 'try-error')) summary(cf)$coef[1, 5] else NA_real_

  if (requireNamespace('survRM2', quietly = TRUE)) {
    r12 <- try(survRM2::rmst2(time = d$time, status = d$status, arm = d$arm, tau = 12), silent = TRUE)
    out$rmst12_p <- if (!inherits(r12, 'try-error')) r12$unadjusted.result['RMST (arm=1)-(arm=0)', 'p'] else NA_real_
  } else {
    out$rmst12_p <- NA_real_
  }

  out$censor_frac <- mean(d$status == 0)
  out$censor_frac_arm0 <- mean(d$status[d$arm == 0] == 0)
  out$censor_frac_arm1 <- mean(d$status[d$arm == 1] == 0)
  as.data.frame(out)
}

mcse_p <- function(p, R) sqrt(p * (1 - p) / R)

run_one_scenario <- function(sc, R = 2000, cores = 2, base_seed = 20260228L) {
  set.seed(base_seed + sc$scenario_id)
  seeds <- sample.int(1e9, R)

  one <- function(sd) {
    set.seed(sd)
    lambdaC <- if (sc$cens_prop == 0) 0 else (sc$cens_prop / (1 - sc$cens_prop)) * sc$lambda0
    d <- sim_data(n = sc$n_total, prev = sc$prev, lambda0 = sc$lambda0, lambdaC = lambdaC)
    analyze_one(d)
  }

  rows <- if (cores > 1) mclapply(seeds, one, mc.cores = cores) else lapply(seeds, one)
  rows <- do.call(rbind, rows)

  methods <- c('logrank_p', 'wlr_fh00_p', 'wlr_fh01_p', 'wlr_fh11_p', 'wlr_fh10_p', 'maxcombo_p', 'maxcombo_bonf_p', 'cox_p', 'rmst12_p')
  out <- list(
    scenario_id = sc$scenario_id,
    n_total = sc$n_total,
    lambda0 = sc$lambda0,
    cens_prop = sc$cens_prop,
    prev = sc$prev,
    replications = R,
    mean_censor_frac = mean(rows$censor_frac, na.rm = TRUE),
    mean_censor_frac_arm0 = mean(rows$censor_frac_arm0, na.rm = TRUE),
    mean_censor_frac_arm1 = mean(rows$censor_frac_arm1, na.rm = TRUE)
  )

  for (m in methods) {
    pr <- mean(rows[[m]] < 0.05, na.rm = TRUE)
    out[[paste0(m, '_reject')]] <- pr
    out[[paste0(m, '_reject_mcse')]] <- mcse_p(pr, R)
  }

  as.data.frame(out)
}

run <- function(R = 2000, cores = 2, out = 'results/strict_null_benchmark.csv') {
  scenarios <- data.frame(
    scenario_id = 1:6,
    n_total = c(300, 500, 500, 500, 1500, 500),
    lambda0 = c(log(2) / 12, log(2) / 12, log(2) / 12, log(2) / 36, log(2) / 12, log(2) / 6),
    cens_prop = c(0.1, 0.1, 0.3, 0.1, 0.1, 0.1),
    prev = c(0.3, 0.3, 0.3, 0.3, 0.3, 0.5)
  )

  dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
  ans <- lapply(seq_len(nrow(scenarios)), function(i) {
    cat('Running strict-null scenario', i, 'of', nrow(scenarios), '\n')
    run_one_scenario(scenarios[i, ], R = R, cores = cores)
  })
  ans <- do.call(rbind, ans)
  write.csv(ans, out, row.names = FALSE)
  cat('Wrote', out, '\n')
}

if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  R <- ifelse(length(args) >= 1, as.integer(args[1]), 2000)
  cores <- ifelse(length(args) >= 2, as.integer(args[2]), 2)
  out <- ifelse(length(args) >= 3, args[3], 'results/strict_null_benchmark.csv')
  run(R, cores, out)
}
