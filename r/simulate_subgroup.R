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

have_survRM2 <- requireNamespace("survRM2", quietly = TRUE)
have_coxphw <- requireNamespace("coxphw", quietly = TRUE)
have_flexsurv <- requireNamespace("flexsurv", quietly = TRUE)

p_from_chi1 <- function(x) pchisq(x, df = 1, lower.tail = FALSE)

sim_data <- function(n = 400, prev = 0.3, lambda0 = 0.12, hr_pos = 0.8, hr_neg = 1.0, cens = 0.05) {
  arm <- rbinom(n, 1, 0.5)
  biom <- rbinom(n, 1, prev)
  haz <- rep(lambda0, n)
  haz[arm == 1 & biom == 1] <- lambda0 * hr_pos
  haz[arm == 1 & biom == 0] <- lambda0 * hr_neg
  t_event <- rexp(n, rate = haz)
  t_cens <- rexp(n, rate = cens)
  time <- pmin(t_event, t_cens)
  status <- as.integer(t_event <= t_cens)
  data.frame(time = time, status = status, arm = arm, biom = biom)
}

safe <- function(expr, default = NA_real_) {
  out <- try(eval.parent(substitute(expr)), silent = TRUE)
  if (inherits(out, "try-error")) default else out
}

weighted_logrank_p <- function(d, rho = 0, gamma = 0) {
  sf <- survfit(Surv(time, status) ~ 1, data = d)
  # Map each event time to pooled KM S(t-)
  et <- sort(unique(d$time[d$status == 1]))
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
    d0 <- sum(d$arm == 0 & evt)
    dd <- d1 + d0
    if (n <= 1 || dd == 0) next
    O <- O + w * d1
    E <- E + w * dd * (n1 / n)
    V <- V + w^2 * (n1 * n0 * dd * (n - dd)) / (n^2 * (n - 1))
  }
  if (V <= 0) return(1)
  z <- (O - E)^2 / V
  p_from_chi1(z)
}

analyze_one <- function(d, tau = c(6, 12)) {
  out <- list()

  # Testing methods (Table 3)
  out$logrank_p <- safe({
    survival::survdiff(Surv(time, status) ~ arm, data = d)$chisq |> p_from_chi1()
  }, 1)

  out$wlr_gehan_p <- weighted_logrank_p(d, rho = 1, gamma = 0)
  out$wlr_fh00_p <- weighted_logrank_p(d, rho = 0, gamma = 0)
  out$wlr_fh01_p <- weighted_logrank_p(d, rho = 0, gamma = 1)
  out$wlr_fh11_p <- weighted_logrank_p(d, rho = 1, gamma = 1)
  out$wlr_fh10_p <- weighted_logrank_p(d, rho = 1, gamma = 0)
  out$wlr_modest_p <- weighted_logrank_p(d, rho = 0, gamma = 0.5)
  out$maxcombo_p <- min(c(out$wlr_fh00_p, out$wlr_fh01_p, out$wlr_fh11_p, out$wlr_fh10_p), na.rm = TRUE) * 4
  out$maxcombo_p <- min(out$maxcombo_p, 1)

  # Cox PH HR estimate + CI
  cfit <- try(coxph(Surv(time, status) ~ arm, data = d), silent = TRUE)
  if (!inherits(cfit, "try-error")) {
    b <- coef(cfit)[1]; se <- sqrt(vcov(cfit)[1,1])
    out$cox_hr <- exp(b)
    out$cox_lcl <- exp(b - 1.96 * se)
    out$cox_ucl <- exp(b + 1.96 * se)
    out$cox_p <- summary(cfit)$coef[1,5]
  } else {
    out$cox_hr <- out$cox_lcl <- out$cox_ucl <- out$cox_p <- NA_real_
  }

  # Difference in median survival (arm1-arm0)
  q1 <- safe({ as.numeric(quantile(survfit(Surv(time,status)~1, data=d[d$arm==1,]), probs=.5)$quantile) })
  q0 <- safe({ as.numeric(quantile(survfit(Surv(time,status)~1, data=d[d$arm==0,]), probs=.5)$quantile) })
  out$median_diff <- q1 - q0

  # Milestone KM survival (6,12)
  for (tt in tau) {
    sf <- try(survfit(Surv(time,status) ~ arm, data = d), silent = TRUE)
    if (!inherits(sf, "try-error")) {
      sm <- summary(sf, times = tt, extend = TRUE)
      # strata order arm=0 then arm=1
      s0 <- sm$surv[1]; s1 <- sm$surv[2]
      out[[paste0("mile",tt,"_diff")]] <- s1 - s0
    } else {
      out[[paste0("mile",tt,"_diff")]] <- NA_real_
    }
  }

  # RMST
  for (tt in tau) {
    key <- paste0("rmst",tt)
    if (have_survRM2) {
      rr <- try(survRM2::rmst2(time = d$time, status = d$status, arm = d$arm, tau = tt), silent = TRUE)
      if (!inherits(rr, "try-error")) {
        # arm1-arm0
        est <- rr$unadjusted.result["RMST (arm=1)-(arm=0)", "Est."]
        lcl <- rr$unadjusted.result["RMST (arm=1)-(arm=0)", "lower .95"]
        ucl <- rr$unadjusted.result["RMST (arm=1)-(arm=0)", "upper .95"]
        p <- rr$unadjusted.result["RMST (arm=1)-(arm=0)", "p"]
        out[[paste0(key, "_diff")]] <- est
        out[[paste0(key, "_lcl")]] <- lcl
        out[[paste0(key, "_ucl")]] <- ucl
        out[[paste0(key, "_p")]] <- p
      } else {
        out[[paste0(key, "_diff")]] <- out[[paste0(key, "_lcl")]] <- out[[paste0(key, "_ucl")]] <- out[[paste0(key, "_p")]] <- NA_real_
      }
    } else {
      out[[paste0(key, "_diff")]] <- out[[paste0(key, "_lcl")]] <- out[[paste0(key, "_ucl")]] <- out[[paste0(key, "_p")]] <- NA_real_
    }
  }

  # Weighted Cox (aHR)
  if (have_coxphw) {
    wfit <- try(coxphw::coxphw(Surv(time, status) ~ arm, data = d, template = "AHR"), silent = TRUE)
    if (!inherits(wfit, "try-error")) {
      b <- coef(wfit)[1]; se <- sqrt(vcov(wfit)[1,1])
      out$ahr <- exp(b)
      out$ahr_lcl <- exp(b - 1.96 * se)
      out$ahr_ucl <- exp(b + 1.96 * se)
      out$ahr_p <- 2 * pnorm(abs(b/se), lower.tail = FALSE)
    } else out$ahr <- out$ahr_lcl <- out$ahr_ucl <- out$ahr_p <- NA_real_
  } else out$ahr <- out$ahr_lcl <- out$ahr_ucl <- out$ahr_p <- NA_real_

  # Piecewise exponential model via split Poisson (3,12)
  cut <- c(3, 12)
  ds <- survSplit(Surv(time, status) ~ ., data = d, cut = cut, end = "time", event = "status")
  ds$interval <- factor(ds$time)
  ds$offset <- with(ds, log(time - pmax(0, c(0, head(time, -1)))))
  pch <- try(glm(status ~ arm + strata(interval), family = poisson(), data = ds), silent = TRUE)
  out$pch_p <- if (!inherits(pch, "try-error")) summary(pch)$coef["arm",4] else NA_real_

  # Parametric model (weibull)
  wb <- try(survreg(Surv(time,status) ~ arm, data = d, dist = "weibull"), silent = TRUE)
  if (!inherits(wb, "try-error")) {
    b <- coef(wb)["arm"]
    out$param_ahr <- exp(-b) # approximation to HR-like direction
  } else out$param_ahr <- NA_real_

  as.data.frame(out)
}

true_median_treatment <- function(sc) {
  f <- function(t) sc$prev * exp(-sc$lambda0 * sc$hr_pos * t) + (1 - sc$prev) * exp(-sc$lambda0 * sc$hr_neg * t) - 0.5
  uniroot(f, c(1e-8, 1e4))$root
}

true_ahr_coxphw <- function(sc, n_big = 5000, seed = 999) {
  if (!have_coxphw) return(NA_real_)
  set.seed(seed)
  d <- sim_data(n = n_big, prev = sc$prev, lambda0 = sc$lambda0, hr_pos = sc$hr_pos, hr_neg = sc$hr_neg, cens = sc$cens)
  fit <- try(coxphw::coxphw(Surv(time, status) ~ arm, data = d, template = "AHR"), silent = TRUE)
  if (inherits(fit, "try-error")) return(NA_real_)
  as.numeric(exp(coef(fit)[1]))
}

truth_table <- function(sc) {
  # True quantities under subgroup-mixture exponential model
  l0 <- sc$lambda0
  p <- sc$prev
  hp <- sc$hr_pos
  hn <- sc$hr_neg

  S0 <- function(t) exp(-l0 * t)
  S1 <- function(t) p * exp(-l0 * hp * t) + (1 - p) * exp(-l0 * hn * t)

  rmst <- function(Sfun, tau) integrate(Sfun, lower = 0, upper = tau)$value

  cox_truth <- {
    # pseudo-true PH coefficient from large sample Cox PH fit
    set.seed(12345)
    db <- sim_data(n = 5000, prev = p, lambda0 = l0, hr_pos = hp, hr_neg = hn, cens = sc$cens)
    cf <- coxph(Surv(time, status) ~ arm, data = db)
    as.numeric(exp(coef(cf)[1]))
  }

  list(
    cox_hr = cox_truth,
    ahr = true_ahr_coxphw(sc),
    rmst6_diff = rmst(S1, 6) - rmst(S0, 6),
    rmst12_diff = rmst(S1, 12) - rmst(S0, 12),
    mile6_diff = S1(6) - S0(6),
    mile12_diff = S1(12) - S0(12),
    median_diff = true_median_treatment(sc) - (log(2) / l0),
    param_ahr = cox_truth
  )
}

summarise_perf <- function(df, sc, alpha = 0.05) {
  truth <- truth_table(sc)

  test_cols <- c("logrank_p","wlr_gehan_p","wlr_fh00_p","wlr_fh01_p","wlr_fh11_p","wlr_fh10_p","wlr_modest_p","maxcombo_p","cox_p","rmst6_p","rmst12_p","ahr_p","pch_p")
  est_cols <- c("cox_hr","ahr","param_ahr","median_diff","rmst6_diff","rmst12_diff","mile6_diff","mile12_diff")

  out <- list()
  for (nm in test_cols) {
    if (!nm %in% names(df)) next
    p <- df[[nm]]
    out[[paste0(nm,"_reject")]] <- mean(p < alpha, na.rm = TRUE)
  }

  for (nm in est_cols) {
    if (!nm %in% names(df)) next
    x <- df[[nm]]
    t0 <- truth[[nm]]
    if (is.na(t0)) {
      out[[paste0(nm,"_mean")]] <- mean(x, na.rm = TRUE)
    } else {
      out[[paste0(nm,"_bias")]] <- mean(x - t0, na.rm = TRUE)
      out[[paste0(nm,"_mse")]] <- mean((x - t0)^2, na.rm = TRUE)
    }
  }

  # coverage + half-width where CIs available
  ci_sets <- list(
    cox = c("cox_lcl","cox_ucl","cox_hr"),
    ahr = c("ahr_lcl","ahr_ucl","ahr")
  )
  for (k in names(ci_sets)) {
    l <- df[[ci_sets[[k]][1]]]; u <- df[[ci_sets[[k]][2]]]
    t0 <- truth[[ci_sets[[k]][3]]]
    if (!is.null(l) && !is.null(u) && !is.na(t0)) {
      out[[paste0(k,"_coverage")]] <- mean(l <= t0 & u >= t0, na.rm = TRUE)
      out[[paste0(k,"_halfwidth")]] <- mean((u - l)/2, na.rm = TRUE)
    }
  }

  as.data.frame(out)
}

run_scenario <- function(sc, n_sims = 2500, cores = 1, seed = 1) {
  set.seed(seed)
  seeds <- sample.int(1e9, n_sims)

  one <- function(sd, sc_local) {
    set.seed(sd)
    d <- sim_data(n = sc_local$n_total, prev = sc_local$prev, lambda0 = sc_local$lambda0,
                  hr_pos = sc_local$hr_pos, hr_neg = sc_local$hr_neg, cens = sc_local$cens)
    analyze_one(d)
  }

  if (cores > 1) {
    cl <- makeCluster(cores)
    on.exit(stopCluster(cl), add = TRUE)
    clusterExport(cl, c("sim_data", "analyze_one", "weighted_logrank_p", "safe", "p_from_chi1",
                        "have_survRM2", "have_coxphw"), envir = environment())
    clusterEvalQ(cl, suppressPackageStartupMessages(library(survival)))
    if (have_survRM2) clusterEvalQ(cl, suppressPackageStartupMessages(library(survRM2)))
    if (have_coxphw) clusterEvalQ(cl, suppressPackageStartupMessages(library(coxphw)))
    rows <- parLapplyLB(cl, seeds, function(sd, sci) {
      set.seed(sd)
      d <- sim_data(n = sci$n_total, prev = sci$prev, lambda0 = sci$lambda0,
                    hr_pos = sci$hr_pos, hr_neg = sci$hr_neg, cens = sci$cens)
      analyze_one(d)
    }, sci = sc)
  } else {
    rows <- lapply(seeds, function(sd) one(sd, sc))
  }

  dat <- do.call(rbind, rows)
  perf <- summarise_perf(dat, sc)
  cbind(data.frame(scenario = sc$name, n_sims = n_sims, cores = cores), perf)
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  n_sims <- ifelse(length(args) >= 1, as.integer(args[1]), 2500)
  out_csv <- ifelse(length(args) >= 2, args[2], "results/subgroup_methods_summary.csv")

  scenarios <- list(
    list(name = "subgroup_null", n_total = 400, prev = 0.3, lambda0 = 0.12, hr_pos = 1.0, hr_neg = 1.0, cens = 0.05),
    list(name = "subgroup_mild", n_total = 400, prev = 0.3, lambda0 = 0.12, hr_pos = 0.8, hr_neg = 1.0, cens = 0.05),
    list(name = "subgroup_strong", n_total = 400, prev = 0.3, lambda0 = 0.12, hr_pos = 0.7, hr_neg = 1.0, cens = 0.05)
  )

  dir.create(dirname(out_csv), recursive = TRUE, showWarnings = FALSE)
  all <- do.call(rbind, lapply(seq_along(scenarios), function(i) run_scenario(scenarios[[i]], n_sims = n_sims, seed = 202600 + i)))
  write.csv(all, out_csv, row.names = FALSE)
  cat("Wrote", out_csv, "\n")
}

main()
