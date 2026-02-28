#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(survival)
  library(parallel)
})

p_from_chi1 <- function(x) pchisq(x, df=1, lower.tail=FALSE)

sim_data <- function(n, prev, lambda0, hr_pos, hr_neg, lambdaC) {
  arm <- rbinom(n,1,0.5)
  biom <- rbinom(n,1,prev)
  haz <- rep(lambda0, n)
  haz[arm==1 & biom==1] <- lambda0*hr_pos
  haz[arm==1 & biom==0] <- lambda0*hr_neg
  te <- rexp(n, rate=haz)
  tc <- if (lambdaC > 0) rexp(n, rate=lambdaC) else rep(Inf,n)
  t <- pmin(te, tc)
  d <- as.integer(te<=tc)
  data.frame(time=t, status=d, arm=arm, biom=biom)
}

weighted_logrank_p <- function(d, rho=0, gamma=0) {
  sf <- survfit(Surv(time,status)~1, data=d)
  et <- sort(unique(d$time[d$status==1]))
  if (length(et)==0) return(1)
  ss <- summary(sf, times=et, extend=TRUE)
  S <- ss$surv
  O <- E <- V <- 0
  for (i in seq_along(et)) {
    tt <- et[i]; w <- (S[i]^rho)*((1-S[i])^gamma)
    risk <- d$time >= tt
    n1 <- sum(d$arm==1 & risk); n0 <- sum(d$arm==0 & risk); n <- n1+n0
    evt <- d$time==tt & d$status==1
    d1 <- sum(d$arm==1 & evt); dd <- sum(evt)
    if (n<=1 || dd==0) next
    O <- O + w*d1
    E <- E + w*dd*(n1/n)
    V <- V + w^2 * (n1*n0*dd*(n-dd))/((n^2)*(n-1))
  }
  if (V<=0) return(1)
  p_from_chi1((O-E)^2/V)
}

analyze_one <- function(d) {
  out <- list()
  out$logrank_p <- tryCatch(p_from_chi1(survdiff(Surv(time,status)~arm,data=d)$chisq), error=function(e) 1)
  out$wlr_fh00_p <- weighted_logrank_p(d,0,0)
  out$wlr_fh01_p <- weighted_logrank_p(d,0,1)
  out$wlr_fh11_p <- weighted_logrank_p(d,1,1)
  out$wlr_fh10_p <- weighted_logrank_p(d,1,0)

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

  cf <- try(coxph(Surv(time,status)~arm,data=d), silent=TRUE)
  if (!inherits(cf,'try-error')) {
    b <- coef(cf)[1]
    out$cox_p <- summary(cf)$coef[1,5]
    out$cox_hr <- exp(b)
  } else {
    out$cox_p <- NA_real_; out$cox_hr <- NA_real_
  }

  sf <- try(survfit(Surv(time,status)~arm,data=d), silent=TRUE)
  if (!inherits(sf,'try-error')) {
    sm6 <- summary(sf,times=6,extend=TRUE)$surv
    sm12 <- summary(sf,times=12,extend=TRUE)$surv
    out$mile6_diff <- sm6[2]-sm6[1]
    out$mile12_diff <- sm12[2]-sm12[1]
  } else {
    out$mile6_diff <- NA_real_; out$mile12_diff <- NA_real_
  }

  if (requireNamespace('survRM2', quietly=TRUE)) {
    r6 <- try(survRM2::rmst2(time=d$time,status=d$status,arm=d$arm,tau=6),silent=TRUE)
    r12 <- try(survRM2::rmst2(time=d$time,status=d$status,arm=d$arm,tau=12),silent=TRUE)
    if (!inherits(r6,'try-error')) {
      out$rmst6_diff <- r6$unadjusted.result['RMST (arm=1)-(arm=0)','Est.']
      out$rmst6_p <- r6$unadjusted.result['RMST (arm=1)-(arm=0)','p']
    } else {out$rmst6_diff <- NA_real_; out$rmst6_p <- NA_real_}
    if (!inherits(r12,'try-error')) {
      out$rmst12_diff <- r12$unadjusted.result['RMST (arm=1)-(arm=0)','Est.']
      out$rmst12_p <- r12$unadjusted.result['RMST (arm=1)-(arm=0)','p']
    } else {out$rmst12_diff <- NA_real_; out$rmst12_p <- NA_real_}
  } else {
    out$rmst6_diff <- out$rmst6_p <- out$rmst12_diff <- out$rmst12_p <- NA_real_
  }
  as.data.frame(out)
}

truth_values <- function(prev, lambda0, hr_pos, hr_neg) {
  S0 <- function(t) exp(-lambda0*t)
  S1 <- function(t) prev*exp(-lambda0*hr_pos*t) + (1-prev)*exp(-lambda0*hr_neg*t)
  rmst <- function(tau) {
    (prev*(1-exp(-lambda0*hr_pos*tau))/(lambda0*hr_pos) +
      (1-prev)*(1-exp(-lambda0*hr_neg*tau))/(lambda0*hr_neg)) -
      ((1-exp(-lambda0*tau))/lambda0)
  }
  c(
    true_mile6 = S1(6)-S0(6),
    true_mile12 = S1(12)-S0(12),
    true_rmst6 = rmst(6),
    true_rmst12 = rmst(12),
    true_hr_mix = prev*hr_pos + (1-prev)*hr_neg
  )
}

summarise_scenario <- function(rows, truth, alpha=0.05) {
  R <- nrow(rows)
  pcols <- c('logrank_p','wlr_fh00_p','wlr_fh01_p','wlr_fh11_p','wlr_fh10_p','maxcombo_p','cox_p','rmst6_p','rmst12_p')

  mcse_p <- function(p, R) sqrt(pmax(p * (1 - p), 0) / R)
  mcse_bias <- function(bias, mse, R) sqrt(pmax(mse - bias^2, 0) / R)

  out <- list()
  for (pc in pcols) {
    pr <- mean(rows[[pc]] < alpha, na.rm=TRUE)
    out[[paste0(pc,'_reject')]] <- pr
    out[[paste0(pc,'_reject_mcse')]] <- mcse_p(pr, R)
  }

  cox_delta <- rows$cox_hr - truth['true_hr_mix']
  out$cox_hr_bias <- mean(cox_delta, na.rm=TRUE)
  out$cox_hr_mse  <- mean(cox_delta^2, na.rm=TRUE)
  out$cox_hr_bias_mcse <- mcse_bias(out$cox_hr_bias, out$cox_hr_mse, R)

  r6_delta <- rows$rmst6_diff - truth['true_rmst6']
  out$rmst6_bias <- mean(r6_delta, na.rm=TRUE)
  out$rmst6_mse  <- mean(r6_delta^2, na.rm=TRUE)
  out$rmst6_bias_mcse <- mcse_bias(out$rmst6_bias, out$rmst6_mse, R)

  r12_delta <- rows$rmst12_diff - truth['true_rmst12']
  out$rmst12_bias <- mean(r12_delta, na.rm=TRUE)
  out$rmst12_mse  <- mean(r12_delta^2, na.rm=TRUE)
  out$rmst12_bias_mcse <- mcse_bias(out$rmst12_bias, out$rmst12_mse, R)

  m6_delta <- rows$mile6_diff - truth['true_mile6']
  out$mile6_bias <- mean(m6_delta, na.rm=TRUE)
  out$mile6_mse  <- mean(m6_delta^2, na.rm=TRUE)
  out$mile6_bias_mcse <- mcse_bias(out$mile6_bias, out$mile6_mse, R)

  m12_delta <- rows$mile12_diff - truth['true_mile12']
  out$mile12_bias <- mean(m12_delta, na.rm=TRUE)
  out$mile12_mse  <- mean(m12_delta^2, na.rm=TRUE)
  out$mile12_bias_mcse <- mcse_bias(out$mile12_bias, out$mile12_mse, R)

  as.data.frame(out)
}

run <- function(replications=200, cores=max(1, detectCores()-1), out='results/subgroup_protocol_grid_summary.csv', base_seed=20260228L, max_cells=NULL) {
  lambda0_levels <- c(log(2)/36, log(2)/12, log(2)/6)
  cens_props <- c(0.0, 0.1, 0.3)
  prev_levels <- c(0.1,0.3,0.5)
  overall_levels <- c(1.0,0.9,0.8,0.7)
  rel_levels <- c(0.9,0.8,0.7)
  n_levels <- c(300,500,1000,1500)

  grid <- expand.grid(lambda0=lambda0_levels, cens_prop=cens_props, prev=prev_levels,
                      hr_overall=overall_levels, rel=rel_levels, n_total=n_levels)
  grid <- grid[order(grid$n_total, grid$lambda0, grid$cens_prop, grid$prev, grid$hr_overall, grid$rel), ]
  rownames(grid) <- NULL
  grid$scenario_id <- seq_len(nrow(grid))
  if (!is.null(max_cells)) {
    grid <- head(grid, as.integer(max_cells))
  }

  dir.create(dirname(out), recursive=TRUE, showWarnings=FALSE)
  results <- list()

  for (i in seq_len(nrow(grid))) {
    g <- grid[i,]
    hr_pos <- g$hr_overall * g$rel
    hr_neg <- (g$hr_overall - g$prev*hr_pos)/(1-g$prev)
    if (hr_neg <= 0) next
    lambdaC <- if (g$cens_prop==0) 0 else (g$cens_prop/(1-g$cens_prop))*g$lambda0

    truth <- truth_values(g$prev,g$lambda0,hr_pos,hr_neg)

    set.seed(base_seed + as.integer(g$scenario_id))
    seeds <- sample.int(1e9, replications)
    one <- function(sd) {
      set.seed(sd)
      d <- sim_data(n=g$n_total, prev=g$prev, lambda0=g$lambda0, hr_pos=hr_pos, hr_neg=hr_neg, lambdaC=lambdaC)
      analyze_one(d)
    }
    rows <- if (cores>1) mclapply(seeds, one, mc.cores=cores) else lapply(seeds, one)
    rows <- do.call(rbind, rows)
    s <- summarise_scenario(rows, truth)
    s$scenario_id <- g$scenario_id
    s$replications <- replications
    s$n_total <- g$n_total
    s$lambda0 <- g$lambda0
    s$cens_prop <- g$cens_prop
    s$prev <- g$prev
    s$hr_overall <- g$hr_overall
    s$rel <- g$rel
    s$hr_pos <- hr_pos
    s$hr_neg <- hr_neg
    s$true_hr_mix <- truth['true_hr_mix']
    s$true_rmst6 <- truth['true_rmst6']
    s$true_rmst12 <- truth['true_rmst12']
    s$true_mile6 <- truth['true_mile6']
    s$true_mile12 <- truth['true_mile12']

    results[[length(results)+1]] <- s
    if (i %% 10 == 0) {
      cat('Completed', i, 'of', nrow(grid), '\n')
      tmp <- do.call(rbind, results)
      write.csv(tmp, out, row.names=FALSE)
    }
  }
  final <- do.call(rbind, results)
  write.csv(final, out, row.names=FALSE)
  cat('Wrote', out, '\n')
}

if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly=TRUE)
  replications <- ifelse(length(args)>=1, as.integer(args[1]), 200)
  cores <- ifelse(length(args)>=2, as.integer(args[2]), max(1, detectCores()-1))
  out <- ifelse(length(args)>=3, args[3], 'results/subgroup_protocol_grid_summary.csv')
  base_seed <- ifelse(length(args)>=4, as.integer(args[4]), 20260228L)
  max_cells <- ifelse(length(args)>=5, as.integer(args[5]), NA_integer_)
  if (is.na(max_cells)) max_cells <- NULL
  run(replications, cores, out, base_seed, max_cells)
}
