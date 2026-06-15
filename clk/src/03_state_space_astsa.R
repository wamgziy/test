#!/usr/bin/env Rscript

# State-space vendor comparison using astsa::Kfilter2.
#
# Model for one CUSIP or bucket at a time:
#
#   x_t = x_{t-1} + beta_cc * d_current_coupon_t
#                   + beta_10y * d_ten_year_rate_t + w_t
#
#   y_{A,t} = x_t + e_{A,t}
#   y_{B,t} = x_t + alpha_B + gamma_B,regime_t + e_{B,t}
#   y_{C,t} = x_t + alpha_C + gamma_C,regime_t + e_{C,t}
#
# Vendor A is the reference vendor for identification. Change it in config.
# The filtered latent price x_t is not an external truth; it is a latent
# model-based consensus informed by market drivers and all vendors.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript src/03_state_space_astsa.R config/regime_config.json")
}

if (!requireNamespace("astsa", quietly = TRUE)) {
  stop("The astsa package is required. Install it with install.packages('astsa').")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("The jsonlite package is required. Install it with install.packages('jsonlite').")
}

cfg_path <- normalizePath(args[[1]], mustWork = TRUE)
cfg <- jsonlite::fromJSON(cfg_path, simplifyVector = FALSE)
base_dir <- dirname(dirname(cfg_path))

resolve_path <- function(path) {
  if (grepl("^/", path)) return(path)
  file.path(base_dir, path)
}

read_required_csv <- function(path) {
  full <- resolve_path(path)
  if (!file.exists(full)) stop("Missing file: ", full)
  read.csv(full, stringsAsFactors = FALSE, check.names = FALSE)
}

date_col <- cfg$columns$date
cusip_col <- cfg$columns$cusip
vendor_col <- cfg$columns$vendor
price_col <- cfg$columns$price
bucket_col <- cfg$columns$bucket
weight_col <- cfg$columns$weight
cc_col <- cfg$columns$current_coupon
ten_col <- cfg$columns$ten_year_rate
vendors <- unlist(cfg$vendors)
reference_vendor <- cfg$state_space$reference_vendor
if (!(reference_vendor %in% vendors)) stop("reference_vendor must be in vendors.")
vendors <- c(reference_vendor, setdiff(vendors, reference_vendor))

prices <- read_required_csv(cfg$input$price_file)
market <- read_required_csv(cfg$input$market_file)
regimes <- read_required_csv(file.path(cfg$output$tables_dir, "regimes_by_date.csv"))

required_price <- c(date_col, cusip_col, vendor_col, price_col)
missing_price <- setdiff(required_price, names(prices))
if (length(missing_price) > 0) stop("Price file missing columns: ", paste(missing_price, collapse = ", "))

prices[[date_col]] <- as.Date(prices[[date_col]])
market[[date_col]] <- as.Date(market[[date_col]])
regimes$date <- as.Date(regimes$date)

if (!(weight_col %in% names(prices))) prices[[weight_col]] <- 1
if (!(bucket_col %in% names(prices))) prices[[bucket_col]] <- "ALL"

state_cfg <- cfg$state_space
unit_col <- state_cfg$unit_col
if (!(unit_col %in% names(prices))) stop("state_space.unit_col not found in price file: ", unit_col)
max_units <- as.integer(state_cfg$max_units)
min_obs <- as.integer(state_cfg$min_observations)
regime_col <- state_cfg$regime_column
if (!(regime_col %in% names(regimes))) stop("Regime column not found in regimes_by_date.csv: ", regime_col)

market <- market[order(market[[date_col]]), ]
market$d_current_coupon <- c(NA, diff(market[[cc_col]]))
market$d_ten_year_rate <- c(NA, diff(market[[ten_col]]))
market_features <- market[, c(date_col, "d_current_coupon", "d_ten_year_rate")]
names(market_features)[1] <- "date"

weighted_mean <- function(x, w) {
  ok <- is.finite(x) & is.finite(w) & w > 0
  if (!any(ok)) return(mean(x, na.rm = TRUE))
  sum(x[ok] * w[ok]) / sum(w[ok])
}

make_unit_vendor_wide <- function(prices, unit_id) {
  sub <- prices[prices[[unit_col]] == unit_id & prices[[vendor_col]] %in% vendors, ]
  if (nrow(sub) == 0) return(NULL)
  aggregate_rows <- aggregate(
    sub[[price_col]],
    by = list(date = sub[[date_col]], vendor = sub[[vendor_col]]),
    FUN = mean,
    na.rm = TRUE
  )
  names(aggregate_rows)[3] <- "price"
  wide <- reshape(
    aggregate_rows,
    idvar = "date",
    timevar = "vendor",
    direction = "wide"
  )
  names(wide) <- sub("^price\\.", "", names(wide))
  wide <- wide[order(wide$date), ]
  wide
}

make_inputs <- function(df, regime_col) {
  regime_raw <- as.character(df[[regime_col]])
  regime_raw[is.na(regime_raw)] <- "missing"
  regime_levels <- setdiff(sort(unique(regime_raw)), c("normal", "missing", "insufficient_history"))
  if (length(regime_levels) == 0) {
    input <- cbind(
      d_current_coupon = df$d_current_coupon,
      d_ten_year_rate = df$d_ten_year_rate,
      const = 1
    )
  } else {
    dummies <- sapply(regime_levels, function(z) as.numeric(regime_raw == z))
    if (is.null(dim(dummies))) dummies <- matrix(dummies, ncol = 1)
    colnames(dummies) <- paste0("regime_", make.names(regime_levels))
    input <- cbind(
      d_current_coupon = df$d_current_coupon,
      d_ten_year_rate = df$d_ten_year_rate,
      const = 1,
      dummies
    )
  }
  input[!is.finite(input)] <- 0
  input
}

fit_one_unit <- function(unit_id) {
  wide <- make_unit_vendor_wide(prices, unit_id)
  if (is.null(wide)) return(NULL)
  missing_vendors <- setdiff(vendors, names(wide))
  if (length(missing_vendors) > 0) return(NULL)

  df <- merge(wide[, c("date", vendors)], market_features, by = "date", all.x = TRUE)
  df <- merge(df, regimes[, c("date", regime_col)], by = "date", all.x = TRUE)
  df <- df[order(df$date), ]
  complete <- complete.cases(df[, c(vendors, "d_current_coupon", "d_ten_year_rate")])
  df <- df[complete, ]
  if (nrow(df) < min_obs) return(NULL)

  y <- as.matrix(df[, vendors])
  storage.mode(y) <- "double"
  input <- make_inputs(df, regime_col)
  n <- nrow(y)
  qdim <- length(vendors)
  rdim <- ncol(input)
  regime_input_cols <- setdiff(seq_len(rdim), c(1, 2, 3))

  A <- array(1, dim = c(qdim, 1, n))
  Phi <- matrix(1, 1, 1)
  Theta <- matrix(1, 1, 1)
  S <- matrix(0, 1, qdim)
  mu0 <- matrix(median(y[1, ], na.rm = TRUE), 1, 1)
  Sigma0 <- matrix(100, 1, 1)

  n_other <- qdim - 1
  n_gamma <- n_other * length(regime_input_cols)
  par_names <- c(
    "beta_current_coupon",
    "beta_ten_year_rate",
    "log_state_sd",
    paste0("log_obs_sd_", vendors),
    paste0("alpha_", vendors[-1]),
    if (n_gamma > 0) as.vector(outer(vendors[-1], colnames(input)[regime_input_cols], paste, sep = "_")) else character(0)
  )

  initial_sd <- apply(y, 2, sd, na.rm = TRUE)
  initial_sd[!is.finite(initial_sd) | initial_sd <= 0] <- 0.5
  par0 <- c(
    beta_current_coupon = 0,
    beta_ten_year_rate = 0,
    log_state_sd = log(0.25),
    setNames(log(pmax(initial_sd / 5, 0.05)), paste0("log_obs_sd_", vendors)),
    setNames(rep(0, n_other), paste0("alpha_", vendors[-1])),
    if (n_gamma > 0) setNames(rep(0, n_gamma), par_names[(3 + qdim + n_other + 1):length(par_names)]) else numeric(0)
  )

  build_matrices <- function(par) {
    Ups <- matrix(0, 1, rdim)
    Ups[1, 1] <- par["beta_current_coupon"]
    Ups[1, 2] <- par["beta_ten_year_rate"]

    Gam <- matrix(0, qdim, rdim)
    if (rdim >= 3) {
      for (j in seq_along(vendors[-1])) {
        row <- j + 1
        Gam[row, 3] <- par[paste0("alpha_", vendors[-1][j])]
      }
    }
    if (length(regime_input_cols) > 0) {
      idx <- 3 + qdim + n_other
      gamma_values <- par[(idx + 1):length(par)]
      gamma_mat <- matrix(gamma_values, nrow = n_other, byrow = TRUE)
      for (j in seq_len(n_other)) {
        Gam[j + 1, regime_input_cols] <- gamma_mat[j, ]
      }
    }

    cQ <- matrix(exp(par["log_state_sd"]), 1, 1)
    cR <- diag(exp(par[paste0("log_obs_sd_", vendors)]), qdim, qdim)
    list(Ups = Ups, Gam = Gam, cQ = cQ, cR = cR)
  }

  neg_loglik <- function(par) {
    mats <- build_matrices(par)
    val <- tryCatch(
      astsa::Kfilter2(
        num = n,
        y = y,
        A = A,
        mu0 = mu0,
        Sigma0 = Sigma0,
        Phi = Phi,
        Ups = mats$Ups,
        Gam = mats$Gam,
        Theta = Theta,
        cQ = mats$cQ,
        cR = mats$cR,
        S = S,
        input = input
      )$like,
      error = function(e) Inf
    )
    if (!is.finite(val)) return(1e12)
    as.numeric(val)
  }

  opt <- optim(par0, neg_loglik, method = "BFGS", control = list(maxit = 1000, reltol = 1e-8))
  mats <- build_matrices(opt$par)
  kf <- astsa::Kfilter2(
    num = n,
    y = y,
    A = A,
    mu0 = mu0,
    Sigma0 = Sigma0,
    Phi = Phi,
    Ups = mats$Ups,
    Gam = mats$Gam,
    Theta = Theta,
    cQ = mats$cQ,
    cR = mats$cR,
    S = S,
    input = input
  )

  latent <- data.frame(
    unit = unit_id,
    date = df$date,
    filtered_latent_price = as.numeric(kf$xf[1, 1, ]),
    predicted_latent_price = as.numeric(kf$xp[1, 1, ])
  )

  innov_rows <- list()
  for (j in seq_along(vendors)) {
    sig_j <- sapply(seq_len(n), function(i) sqrt(kf$sig[j, j, i]))
    innov_j <- as.numeric(kf$innov[j, 1, ])
    innov_rows[[j]] <- data.frame(
      unit = unit_id,
      date = df$date,
      vendor = vendors[j],
      regime = as.character(df[[regime_col]]),
      innovation = innov_j,
      innovation_sd = sig_j,
      standardized_innovation = innov_j / sig_j
    )
  }
  innovations <- do.call(rbind, innov_rows)

  params <- data.frame(
    unit = unit_id,
    parameter = names(opt$par),
    estimate = as.numeric(opt$par),
    row.names = NULL
  )
  sd_rows <- data.frame(
    unit = unit_id,
    parameter = c("state_sd", paste0("obs_sd_", vendors)),
    estimate = c(exp(opt$par["log_state_sd"]), exp(opt$par[paste0("log_obs_sd_", vendors)])),
    row.names = NULL
  )
  params <- rbind(params, sd_rows)

  score <- aggregate(
    cbind(
      innovation = innovations$innovation,
      abs_innovation = abs(innovations$innovation),
      standardized_abs_innovation = abs(innovations$standardized_innovation),
      outlier = as.numeric(abs(innovations$standardized_innovation) > 3)
    ),
    by = list(unit = innovations$unit, vendor = innovations$vendor, regime = innovations$regime),
    FUN = function(x) c(mean = mean(x, na.rm = TRUE), median = median(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE))
  )
  score <- do.call(data.frame, score)
  names(score) <- gsub("\\.", "_", names(score))
  counts <- aggregate(
    innovations$innovation,
    by = list(unit = innovations$unit, vendor = innovations$vendor, regime = innovations$regime),
    FUN = length
  )
  names(counts)[4] <- "n"
  score <- merge(score, counts, by = c("unit", "vendor", "regime"), all.x = TRUE)

  list(params = params, latent = latent, innovations = innovations, score = score, convergence = opt$convergence)
}

units <- sort(unique(prices[[unit_col]]))
units <- units[seq_len(min(length(units), max_units))]

param_all <- list()
latent_all <- list()
innov_all <- list()
score_all <- list()
fit_status <- data.frame(unit = character(), status = character(), stringsAsFactors = FALSE)

for (u in units) {
  cat("Fitting", unit_col, u, "\n")
  res <- tryCatch(fit_one_unit(u), error = function(e) {
    fit_status <<- rbind(fit_status, data.frame(unit = as.character(u), status = paste("error:", e$message)))
    NULL
  })
  if (is.null(res)) {
    fit_status <- rbind(fit_status, data.frame(unit = as.character(u), status = "skipped_insufficient_data"))
    next
  }
  param_all[[length(param_all) + 1]] <- res$params
  latent_all[[length(latent_all) + 1]] <- res$latent
  innov_all[[length(innov_all) + 1]] <- res$innovations
  score_all[[length(score_all) + 1]] <- res$score
  fit_status <- rbind(fit_status, data.frame(unit = as.character(u), status = paste0("fit_convergence_", res$convergence)))
}

tables_dir <- resolve_path(cfg$output$tables_dir)
dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)

write_or_empty <- function(obj_list, file, cols) {
  out <- if (length(obj_list) > 0) do.call(rbind, obj_list) else as.data.frame(setNames(replicate(length(cols), character(0), simplify = FALSE), cols))
  write.csv(out, file.path(tables_dir, file), row.names = FALSE)
}

write_or_empty(param_all, "state_space_parameter_estimates.csv", c("unit", "parameter", "estimate"))
write_or_empty(latent_all, "state_space_filtered_latent_price.csv", c("unit", "date", "filtered_latent_price", "predicted_latent_price"))
write_or_empty(innov_all, "state_space_innovations.csv", c("unit", "date", "vendor", "regime", "innovation", "innovation_sd", "standardized_innovation"))
write_or_empty(score_all, "state_space_vendor_residual_scorecard.csv", character(0))
write.csv(fit_status, file.path(tables_dir, "state_space_fit_status.csv"), row.names = FALSE)

cat("Wrote state-space outputs to", tables_dir, "\n")
