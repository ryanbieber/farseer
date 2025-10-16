// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

functions {
  matrix get_changepoint_matrix(vector t, vector t_change, int T, int S) {
    // Assumes t and t_change are sorted.
    matrix[T, S] A;
    row_vector[S] a_row;
    int cp_idx;

    // Start with an empty matrix.
    A = rep_matrix(0, T, S);
    a_row = rep_row_vector(0, S);
    cp_idx = 1;

    // Fill in each row of A.
    for (i in 1:T) {
      while ((cp_idx <= S) && (t[i] >= t_change[cp_idx])) {
        a_row[cp_idx] = 1;
        cp_idx = cp_idx + 1;
      }
      A[i] = a_row;
    }
    return A;
  }

  // Logistic trend functions

  vector logistic_gamma(real k, real m, vector delta, vector t_change, int S) {
    vector[S] gamma;  // adjusted offsets, for piecewise continuity
    vector[S + 1] k_s;  // actual rate in each segment
    real m_pr;

    // Compute the rate in each segment
    k_s = append_row(k, k + cumulative_sum(delta));

    // Piecewise offsets
    m_pr = m; // The offset in the previous segment
    for (i in 1:S) {
      gamma[i] = (t_change[i] - m_pr) * (1 - k_s[i] / k_s[i + 1]);
      m_pr = m_pr + gamma[i];  // update for the next segment
    }
    return gamma;
  }

  vector logistic_trend(
    real k,
    real m,
    vector delta,
    vector t,
    vector cap,
    matrix A,
    vector t_change,
    int S
  ) {
    vector[S] gamma;

    gamma = logistic_gamma(k, m, delta, t_change, S);
    return cap .* inv_logit((k + A * delta) .* (t - (m + A * gamma)));
  }

  // Linear trend function

  vector linear_trend(
    real k,
    real m,
    vector delta,
    vector t,
    matrix A,
    vector t_change
  ) {
    return (k + A * delta) .* t + (m + A * (-t_change .* delta));
  }

  // Flat trend function

  vector flat_trend(
    real m,
    int T
  ) {
    return rep_vector(m, T);
  }

  // Partial sum function for reduce_sum
  real partial_sum(
    array[] int slice_n,
    int start,
    int end,
    vector y,
    matrix X_sa,
    matrix X_sm,
    vector trend,
    vector beta,
    real sigma_obs,
    vector weights
  ) {
    real lp = 0;
    for (i in start:end) {
      int n = slice_n[i - start + 1];
      real mu = dot_product(X_sa[n], beta) + trend[n] * (1 + dot_product(X_sm[n], beta));
      lp += weights[n] * normal_lpdf(y[n] | mu, sigma_obs);
    }
    return lp;
  }
}

data {
  int T;                // Number of time periods
  int<lower=0> K;       // Number of regressors (can be 0)
  vector[T] t;          // Time
  vector[T] cap;        // Capacities for logistic trend
  vector[T] y;          // Time series
  int S;                // Number of changepoints
  vector[S] t_change;   // Times of trend changepoints
  matrix[T,K] X;        // Regressors
  vector<lower=0>[K] sigmas;     // Scale on seasonality prior (must be positive)
  real<lower=0> tau;    // Scale on changepoints prior
  int trend_indicator;  // 0 for linear, 1 for logistic, 2 for flat
  vector[K] s_a;        // Indicator of additive features
  vector[K] s_m;        // Indicator of multiplicative features
  vector<lower=0>[T] weights;  // Observation weights
  int<lower=1> grainsize;  // Grainsize for reduce_sum (e.g., 1)
}

transformed data {
  matrix[T, S] A = get_changepoint_matrix(t, t_change, T, S);
  matrix[T, K] X_sa = X .* rep_matrix(s_a', T);
  matrix[T, K] X_sm = X .* rep_matrix(s_m', T);
  array[T] int n_seq;

  // Create sequence of indices for reduce_sum
  for (n in 1:T) {
    n_seq[n] = n;
  }
}

parameters {
  real k;                   // Base trend growth rate
  real m;                   // Trend offset
  vector[S] delta;          // Trend rate adjustments
  real<lower=0> sigma_obs;  // Observation noise
  vector[K] beta;           // Regressor coefficients
}

transformed parameters {
  vector[T] trend;
  if (trend_indicator == 0) {
    trend = linear_trend(k, m, delta, t, A, t_change);
  } else if (trend_indicator == 1) {
    trend = logistic_trend(k, m, delta, t, cap, A, t_change, S);
  } else if (trend_indicator == 2) {
    trend = flat_trend(m, T);
  }
}

model {
  //priors
  k ~ normal(0, 5);
  m ~ normal(0, 5);
  delta ~ double_exponential(0, tau);
  sigma_obs ~ normal(0, 0.5);

  // Only apply beta prior if there are regressors
  if (K > 0) {
    for (i in 1:K) {
      // Ensure sigma is positive, use max to avoid zero scale
      beta[i] ~ normal(0, fmax(sigmas[i], 0.01));
    }
  }

  // Likelihood with reduce_sum for multithreading
  target += reduce_sum(
    partial_sum,
    n_seq,
    grainsize,
    y,
    X_sa,
    X_sm,
    trend,
    beta,
    sigma_obs,
    weights
  );
}
