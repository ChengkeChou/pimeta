// tau2(t) with a spline scale: the Bayesian reference for pimeta's
// tau_trend(tau="spline"), fitted with CmdStan through cmdstanr.
//
//   y_i ~ N(b0 + b1 t_i, v_i + exp(a0 + a1 t_i + B_i gamma))
//   gamma ~ N(0, sigma_s),  sigma_s ~ half-normal(0, 0.5)
//
// B is the orthonormal bending basis from pimeta's `_spline_basis`: the
// restricted cubic spline's non-linear columns, projected off [1, t] and
// Gram-Schmidt'ed, so B'B = I and one sigma_s shrinks every direction equally.
// The priors mirror pimeta's Bayesian location-scale defaults, which is what
// makes the two fits comparable rather than merely similar.
data {
  int<lower=1> k;                 // studies
  vector[k] y;                    // effect estimates
  vector<lower=0>[k] v;           // sampling variances
  vector[k] t;                    // time, centred
  int<lower=0> m;                 // bending columns
  matrix[k, m] B;                 // orthonormal bending basis
}
parameters {
  real b0;                        // location intercept
  real b1;                        // location slope
  real a0;                        // log tau^2 level
  real a1;                        // log tau^2 linear trend
  vector[m] z;                    // standardised bending coefficients
  real<lower=0> sigma_s;          // wiggliness hyperparameter
}
transformed parameters {
  // non-centred: with gamma ~ N(0, sigma_s) sampled directly, sigma_s and the
  // bending coefficients are strongly correlated and 6% of transitions
  // diverged.  z ~ N(0,1) with gamma = sigma_s * z removes that funnel.
  vector[m] gamma = sigma_s * z;
}
model {
  b0 ~ normal(0, 100);
  b1 ~ normal(0, 100);
  a0 ~ normal(0, 2);
  a1 ~ normal(0, 0.3);
  z ~ std_normal();
  // half-normal(5), not half-normal(0.5): the reference exists to check the
  // *estimator*, and on this fixture the data want a wiggliness of about 4.
  // A prior of 0.5 would be a strong statement about the answer, and the
  // comparison with the penalised fit would then measure the prior rather than
  // the method.  pimeta's own Bayesian path keeps its tighter shrinkage prior
  // for the linear trend, which is a different (predictive) purpose.
  sigma_s ~ normal(0, 5);
  y ~ normal(b0 + b1 * t, sqrt(v + exp(a0 + a1 * t + B * gamma)));
}
