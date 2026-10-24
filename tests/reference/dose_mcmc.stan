// Dose-response NMA with a random effect on the curve's parameters.
// Reference implementation for pimeta.nma_dose(inference="mcmc"), written for
// rstan 2.32.7.  The model, in words: each study has its own Emax and log ED50
// for each treatment, drawn around the pooled values, and the observed contrasts
// are normal with the known within-study variance.
data {
  int<lower=1> K;                 // contrasts
  int<lower=1> S;                 // studies
  int<lower=1> P;                 // parameters per curve (2: emax, log ed50)
  int<lower=1> B;                 // curves (treatments with a nonzero dose)
  array[K] int<lower=1, upper=S> study;
  array[K] int<lower=-1, upper=B> blk1;   // -1 when the arm has no curve
  array[K] int<lower=-1, upper=B> blk2;
  vector[K] y;
  vector<lower=0>[K] vi;
  vector[K] dose1;
  vector[K] dose2;
  real<lower=0> prior_beta;
  real<lower=0> prior_het;
}
parameters {
  matrix[S, B * P] z;             // non-centred study deviations
  vector[B * P] beta;             // pooled curve parameters
  vector<lower=0>[B * P] sd;      // between-study SD of each parameter
}
transformed parameters {
  matrix[S, B * P] per_study;
  for (s in 1:S)
    for (j in 1:(B * P))
      per_study[s, j] = beta[j] + z[s, j] * sd[j];
}
model {
  beta ~ normal(0, prior_beta);
  sd ~ normal(0, prior_het);        // half-normal by the lower bound
  to_vector(z) ~ std_normal();
  for (k in 1:K) {
    real mu = 0;
    if (blk2[k] > 0) {
      int b = blk2[k];
      real e = per_study[study[k], (b - 1) * P + 1];
      real ed = exp(per_study[study[k], (b - 1) * P + 2]);
      mu += e * dose2[k] / (ed + dose2[k]);
    }
    if (blk1[k] > 0) {
      int b = blk1[k];
      real e = per_study[study[k], (b - 1) * P + 1];
      real ed = exp(per_study[study[k], (b - 1) * P + 2]);
      mu -= e * dose1[k] / (ed + dose1[k]);
    }
    target += normal_lpdf(y[k] | mu, sqrt(vi[k]));
  }
}
