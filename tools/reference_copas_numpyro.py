"""Independent NUTS reference for the Bayesian Copas posterior.

Run with an interpreter that has ``numpyro`` and ``jax`` — the base Anaconda
environment here does — while pimeta itself needs neither::

    D:\\Users\\dell\\anaconda3\\python.exe tools\\reference_copas_numpyro.py

What this is and is not.  The *model* is the same one `pimeta.copas` states —
there is only one Copas likelihood — so agreement here does not validate the
formula.  What it validates is everything else: the parameterisation and its
Jacobians, the priors, and the sampler.  A wrong Jacobian or a mis-specified
prior is a systematic shift that two samplers on the same wrong target would
share and this comparison would not catch either; that is why
``tests/test_copas_bayes.py`` pins the likelihood against the conditional MLE
and the generator against the likelihood first, and this script only then asks
whether the posterior was sampled correctly.

The comparison is a z-score per parameter,
``|mean_pimeta - mean_numpyro| / sqrt(se_pimeta^2 + se_numpyro^2)`` with the
standard errors taken from the effective sample sizes.  Two correct samplers
differ by a few tenths; a z above 3 means one of them is not sampling the
posterior it claims.

Exit code is 0 when every parameter agrees and 1 otherwise, so this can be put
in a CI job that has numpyro available.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np


def simulate(*, mu=0.3, tau=0.2, gamma0=0.0, gamma1=0.6, rho=0.6, k=24,
             seed=11):
    """The same generator the test module uses, so the data are identical."""
    rng = np.random.default_rng(seed)
    sei = np.exp(rng.uniform(np.log(0.05), np.log(0.45), k))
    theta = rng.normal(mu, tau, k)
    y = theta + rng.normal(0.0, 1.0, k) * sei
    total = np.sqrt(sei ** 2 + tau ** 2)
    score = (gamma0 + gamma1 / sei + rho * (y - mu) / total
             + np.sqrt(1.0 - rho ** 2) * rng.normal(0.0, 1.0, k))
    keep = score > 0.0
    return y[keep], sei[keep]


def pimeta_summary(yi, sei, gamma):
    from pimeta.copas import bayes_copas

    fit = bayes_copas(yi, sei=sei, gamma=gamma,
                      mcmc={"n_draws": 4000, "warmup": 2000, "chains": 4,
                            "seed": 20261024, "check": False})
    rows = fit.posterior.summary()
    return {r["term"]: (r["mean"], r["sd"], r["ess"]) for r in rows}


def numpyro_summary(yi, sei, gamma, *, seed=20261024):
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.diagnostics import summary as nuts_summary
    from numpyro.infer import MCMC, NUTS

    jax.config.update("jax_enable_x64", True)
    numpyro.set_host_device_count(4)
    yi = jnp.asarray(yi, dtype=jnp.float64)
    sei = jnp.asarray(sei, dtype=jnp.float64)

    def model():
        mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
        rho = numpyro.sample("rho", dist.Uniform(-1.0, 1.0))
        tau = numpyro.sample("tau", dist.HalfNormal(1.0))
        if gamma is None:
            g0 = numpyro.sample("gamma0", dist.Normal(0.0, 1.0))
            g1 = numpyro.sample("gamma1", dist.HalfNormal(1.0))
        else:
            g0 = jnp.float64(gamma[0])
            g1 = jnp.float64(gamma[1])
        u = g0 + g1 / sei
        lam = jnp.exp(dist.Normal(0.0, 1.0).log_prob(u)
                      - jax.scipy.special.log_ndtr(u))
        lam = jnp.clip(lam, 0.0, 1e300)
        denom = 1.0 - rho * rho * lam * (u + lam)
        numpyro.factor("denom", jnp.sum(jnp.where(denom > 0.0, 0.0, -1e9)))
        sigma = jnp.sqrt(sei ** 2 / jnp.clip(denom, 1e-12, None))
        s2t2 = sigma * sigma + tau * tau
        rho_t = rho * sigma / jnp.sqrt(s2t2)
        v = (u + rho_t * (yi - mu) / jnp.sqrt(s2t2)) / jnp.sqrt(1.0 - rho_t ** 2)
        ll = (0.5 * jnp.log(s2t2) + (yi - mu) ** 2 / (2.0 * s2t2)
              - jax.scipy.special.log_ndtr(jnp.maximum(v, -37.0)))
        numpyro.factor("copas", -jnp.sum(ll))

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=2000, num_samples=4000, num_chains=4,
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    per_chain = {k: np.asarray(v) for k, v in
                 mcmc.get_samples(group_by_chain=True).items()}
    stats = nuts_summary(per_chain)
    flat = {k: v.reshape(-1) for k, v in per_chain.items()}

    def entry(name, values, ess_from):
        return (float(np.mean(values)), float(np.std(values, ddof=1)),
                float(stats[ess_from]["n_eff"]))

    out = {"mu": entry("mu", flat["mu"], "mu"),
           "rho": entry("rho", flat["rho"], "rho"),
           "tau": entry("tau", flat["tau"], "tau"),
           "psi(rho)": entry("psi", np.arctanh(flat["rho"]), "rho"),
           "log_tau": entry("log_tau", np.log(flat["tau"]), "tau")}
    if gamma is None:
        out["gamma0"] = entry("gamma0", flat["gamma0"], "gamma0")
        out["log_gamma1"] = entry("log_gamma1", np.log(flat["gamma1"]),
                                  "gamma1")
    return out


def compare(label: str, left: dict, right: dict) -> bool:
    print(f"\n=== {label} ===")
    print(f"{'parameter':<12}{'pimeta':>26}{'numpyro NUTS':>26}{'z':>8}")
    worst = 0.0
    ok = True
    for name in left:
        if name not in right:
            continue
        m1, s1, e1 = left[name]
        m2, s2, e2 = right[name]
        se = np.sqrt(s1 ** 2 / max(e1, 1.0) + s2 ** 2 / max(e2, 1.0))
        z = abs(m1 - m2) / max(se, 1e-12)
        worst = max(worst, z)
        if z > 3.0:
            ok = False
        print(f"{name:<12}{m1:>12.5f} (sd {s1:5.3f})"
              f"{m2:>12.5f} (sd {s2:5.3f}){z:>8.2f}")
    print(f"{'worst z':<12}{worst:>52.2f}")
    return ok


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    yi, sei = simulate(seed=7, k=20)
    print(f"{yi.size} studies selected; sei from "
          f"{sei.min():.3f} to {sei.max():.3f}")
    good = True
    for label, gamma in (("selection parameters fixed at (0, 1)", (0.0, 1.0)),
                         ("selection parameters estimated", None)):
        left = pimeta_summary(yi, sei, gamma)
        right = numpyro_summary(yi, sei, gamma)
        good &= compare(label, left, right)
    print("\nRESULT:", "agree within Monte Carlo error"
          if good else "DISAGREEMENT")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
