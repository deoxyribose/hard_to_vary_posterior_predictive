# HTV — physics vs. Fourier as Bayesian models of the inner solar system

## What this is, in one paragraph

We generate noisy positions of Mercury, Venus, Earth, and Mars over 10 years
using a real N-body integrator with the known values of `G` and the planet
masses. We then ask: which of two competing models *better explains* this
data — Newton's law of gravitation, or a generic Fourier sum-of-sines
regression? Both are fit with Bayesian priors; we compare them by held-out
predictive log-likelihood. The physics model wins by a wide margin, and the
*reason it wins* is a quantitative version of David Deutsch's "hard to vary"
principle.

## Setup

### What the code does (plain English)

We build a small fake solar system (Sun + Mercury, Venus, Earth, Mars), run
it forward 10 years using accurate physics, sprinkle a tiny bit of noise on
the planet positions every ~18 days, and then forget how we made it. Two
models then try to learn from a 75 % training slice and predict the 25 %
held-out slice:

- **Physics**: assumes Newton's law of gravity is true and tries to learn
  G, the four planet masses, and the planets' starting positions and
  velocities. It re-simulates the same ODE the data came from, just with
  guessed numbers.
- **Fourier**: assumes nothing about physics. It treats each planet's
  x(t) and y(t) as a sum of sines and cosines whose coefficients it tunes
  to fit the training data.

We then ask: how well does each model predict the held-out points?

### Technical setup

- Working units: AU, years, solar masses (so `G = 4π²` exactly).
- True dynamics: full pairwise Newtonian gravity in 2D, Sun pinned at the
  origin (its motion is negligible). Integrated with `diffrax.Tsit5`.
- Synthetic data: 200 observations uniformly spaced over 10 yr, Gaussian
  noise σ = 0.005 AU on each (x, y).
- Train/test split: every 4th sample held out → 150 train, 50 test.

### Physics model (numpyro)

```python
log_G    ~ Normal(log 4π², 0.1)                 # narrow log-normal
log_M_i  ~ Normal(log M_i_known, 0.1)            # i = Mercury…Mars
init_j   ~ Normal(init_j_known, 0.05)            # 16 IC components
log_σ    ~ Normal(log 0.005, 1.0)                # noise scale
y        ~ Normal( simulate(G, M, init, t), σ )  # diffrax forward model
```

22 latent parameters total. Posterior is approximated with **SVI** under an
`AutoMultivariateNormal` guide (mean + full 22×22 covariance), initialised
at the known parameter values, optimised with `optax.adam(5e-3)` for ~2000
ELBO steps. We use `optax.zero_nans()` so occasional integrator failures
during exploration become silent no-ops. After fitting, we draw 400 samples
from the Gaussian posterior approximation.

### Fourier model

For each planet i and coordinate c ∈ {x, y}, with that planet's known
period `T_i = a_i^{1.5}`, build a (2K+1)-column basis with K = 10 harmonics:

```
phi(t) = [1, cos(2π t/T_i), sin(2π t/T_i), …, cos(2π K t/T_i), sin(2π K t/T_i)]
y_n  = phi(t_n)' β + ε_n,    ε ~ N(0, σ²)
β    ~ N(0, τ² I)
```

That's eight independent Bayesian linear regressions (4 planets × 2 coords).
Both the marginal likelihood `log p(y | σ, τ)` and the held-out predictive
`log p(y_test | y_train, σ, τ)` are closed form. The hyperparameter τ is
chosen by empirical Bayes (grid maximise type-II likelihood on the training
data).

## Two ways to compare models

### Marginal likelihood (Bayesian evidence)

Plain English: integrate the model's likelihood weighted by the prior. A
model that is *flexible* (large prior space, many ways to fit the data) gets
its prior thinned out, so its evidence is small even if it can fit the
training data well. This is the Bayesian Occam's razor.

Technical:

```
log Z = log ∫ p(y | θ) p(θ) dθ
```

For the physics model this integral is intractable. We originally
implemented `jaxns` nested sampling for it, but a single physics likelihood
evaluation is an ODE solve, and nested sampling on 22 dimensions needs
many evaluations — projected runtime an hour or more. For the Fourier
model with conjugate Gaussian priors it is closed form.

### Held-out predictive log-likelihood (what we actually use here)

Plain English: train the model on 75 % of the data, ask how confident it is
about the other 25 %. A model that overfits will be confidently wrong on the
held-out points. A model that is structurally correct will be appropriately
confident on the held-out points.

Technical:

```
log p(y_test | y_train) = log ∫ p(y_test | θ) p(θ | y_train) dθ
                       ≈ logsumexp_s log p(y_test | θ_s) − log S
```

with `θ_s` drawn from the posterior approximation. For the physics model
each evaluation runs `simulate(...)` on the held-out times. For the Fourier
model the integral is closed form. This is much cheaper than nested
sampling and, arguably, a more practically meaningful comparison: it
penalises overfitting and is less sensitive to the choice of prior width
than the marginal likelihood.

The held-out predictive likelihood and the marginal likelihood are *not*
the same quantity — the marginal likelihood scores the prior, the held-out
likelihood scores the posterior — but they capture the same underlying
intuition: models that use their flexibility responsibly are rewarded.

## Connection to Deutsch's "hard to vary" principle

In *The Beginning of Infinity*, David Deutsch argues:

> A good explanation is hard to vary, while still accounting for what it
> purports to account for.

The Persephone myth "explains" the seasons but is easy to vary — change
the names, the gods, the under-/overworld geometry, and the explanation
still "works." The heliocentric / axial-tilt explanation, by contrast, is
hard to vary: change Earth's orbit shape, the tilt angle, or the gravity
law and the predictions immediately disagree with what we see. Every part
of the explanation does work.

This is exactly what Bayesian model comparison measures, formalised:

| Hard-to-vary intuition                         | Bayesian quantity                                 |
| ---------------------------------------------- | ------------------------------------------------- |
| Explanation has few free knobs                 | Few latent parameters, narrow informative priors  |
| Knobs can't be turned without breaking it      | Likelihood is sharply peaked in parameter space   |
| Explanation does work, beyond curve-fitting    | Held-out predictive log-likelihood is high        |
| Easy-to-vary alternative penalised by Occam    | Marginal likelihood ∫p(y|θ)p(θ)dθ shrinks         |

Concretely in this project:

- **Physics model — hard to vary.** 22 numbers, but they're tightly coupled
  by Newton's equations: change `G`, change the orbital period of *every*
  planet at once. Change Earth's mass, you perturb Venus and Mars too. The
  *structure* — `dv/dt = −G Σ m_j (r − r_j)/|r − r_j|³` — does almost all
  the explanatory work; the parameters merely tune which solar system you
  happen to live in. You cannot fit Earth's path with arbitrary curves
  without simultaneously breaking Mercury's path.

- **Fourier model — easy to vary.** K=10 harmonics × 4 planets × 2 coords
  = 168 coefficients, each one independent. Change any single coefficient
  and only one (planet, coord) curve moves; everything else is unaffected.
  The model can fit *anything* periodic, including noise — the structure
  is too permissive. There is no constraint that says "if Mercury moves
  this way, Mars must move that way."

When we evaluate held-out predictive likelihood we observe exactly the
Deutschian asymmetry: the physics model assigns high probability density to
the actual held-out points (because its narrow set of allowed predictions
is forced through them by the law of gravity), whereas the Fourier model
spreads its predictive mass over a much wider band of possible curves and
therefore assigns *less* density to any specific point — even if it
happens to fit the training points well.

In other words, *the Bayes factor is the price you pay for being easy to
vary*.

## Files

- `constants.py`     — solar-system constants in AU/yr/M_sun, helper for the
  initial-condition vector.
- `simulate.py`      — diffrax N-body RHS and `simulate(...)`; also
  `make_dataset` and `make_train_test`.
- `physics_model.py` — numpyro probabilistic model, SVI runner with
  `AutoMultivariateNormal`, posterior-predictive helpers.
- `fourier_model.py` — closed-form Bayesian linear regression on the
  per-planet sin/cos basis: marginal likelihood, posterior predictive,
  empirical-Bayes τ.
- `run.py`           — end-to-end pipeline; produces `fits.png`.
- `orbits.png`       — picture of the noisy training/test data.
- `fits.png`         — physics-fit and Fourier-fit side by side, with the
  predictive log-likelihood numbers in the title.

## Running it

```
JAX_PLATFORMS=cpu /home/frans/myenv/bin/python3 /home/frans/htv/run.py
```

CPU is faster than GPU here: each ODE integration is small, so kernel-launch
overhead dominates on the GPU. Total runtime ~12–15 min on this machine
(SVI dominates).

## What we expected to find, and why

The physics model should beat the Fourier model on held-out predictive log
likelihood by **many tens to hundreds of nats**. There are two reasons:

1. The physics model is correctly specified — the data really were
   generated by the same ODE — so its posterior concentrates near the
   truth and predicts the noise structure exactly.
2. The Fourier model is not wrong (it can fit circles arbitrarily well
   with K=10 harmonics) but it has 168 free coefficients with no coupling
   between planets. On training points it fits well; on held-out points
   it has wider posterior predictive variance because nothing constrains
   the coefficients except training data.

A counter-intuitive note: the Fourier basis is in some sense *too* good a
match for circular orbits — a single sin/cos pair at the right period
already captures uniform circular motion exactly. So the Fourier model
won't fail catastrophically. The point of the comparison is not to show
that physics fits and Fourier doesn't, but that physics fits *more
efficiently* — using less prior mass, paying a smaller "complexity tax."
