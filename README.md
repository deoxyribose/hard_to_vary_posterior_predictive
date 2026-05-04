# HTV — physics vs. Fourier as Bayesian models of the inner solar system

## What this is, in one paragraph

We generate noisy positions of Mercury, Venus, Earth, and Mars over 10 years
using a real N-body integrator with the known values of `G` and the planet
masses. We then ask: which of two competing models *better explains* this
data — Newton's law of gravitation, or a generic Fourier sum-of-sines
regression? Both are fit with Bayesian priors; we compare them by held-out
predictive log-likelihood. Physics wins (by ~16 nats on this dataset), and
the *reason it wins* is a quantitative version of David Deutsch's "hard to
vary" principle: physics has 6 tightly coupled knobs vs. Fourier's 168
independent ones, so physics pays a smaller flexibility tax.

## Setup

### What the code does (plain English)

We build a small fake solar system (Sun + Mercury, Venus, Earth, Mars), run
it forward 10 years using accurate physics, sprinkle a tiny bit of noise on
the planet positions every ~18 days, and then forget how we made it. Two
models then try to learn from a 75 % training slice and predict the 25 %
held-out slice:

- **Physics**: assumes Newton's law of gravity is true and tries to learn
  G, the four planet masses, and the observation noise scale. It
  re-simulates the same ODE the data came from, just with guessed numbers.
  Initial conditions are taken as known (a fair concession — for real
  astronomy you measure them).
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
log_G    ~ Normal(log 4π², 0.1)                       # narrow log-normal
log_M_i  ~ Normal(log M_i_known, 0.1)                  # i = Mercury…Mars
log_σ    ~ Normal(log 0.005, 1.0)                      # noise scale
y        ~ Normal( simulate(G, M, INIT_KNOWN, t), σ )  # diffrax forward model
```

**Six latent parameters** (`log_G`, four `log_M`, `log_σ`). Initial conditions
are fixed at their known circular values; making them latent costs 16 extra
weakly-constrained dimensions and turned out to be unnecessary for the
inference task here.

Inference is two-stage:

1. **MAP via L-BFGS** on the flattened parameter vector
   (`scipy.optimize.minimize(method="L-BFGS-B", jac=True)` with JAX-jitted
   `value_and_grad`). Converges in 2 iterations from the known-truth init,
   with final `|grad|_∞ ≈ 8` (the irreducible noise gradient). Adam fails
   here: ∂loss/∂log_G is ~10⁵ at truth while ∂loss/∂log_M ≈ 5, and Adam's
   first-step bias correction takes a step of size `lr·sign(grad)` regardless
   of magnitude — it always overshoots the steep direction. Quasi-Newton
   handles the 5-OOM ill-conditioning natively.
2. **SVI** on top, with an `AutoMultivariateNormal` guide (mean + full 6×6
   covariance) initialised at the L-BFGS MAP with `init_scale=1e-3`.
   Optimised with `optax.chain(zero_nans, clip_by_global_norm(1.0), adam(1e-4))`
   for 1500 ELBO steps. The ELBO is estimated with
   `Trace_ELBO(num_particles=8, vectorize_particles=True)` — eight Monte
   Carlo samples per step, vmapped through the integrator. A single MC
   sample makes ELBO too noisy because rare guide draws into ODE-stiff
   regions yield catastrophic log p(y|θ); averaging over particles damps
   that out at ~linear extra cost. After fitting, we draw 200 samples from
   the Gaussian posterior approximation.

Result on this dataset: posterior recovers `log_G` to 0.16 σ_post of truth,
masses to within ~3 σ_post (≪ 1 prior σ in absolute terms), and noise scale
σ ≈ 0.0053 (true 0.005, ~5 % inflated).

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
evaluation is an ODE solve and nested sampling on even 6 dimensions needs
many evaluations. For the Fourier model with conjugate Gaussian priors it
is closed form.

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

- **Physics model — hard to vary.** Six numbers, all of them tightly coupled
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

`hard_to_vary.png` makes this concrete: in the left panel we shift `log_G`
by ±1 and ±2 prior σ — every planet's orbit warps coherently because the
same `G` sets every period. In the right panel we shift a *single* Fourier
coefficient by the same ±1, ±2 prior σ — only Earth's x-curve moves; the
other seven (planet, coord) curves are untouched. Same number of "knob
twiddles," vastly different consequences. Physics has 6 knobs that each
move 8 curves; Fourier has 168 knobs that each move 1 curve.

`hard_to_vary.gif` is the same idea sweeping continuously across ±2 prior σ
— in the left panel all four orbits breathe in and out together; in the
right panel only Earth's x-curve moves while the other three planets stand
still.

When we evaluate held-out predictive likelihood we observe exactly the
Deutschian asymmetry: the physics model assigns high probability density to
the actual held-out points (because its narrow set of allowed predictions
is forced through them by the law of gravity), whereas the Fourier model
spreads its predictive mass over a wider band of possible curves and
therefore assigns less density to any specific point — even though it
happens to fit the training points fine.

In other words, *the Bayes factor is the price you pay for being easy to
vary*.

## Files

- `constants.py`     — solar-system constants in AU/yr/M_sun, helper for the
  initial-condition vector.
- `simulate.py`      — diffrax N-body RHS and `simulate(...)`; also
  `make_dataset` and `make_train_test`.
- `physics_model.py` — numpyro probabilistic model, L-BFGS MAP, SVI runner
  with `AutoMultivariateNormal`, posterior-predictive helpers.
- `fourier_model.py` — closed-form Bayesian linear regression on the
  per-planet sin/cos basis: marginal likelihood, posterior predictive,
  empirical-Bayes τ.
- `run.py`           — end-to-end pipeline; produces `fits.png`. Caches SVI
  samples to `svi_samples.pkl` (delete to re-fit).
- `plot_hard_to_vary.py` — perturbation visualisation; produces
  `hard_to_vary.png`. Loads cached samples; cheap to re-run.
- `gif_hard_to_vary.py`  — animated version (continuous knob sweep);
  produces `hard_to_vary.gif`.
- `orbits.png`       — picture of the noisy training/test data.
- `fits.png`         — physics-fit and Fourier-fit side by side, with the
  predictive log-likelihood numbers in the title.
- `hard_to_vary.png` — static visualisation of the per-knob coupling asymmetry.
- `hard_to_vary.gif` — continuous-sweep animation of the same.

## Running it

```
JAX_PLATFORMS=cpu /home/frans/myenv/bin/python3 /home/frans/htv/run.py
```

CPU is faster than GPU here: each ODE integration is small, so kernel-launch
overhead dominates on the GPU. Total runtime ~7–8 min (L-BFGS MAP ~5 s,
8-particle vectorised SVI ~7 min, Fourier closed-form negligible).

After running once, `plot_hard_to_vary.py` reads the cached samples and
regenerates `hard_to_vary.png` in seconds.

## Result

```
log p(y_test | y_train, physics) = +1546
log p(y_test | y_train, fourier) = +1530
                            diff = +16  → physics decisively preferred
```

The asymptotic ceiling for physics on this dataset is ≈ +1551 (see below);
we land at +1546 because `log_σ` is recovered ~5 % high, costing a handful
of nats. So we're effectively at the optimum — there is not much more for
the physics model to extract.

## What we expected, what we found, why the gap is smaller than naively expected

We initially expected physics to win by **tens to hundreds of nats**. The
actual margin is +16. Both directions of the surprise are real:

1. The physics model *is* correctly specified, *does* concentrate near the
   truth, and *does* predict the noise structure exactly. Its predictive
   log density per held-out point is ≈ 3.87 nats — about 0.02 nats below
   the theoretical ceiling −log(σ_noise·√(2π)) − ½ ≈ 3.88.
2. The Fourier model is not far behind: it gets ≈ 3.82 nats per point.
   With K=10 harmonics on circular orbits, a single (cos, sin) pair at the
   right period already nails uniform circular motion exactly. The
   remaining 19 harmonics have small posterior mass (empirical-Bayes
   τ\* ≈ 0.22 keeps them constrained). So the Fourier model is *not* very
   easy to vary in practice — the data tell it which harmonics matter.

The 16-nat margin is therefore the small, real cost the Fourier model pays
for its 168 independent knobs vs. physics's 6 coupled ones, on a problem
where Fourier is structurally well suited. On a less-Fourier-friendly
problem (eccentric or chaotic orbits, planet-planet interactions visible
in the data) the gap should widen substantially.

The deeper Deutschian point still stands and is shown directly in
`hard_to_vary.png`: a single physics knob warps every planet's orbit at
once, while a single Fourier knob moves only one (planet, coord) curve.
That asymmetry is what *would* dominate on harder problems. Here both
models fit well, and the held-out predictive simply scores how *efficiently*
each one does so.

## Are the predictive estimates accurate?

Worth checking, since approximations could distort the +16 nat margin. Net
answer: the Fourier number is exact, the physics number is *slightly*
underestimated (~5 nats) by SVI's residual `log_σ` inflation, so the true
gap is closer to ~+20 nats than +16. The qualitative conclusion is robust;
the variational approximation is mildly *against* physics, not for it.

- **Fourier.** Conjugate Gaussian prior + Gaussian likelihood → posterior
  predictive is closed-form Gaussian; `log p(y_test | y_train)` is
  analytic. (One small concession: τ\* = 0.224 is empirical-Bayes — it
  optimises log-evidence on the train set — which is *favourable* to
  Fourier vs. a flat-prior Bayesian.)
- **Physics, layer 1: MC error.** Estimator
  `logsumexp_s log p(y_test | θ_s) − log S` with S=200. The guide is very
  narrow (`log_G` posterior std = 9·10⁻⁴), so all θ_s give nearly
  identical trajectories and log p_s are nearly identical. Logsumexp
  variance ≈ var(p_s)/S, negligible. Bumping S to 2000 would change the
  answer by ≪ 1 nat.
- **Physics, layer 2: variational gap.** SVI minimises KL(q‖p), which
  systematically underestimates posterior spread. Doesn't bite here:
  1200 noisy positions make the true posterior tight, so a Gaussian guide
  around the MAP is a good approximation locally. The dominant remaining
  bias is `log_σ` recovered ~5 % high. The arithmetic:

  ```
  per-point ceiling at σ=0.005:    −log(0.005·√2π) − ½              = 3.882 nat
  per-point ceiling at σ=0.0053:   −log(0.0053·√2π) − ½(0.005/0.0053)² = 3.878 nat
  per-point measured physics:                                          ≈ 3.865 nat
  ```

  So the SVI predictive is 0.013 nat/point below what its own σ should
  yield, and 0.017 nat/point below the perfect-σ ceiling. Times 400
  held-out dimensions, that's ~5 and ~7 nats respectively — i.e. with a
  longer / better-converged SVI run, physics would land at +1551–+1553
  and the gap would widen to ~+20–23 nats.

- **What would distort the comparison?** Running SVI with the failure
  modes we hit earlier — `log_σ` inflated 25× to absorb a bad fit, or the
  guide collapsing to a wide variance — would tank physics's predictive.
  Once SVI is converged at the L-BFGS MAP, neither of these is happening.

If you want to verify by a different route: replacing SVI with NUTS
(`numpyro.infer.MCMC(NUTS(...))`) draws from the true posterior and the
predictive should land within ≈ 1 nat of the SVI estimate. Slow (maybe an
hour with the ODE in the inner loop), but principled.

## What it took to get the physics model to fit (and what broke first)

Two non-obvious failure modes worth recording:

1. **`simulate(...)` was treating `init_state` as the state at `t=ts[0]`,
   not at `t=0`.** During inference `ts = ts_train` starts at `t ≈ 0.05`
   (we hold out every fourth sample, beginning with index 0). Feeding the
   `t=0` state to a `t=0.05` integrator entry mismatched all four planets
   by their ~18-day phase — at the *true* parameters, RMS residual was
   0.255 AU, far larger than the noise σ = 0.005, and `log σ` had to
   inflate by 50× to absorb it. Fix: explicit `t0=0.0` argument; the
   integrator starts where `INIT_KNOWN` is defined and saves at the
   requested ts.
2. **Adam is the wrong optimizer for this loss landscape.** ∂loss/∂log_G
   ≈ 3·10⁵ at truth while ∂loss/∂log_M ≈ 5 — five orders of magnitude
   spread. Per-element gradient clipping flattens log_G alone, then Adam's
   first-step bias correction takes a step of size `lr · sign(grad)` in
   every coordinate regardless. The first step always overshoots; the
   integrator hits NaN at the perturbed parameters; momentum carries us
   into a region where `σ` inflates to absorb the bad fit. The fix here
   was to swap MAP onto L-BFGS (handles 5-OOM conditioning natively) and
   to use `clip_by_global_norm(1.0)` for SVI so direction is preserved.
