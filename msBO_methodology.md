# Multi-State Bayesian Optimisation (msBO): Methodology

*Audience: accelerator physicists familiar with beam physics and basic probability/statistics.*

---

## 1. The problem: optimising a machine across multiple beam states

Modern accelerators routinely operate with multiple **beam states** — different ion species, charge states, energies, or optics configurations — all sharing the same magnets or RF cavities.
A single knob change (e.g. a corrector current) therefore affects *every* state simultaneously.
The goal is to find one setting of the shared controls that gives acceptable beam quality across **all** states at once.

Formally, let

- $\mathbf{x} \in \mathbb{R}^d$ — the shared control vector (corrector currents, RF phases, …)  
- $s \in \{1, \dots, S\}$ — the beam state index  
- $\mathbf{y}(s) = [y_1(s), \dots, y_J(s)]$ — the $J$ measured observables when the machine is in state $s$ (BPM positions, transmission ratios, …)

The machine acts as a **black box**: given $(\mathbf{x}, s)$ it returns $\mathbf{y}(s)$ after ramping the state and reading the diagnostics.
Each such call takes seconds to minutes on a real machine, so the total budget of evaluations is small (tens to a few hundreds).

The composite objective $f(\mathbf{x})$ encodes the physics goal — for example, minimising BPM position variance across states while keeping beam transmission above a threshold:

$$
f(\mathbf{x}) = 1 - \underbrace{\sum_{j=1}^{J-1}\left[\operatorname{Var}_s\bigl(y_j(\mathbf{x},s)\bigr) + w\,\mathbb{E}_s\bigl[y_j(\mathbf{x},s)^2\bigr]\right]}_{\text{BPM variance term}} + \underbrace{g\!\left(\min_s y_J(\mathbf{x},s)\right)}_{\text{transmission constraint}}
$$

where $g(\cdot)$ is a smooth step function that penalises beam loss below a threshold.

---

## 2. Gaussian Process surrogate model

Because each oracle call is expensive, msBO builds a **probabilistic surrogate model** of the machine response before making the next measurement.
The model of choice is a **Gaussian Process (GP)** — a distribution over functions that, given observed data $\{(\mathbf{x}_i, s_i, \mathbf{y}_i)\}$, returns a Gaussian predictive distribution at any new point:

$$
p\bigl(y_j(\mathbf{x}, s) \mid \text{data}\bigr) = \mathcal{N}\!\bigl(\mu_j(\mathbf{x},s),\;\sigma_j^2(\mathbf{x},s)\bigr)
$$

The **mean** $\mu_j$ is the model's best guess; the **variance** $\sigma_j^2$ quantifies how uncertain it is.
The model automatically becomes more confident near points that have already been measured.

### 2.1 Multi-Task GP — sharing information across states and observables

A plain GP would model each $(j, s)$ pair independently, wasting the fact that different states share the same physics.
msBO uses a **Multi-Task GP** (Bonilla et al. 2007; implemented via BoTorch `MultiTaskGP`) that treats every $(j, s)$ combination as a separate **task** $t = s \cdot J + j$, with $T = S \cdot J$ tasks total.

The key ingredient is an **inter-task covariance matrix** $\mathbf{B} \in \mathbb{R}^{T \times T}$.
The full GP covariance between two observations $(t, \mathbf{x})$ and $(t', \mathbf{x}')$ is:

$$
k\bigl((t,\mathbf{x}),\,(t',\mathbf{x}')\bigr) = B_{tt'} \cdot k_{\text{rbf}}(\mathbf{x}, \mathbf{x}')
$$

where $k_{\text{rbf}}$ is a standard squared-exponential (RBF) kernel over the controls.
The matrix $\mathbf{B}$ is learned from data: if BPM responses in state 0 and state 1 are correlated, $B_{t,t'}$ will be large, and the model will transfer knowledge from one state to the other.
This is the central advantage of a multi-task approach — a measurement at state 0 updates the prediction at state 1 as well.

### 2.2 Training the model

The matrix $\mathbf{B}$, the RBF length-scales $\boldsymbol{\ell}$, and the noise level $\sigma_n^2$ are all **hyperparameters** learned by maximising the **log-marginal likelihood** (LML):

$$
\log p(\mathbf{Y} \mid \mathbf{X}, \boldsymbol{\theta}) = -\tfrac{1}{2}\mathbf{Y}^\top \mathbf{K}_\theta^{-1} \mathbf{Y} - \tfrac{1}{2}\log|\mathbf{K}_\theta| - \tfrac{N}{2}\log 2\pi
$$

where $\mathbf{K}_\theta$ is the $N \times N$ covariance matrix and $N = (\text{number of oracle calls}) \times J$.
Optimisation uses the Adam gradient descent algorithm with a OneCycle learning-rate schedule.
The dominant cost is the **Cholesky factorisation** of $\mathbf{K}_\theta$, which scales as $O(N^3)$.

---

## 3. Acquisition function — choosing the next measurement

The surrogate model is only useful if it tells us *where to measure next*.
This is decided by an **acquisition function** $\alpha(\mathbf{x})$: a scalar that balances **exploitation** (points where the model predicts a high objective) against **exploration** (points where the model is uncertain and might be hiding something better).

### 3.1 Fixed-state acquisition

Because the machine can only be in one state at a time, msBO uses a **fixed-state acquisition function**.
When deciding where to measure in state $s$, only the tasks belonging to that state are sampled from the GP posterior; all other-state tasks use their posterior *mean* (i.e. the model's best current estimate).
This matches reality: you cannot simultaneously measure all states, so you exploit what you already know about the states not currently being scanned.

Two acquisition functions are available:

**Expected Improvement (EI)** — proposes the point most likely to beat the current best observed objective value $f^*$:

$$
\alpha_{\text{EI}}(\mathbf{x}) = \mathbb{E}\bigl[\max(f(\mathbf{x}) - f^*, 0)\bigr]
$$

**Upper Confidence Bound (UCB)** — optimistically assumes the function equals mean plus $\beta$ standard deviations:

$$
\alpha_{\text{UCB}}(\mathbf{x}) = \mu(\mathbf{x}) + \beta\,\sigma(\mathbf{x})
$$

Both are evaluated via **Monte Carlo sampling**: hundreds of joint samples are drawn from the GP posterior and passed through the composite objective $f$, giving an unbiased stochastic estimate of $\alpha$.
The acquisition is then maximised over the control space using gradient-based optimisation.

### 3.2 Knowledge Gradient (KG)

An optional, more expensive acquisition is the **Knowledge Gradient**, which explicitly simulates "what will my model believe after I make this measurement?" and picks the point that leads to the largest expected improvement in the *future* best setting.
It is more information-efficient but 5–10× slower per query.

### 3.3 Trust-region (TurBO)

To avoid getting stuck far from the current best point, msBO optionally applies a **trust region**: acquisition optimisation is restricted to a hyper-rectangle of size `local_bound_size` centred on the current best $\mathbf{x}^*$.
The region grows after consecutive successes (measured objective above a threshold) and shrinks after consecutive failures, following the TurBO algorithm (Eriksson et al. 2019).

---

## 4. The optimisation loop

```
┌──────────────────────────────────────────────────────────┐
│  Initialisation                                          │
│  • Sample n_init control settings with Sobol sequence   │
│  • Order them to minimise total ramping distance         │
│  • Evaluate each at every state → first dataset          │
│  • Train MultiTaskGP on collected data                   │
└───────────────────────┬──────────────────────────────────┘
                        │
              ┌─────────▼────────────────────────────────┐
              │  For each state s in turn:               │
              │  1. Maximise acquisition α(x | state=s)  │
              │  2. Ramp machine to x_new, state s       │
              │  3. Read diagnostics → y_new             │
              │  4. Add (x_new, s, y_new) to dataset     │
              │  5. Re-train MultiTaskGP                 │
              └─────────┬────────────────────────────────┘
                        │
                  repeat until budget exhausted
```

At each iteration the model is retrained from scratch on the growing dataset (warm-start changes this — see Section 6).
The cycle over states is interleaved: state 0 → state 1 → state 2 → state 0 → …, so data from all states accumulates roughly evenly and the inter-state correlations in $\mathbf{B}$ are learned quickly.

---

## 5. Multi-batch queries

### 5.1 The problem with single-step optimisation

In the standard loop above, **one oracle call triggers one model retraining**.
If a single oracle call takes $\Delta t_{\text{oracle}}$ seconds and model training takes $\Delta t_{\text{train}}$ seconds, the total time per setting evaluated is:

$$
\Delta t_{\text{step}} = \Delta t_{\text{oracle}} + \Delta t_{\text{train}} + \Delta t_{\text{acq}}
$$

On a real machine $\Delta t_{\text{oracle}} \sim 10$–60 s; on the virtual machine it is milliseconds.
Training dominates when the dataset grows large (Cholesky is $O(N^3)$).

### 5.2 Querying a batch of $q$ candidates at once

`step_batch(s, q=n_each)` replaces the inner loop `for i in range(n_each): step(s)`.
Instead of querying one candidate, training, then querying again, it:

1. **Optimises a joint $q$-batch acquisition** — finds $q$ settings $\{\mathbf{x}_1, \dots, \mathbf{x}_q\}$ simultaneously.
   The acquisition value is the expected improvement from the *best* of the $q$ points together, accounting for the fact that they will all be evaluated.
2. **Evaluates the $q$ candidates sequentially** on the machine (state changes are still needed between candidates).
3. **Retrains the model once** after all $q$ evaluations.

The result:

| | Oracle calls | Model retrains |
|---|---|---|
| `n_each` × `step()` | $n_{\text{each}}$ | $n_{\text{each}}$ |
| `step_batch(q=n_each)` | $n_{\text{each}}$ | **1** |

The same number of measurements, but $n_{\text{each}}$× fewer retrains.
Since training is the dominant cost for large datasets, this gives a near-linear speedup.

### 5.3 Why does the joint batch acquisition work?

In single-step mode the acquisition greedily picks the single best next point.
Evaluating $q$ points sequentially without retraining would naively pick the same (or nearby) point $q$ times.
The $q$-batch acquisition avoids this by penalising redundant candidates: it evaluates all $q$ points *jointly* through Monte Carlo sampling and rewards diversity.
This is equivalent to asking "what is the best *set* of $q$ experiments to run, knowing I will only update my model once afterwards?"

### 5.4 Asynchronous mode

When `asynchronous=True` the last of the $q$ oracle evaluations is submitted to a background thread before model training begins.
Training and the final oracle measurement therefore overlap:

```
Time →
[oracle 1][oracle 2]···[oracle q-1] [oracle q (async)]
                                    [  train + query  ]
```

This hides most of the training cost behind oracle wall-clock time, which is especially valuable on real machines where each state change takes 10–30 seconds.

---

## 6. Warm-start model training

### 6.1 Cold start (baseline)

Each `train_model()` call initialises the GP hyperparameters $\boldsymbol{\theta} = (\boldsymbol{\ell}, \mathbf{B}, \sigma_n^2)$ **randomly** and runs `model_train_epochs = 200` Adam steps to maximise the LML.
The first ~100 steps are largely wasted finding the right region of the hyperparameter landscape from scratch.

### 6.2 Warm start

Between consecutive BO steps only **one or a few data points are added** to the dataset.
The LML surface therefore changes only slightly — the previous optimum $\boldsymbol{\theta}^*_{\text{prev}}$ is already inside the basin of attraction of the new optimum $\boldsymbol{\theta}^*_{\text{new}}$.

Warm-start seeds the new model with the previous hyperparameters before running Adam:

```
Cold (every call):  random θ  ──200 steps──▶ θ*_new      ✓ but slow
Warm (call ≥ 2):    θ*_prev   ── 50 steps──▶ θ*_new      ✓ same result, 4× faster
```

Two details make this safe:

- **Transform statistics are excluded.** Input normalisation and output standardisation are recomputed from the new dataset on each call.
  Only the kernel and noise parameters are transferred, so the model always sees correctly-scaled data.
- **First call is always cold.** There is no previous model on the very first training call, so 200 epochs are used regardless.

### 6.3 Savings in numbers

With $N_{\text{BO}}$ BO steps after initialisation:

| Mode | Epochs (first call) | Epochs (subsequent) | Total epochs |
|---|---|---|---|
| Cold start | 200 | 200 each | $(1 + N_{\text{BO}}) \times 200$ |
| Warm start | 200 | 50 each | $200 + N_{\text{BO}} \times 50$ |

For $N_{\text{BO}} = 30$ this is 6200 vs 1700 epochs — a **3.6× reduction**.
Per call the speedup approaches 4× as the run lengthens.

### 6.4 Why does it not hurt optimisation quality?

The key insight is that the LML is a *smooth* function of $\boldsymbol{\theta}$.
Adding a single data point changes every entry of $\mathbf{K}_\theta$ by at most $O(1/N)$, which shrinks as data accumulate.
The loss landscape shifts by a proportionally small amount, so the gradient computed at $\boldsymbol{\theta}^*_{\text{prev}}$ already points in roughly the right direction.
Empirically the LML reached after 50 warm steps is indistinguishable from the LML after 200 cold steps, and the downstream optimisation trajectories (best-so-far curves) are statistically identical.

---

## 7. Combined speedup

The two techniques are independent and stack multiplicatively:

| Setting | Training calls per $n_{\text{each}}$ oracle evals | Epochs per training call | Epoch total |
|---|---|---|---|
| Baseline (cold, single) | $n_{\text{each}}$ | 200 | $200 \times n_{\text{each}}$ |
| Warm start only | $n_{\text{each}}$ | 50 | $50 \times n_{\text{each}}$ |
| Multi-batch only | 1 | 200 | 200 |
| **Both** | **1** | **50** | **50** |

Using $n_{\text{each}} = 4$ as in the example notebooks, the combined saving over baseline is $800 / 50 = \mathbf{16\times}$ fewer epochs per group of $n_{\text{each}}$ evaluations.
In wall-clock terms the savings depend on hardware, but on a CPU the model retraining time typically drops from ~30 s per oracle call to ~2–3 s per batch of 4 calls.

---

## 8. Summary

| Component | Role | Key parameter |
|---|---|---|
| Multi-Task GP | Surrogate that shares information across states and observables | `complexity` of the virtual machine; $T = S \times J$ tasks |
| Fixed-state acquisition (EI / UCB / KG) | Selects next measurement point for one state, using model knowledge of all states | `acq_type`, `beta` |
| Trust region (TurBO) | Restricts search to a dynamically-sized region around the current best | `local_bound_size`, `TurBO_*` thresholds |
| Multi-batch (`step_batch`) | Evaluates $q$ candidates per retraining call | `q = n_each` |
| Warm start | Initialises hyperparameters from the previous model | `model_warmstart_epochs = 50` |
