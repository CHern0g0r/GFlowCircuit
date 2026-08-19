# Analysis of the Unequal GFlowNet and REINFORCE Experiments

## 1. Extensive analysis of the current experiments

### 1.1 Scope and methodology

This analysis was computed independently from the `points.csv` files in `outputs/`. The main comparison uses the latest diff-of-product GFlowNet experiment for each circuit and the corresponding diff-of-product REINFORCE experiment. For the Zhu circuits (`c1355`, `c5315`, `apex1`, `bc0`, `dalu`, and `k2`), this means comparing the July GFlowNet runs with the June `zhuDOP` REINFORCE runs. For `adder`, `max`, and `multiplier`, both methods come from the July experiments.

Each sampled terminal circuit was normalized by the initial circuit:

\[
s_n = \frac{s}{s_0}, \qquad d_n = \frac{d}{d_0}.
\]

Both objectives are minimized. The hypervolume is measured relative to the original circuit, `(1, 1)`, as in `show_report.ipynb`. I examined:

- hypervolume for each trained seed separately;
- the Pareto front pooled over all seeds and 50 evaluation samples per seed;
- best size and depth extremes;
- Pareto dominance and front coverage;
- the number of distinct sampled endpoints;
- the mean product improvement over all samples, which measures typical rollout quality rather than only the best discovered tail;
- paired seed differences, with exact sign-flip permutation tests and confidence intervals over the seed-level hypervolume differences.

This distinction between per-seed and pooled results is important. A pooled front answers, “What could the method discover after training many seeds and sampling hundreds of trajectories?” Per-seed hypervolume answers, “How reliably does one trained policy produce a good front?” These are different questions.

### 1.2 Overall results

| Circuit | Mean per-seed HV, GFN | Mean per-seed HV, REINFORCE | Mean difference, GFN − REINFORCE | Pooled HV, GFN / REINFORCE | Assessment |
|---|---:|---:|---:|---:|---|
| `max` | 0.0070 | 0.0032 | +0.0039 | 0.0089 / 0.0047 | Clear and seed-consistent GFN advantage |
| `bc0` | 0.2223 | 0.2071 | +0.0151 | 0.2404 / 0.2256 | Clear GFN advantage |
| `apex1` | 0.1048 | 0.0980 | +0.0068 | 0.1165 / 0.1009 | Moderate GFN advantage |
| `dalu` | 0.0226 | 0.0152 | +0.0073 | 0.0406 / 0.0211 | Better pooled GFN front, but unstable across seeds |
| `k2` | 0.1869 | 0.1857 | +0.0012 | 0.2062 / 0.1920 | Better pooled GFN extreme, no reliable mean advantage |
| `c1355` | 0.0756 | 0.0768 | −0.0012 | 0.0810 / 0.0810 | Effectively tied and probably saturated |
| `adder` | 0 | 0 | 0 | 0 / 0 | Same Pareto optimum; current HV is degenerate |
| `multiplier` | 0.004444 | 0.004442 | approximately 0 | 0.004445 / 0.004444 | Negligible difference; incomplete experiment |
| `c5315` | 0.0756 | 0.0852 | −0.0097 | 0.0864 / 0.0928 | REINFORCE is better by hypervolume |

The seed-level uncertainty changes the interpretation of several apparent pooled-front wins:

| Circuit | 95% CI for paired HV difference | Seed wins/ties/losses for GFN | Exact paired permutation p-value |
|---|---:|---:|---:|
| `max` | `[+0.0025, +0.0053]` | 10 / 0 / 0 | 0.0020 |
| `bc0` | `[+0.0068, +0.0234]` | 9 / 0 / 1 | 0.0059 |
| `apex1` | `[+0.0006, +0.0129]` | 8 / 0 / 2 | 0.0352 |
| `dalu` | `[−0.0028, +0.0175]` | 7 / 0 / 3 | 0.1367 |
| `k2` | `[−0.0044, +0.0069]` | 5 / 0 / 5 | 0.6426 |
| `c1355` | `[−0.0062, +0.0038]` | 5 / 1 / 4 | 0.6836 |
| `c5315` | `[−0.0165, −0.0028]` | 3 / 0 / 7 | 0.0176 |

The strongest evidence for a GFlowNet advantage is therefore on `max`, `bc0`, and `apex1`. The evidence on `dalu` is promising but not conclusive with ten seeds. The pooled `k2` front is better, but the method is not more reliable per seed. `c5315` should not be categorized as a tie: under the chosen hypervolume definition, REINFORCE is currently better.

### 1.3 Analysis by circuit

#### `max`

This is the clearest GFlowNet success. The pooled GFlowNet front contains 12 nondominated points, compared with 3 for REINFORCE. GFlowNet reaches approximately 4.0% size reduction and 33.1% depth reduction at its separate extremes, whereas REINFORCE reaches approximately 1.5% and 31.0%. The GFlowNet pooled front dominates the REINFORCE pooled front, and every paired seed has higher GFlowNet hypervolume.

However, a random GFlowNet rollout is not generally better. The mean normalized product improvement over all samples is 10.9% for GFlowNet and 29.9% for REINFORCE. GFlowNet is finding a much better rare tail while allocating considerable probability to worse endpoints. It is better here as a search-and-archive method, not as a one-shot policy.

#### `bc0`

The GFlowNet front dominates the REINFORCE front. Both methods reach the same best depth reduction, approximately 45.2%, but GFlowNet reaches a better size extreme: approximately 53.7% reduction versus 50.5%. The result is also consistent over seeds, with 9 of 10 GFlowNet seeds winning in hypervolume.

Unlike most of the other circuits, typical GFlowNet quality is also slightly better on `bc0`: mean product improvement is 65.5% versus 64.2% for REINFORCE. This is the most convincing example of both improved discovery and competitive average behavior.

#### `apex1`

The best depth reduction is the same for both methods, approximately 37.0%, but GFlowNet extends the size side of the front from approximately 27.5% to 32.2%. Its pooled front fully dominates the REINFORCE front. Eight of ten paired seeds favor GFlowNet, although the advantage is smaller than for `max` or `bc0`.

As on `max`, the advantage is concentrated in rare samples. Mean product improvement over every sampled trajectory is 45.5% for GFlowNet and 48.1% for REINFORCE.

#### `dalu`

The pooled GFlowNet front is substantially stronger and fully dominates the REINFORCE front. GFlowNet reaches separate size and depth reductions of approximately 36.6% and 17.1%, compared with 21.2% and 11.4% for REINFORCE. Nevertheless, three of ten GFlowNet seeds lose, the confidence interval includes zero, and some GFlowNet seeds have zero hypervolume because none of their endpoints improve both objectives relative to the `(1,1)` reference.

The mean product improvement is only 8.7% for GFlowNet, compared with 20.7% for REINFORCE. Thus the attractive pooled front is being produced by a small number of rare outcomes. This circuit needs more seeds and a probability-of-success analysis before claiming a general advantage.

There is also a strong action-space effect. The older GFlowNet experiment restricted to actions 0–4 had zero hypervolume on `dalu`, while the newer all-action experiment obtained a pooled hypervolume of 0.0406. The additional resubstitution actions appear more important than the algorithm label alone.

#### `k2`

The GFlowNet pooled front dominates the REINFORCE front and reaches a slightly better size extreme, approximately 47.9% versus 45.6%, while both reach approximately 43.5% depth reduction. However, the per-seed comparison is exactly 5 wins and 5 losses, with nearly identical mean hypervolume.

The older restricted-action GFlowNet run actually had higher mean hypervolume, 0.1971, than the current all-action GFlowNet run, 0.1869. The current all-action run nevertheless produced the best pooled extreme. This is another indication that pooled selection rewards rare discoveries and may disagree with policy reliability.

#### `c1355`

Both methods have effectively reached the same search-space ceiling. The GFlowNet pooled front contains `(386,17)`, which weakly dominates the two REINFORCE points `(386,18)` and `(387,17)`, but the numerical hypervolume difference is negligible. Both methods reduce size by approximately 23.4% and depth by approximately 34.6%.

The mean endpoint quality favors REINFORCE—45.4% product improvement versus 38.8% for GFlowNet—even though GFlowNet has the technically dominant pooled point. More training with the same actions and 20-step horizon is unlikely to change the frontier substantially unless the current endpoint is not the reachability limit.

#### `adder`

Both algorithms find exactly the same Pareto endpoint, `(893,255)`, from an initial `(1020,255)`. This is a 12.45% size reduction with no depth change. REINFORCE returns this endpoint for all 500 sampled trajectories; GFlowNet returns it for 449 of 500 and occasionally produces worse endpoints.

The reported hypervolume of zero does not mean there was no optimization. It is caused by the reference point `(1,1)`: a point on the boundary `normalized_depth = 1` has zero two-dimensional dominated area even when size improves. For this circuit, either report the size reduction directly or use a predeclared worse reference point with margin, such as `(1.05,1.05)`, provided the same reference is used for every method.

The lack of depth movement suggests a reachability problem rather than merely a learning problem. Before tuning the optimizer, test random and hand-designed sequences at longer horizons to see whether any available transformation can reduce adder depth.

#### `multiplier`

The two fronts are almost identical. GFlowNet finds `(24315,267)` and `(24316,262)`, while REINFORCE finds `(24316,262)`. The extra GFlowNet point changes one node and has a negligible hypervolume effect. This should be treated as a tie, not a meaningful GFlowNet improvement.

The experiment is also incomplete: only 8 GFlowNet seeds and 7 REINFORCE seeds have checkpoints and sampled points, although both configurations request 10. Neither experiment contains its expected final report JSON. Because `multiplier` is much larger than the other circuits, the missing runs may be related to runtime or job limits, and should be completed before further interpretation.

#### `c5315`

This circuit is a genuine tradeoff, but REINFORCE currently has higher mean and pooled hypervolume. GFlowNet reaches the best size extreme `(1281,30)`, whereas REINFORCE reaches the better depth extreme `(1308,25)`. The complete pooled fronts cross, so neither algorithm fully dominates the other, but REINFORCE covers more valuable area under the current reference.

This is the clearest target for a depth-conditioned or multi-preference GFlowNet. Simply increasing stochastic exploration may produce more points without fixing the missing depth extreme.

### 1.4 Pooled discovery versus typical behavior

The following table shows mean normalized product improvement over all sampled endpoints. It measures the quality of a randomly drawn terminal circuit rather than the quality of the retained Pareto archive.

| Circuit | GFN mean product improvement | REINFORCE mean product improvement |
|---|---:|---:|
| `bc0` | 65.5% | 64.2% |
| `multiplier` | 13.4% | 13.2% |
| `c1355` | 38.8% | 45.4% |
| `c5315` | 36.1% | 40.6% |
| `apex1` | 45.5% | 48.1% |
| `dalu` | 8.7% | 20.7% |
| `k2` | 55.0% | 60.3% |
| `adder` | 11.3% | 12.45% |
| `max` | 10.9% | 29.9% |

This reveals the principal behavior of the present GFlowNet: it usually samples a broader distribution and occasionally discovers endpoints that improve the pooled Pareto front, but its probability mass is not concentrated on high-quality endpoints. That behavior is appropriate if the intended use is to sample many circuits and retain a nondominated archive. It is less attractive if only one or a few inference trajectories can be afforded.

Consequently, future reports should always include both:

1. discovery metrics: pooled front, pooled hypervolume, coverage, and best extremes under a fixed total sampling budget;
2. reliability metrics: mean per-seed hypervolume, expected QoR/product, probability of entering the retained front, and best-of-`N` curves for several values of `N`.

## 2. Budget differences in the experimental setup and how to fix them

### 2.1 Training trajectory budget [DONE]

All current configurations specify 200 episodes, but an episode does not represent the same amount of data for the two algorithms.

The latest GFlowNet configurations contain two conflicting settings:

```yaml
algorithm:
  tb:
    batch_size: 4

tb:
  trajectories_per_episode: 2
```

The code reads the top-level `tb` block first, so the effective GFlowNet batch is 2 trajectories per episode. Therefore, for each seed:

- current GFlowNet: `200 × 2 = 400` training trajectories;
- REINFORCE: `200 × 1 = 200` training trajectories.

The older June GFlowNet runs use four trajectories per episode, or 800 trajectories per seed. Thus the GFlowNet cohorts are not even internally matched by trajectory count.

This is not the only computational difference. GFlowNet performs one optimizer update per episode using its trajectory batch. REINFORCE loops over all 20 trajectory steps and performs a separate Adam update for each step. It therefore performs approximately 4,000 policy optimizer steps per seed, based on only 200 sampled trajectories. A comparison matched only by “episodes” is neither data-matched nor update-matched.

### 2.2 Evaluation and selection budget

The standalone `points.csv` evaluations are mostly well matched: both methods have 50 samples per completed seed. The pooled fronts therefore usually compare 500 samples against 500 samples. This is a good basis for a fixed discovery-budget comparison.

There are still important exceptions and distinctions:

- `multiplier` has 400 GFlowNet samples from 8 seeds and 350 REINFORCE samples from 7 seeds;
- `paper_mode.infer_rollouts` is 4 for the new GFlowNet runs and 10 for REINFORCE, which affects training-time checkpoint evaluation and reported best-of-rollout metrics, although it does not affect the separately generated 50-sample `points.csv` comparison;
- pooling across ten seeds conflates training robustness and inference sampling. A method can obtain a strong pooled front from one lucky seed while being unreliable in nine others.

### 2.3 Recommended fair-budget protocols

There is no single universally fair budget because the algorithms update differently. I recommend reporting at least two protocols.

#### Environment-interaction-matched protocol

- Fix a total number of complete terminal trajectories per seed, for example 1,000 or 4,000.
- Count every training rollout, including exploratory rollouts.
- Set REINFORCE episodes and GFlowNet episodes/batch size so both consume the same number of circuit trajectories.
- Keep horizon, action set, encoder, hidden dimensions, and circuit split identical.
- Evaluate every seed with the same number of fresh samples and the same evaluation seeds.

This is the cleanest comparison of sample efficiency because circuit transformations are the expensive operation.

#### Wall-clock- or compute-matched protocol

- Give each seed the same GPU/CPU wall-clock allowance.
- Record the number of completed circuit transformations, trajectories, optimizer updates, and elapsed time.
- Evaluate the latest completed checkpoint at fixed time intervals.
- Plot hypervolume against wall-clock time and against environment transitions.

This captures practical efficiency when batched GFlowNet execution and per-step REINFORCE updates have different hardware utilization.

#### Fixed discovery-budget protocol

After training, compare best discovered fronts as a function of inference samples:

```text
N = 1, 2, 5, 10, 20, 50, 100, 500
```

For every `N`, report the mean and confidence interval of best-of-`N` hypervolume or archive hypervolume over independent evaluation repetitions. This will show whether GFlowNet's advantage appears after 10 samples or only after hundreds.

### 2.4 Concrete setup changes

1. Remove the duplicate `algorithm.tb` and top-level `tb` settings, or define an explicit precedence rule and log the resolved values.
2. Add resolved fields such as `training_trajectories`, `environment_steps`, `optimizer_steps`, and `wall_time_seconds` to every report.
3. Finish all requested seeds; treat partial experiments as failed or incomplete rather than silently pooling unequal counts.
4. Use paired seeds and shared evaluation seeds across algorithms.
5. Use at least 10 seeds for preliminary conclusions and preferably 20 or more for circuits such as `dalu`, where between-seed variance is large.
6. Save intermediate checkpoints so learning curves can be compared at equal interaction budgets rather than only at episode 200.

## 3. Reward differences and what to do about them

### 3.1 The current objectives are not equivalent

The GFlowNet converts final normalized product improvement into a positive terminal reward:

\[
R(x) = \exp\left(\alpha I_{\mathrm{product}}(x)\right), \qquad \alpha=4.
\]

It then minimizes the trajectory-balance residual. Only the final circuit's size and depth determine `log R`.

REINFORCE is configured differently:

- it receives dense per-step differences in normalized `size × depth`;
- it subtracts a Zhu/resyn2 per-step baseline;
- it uses `gamma = 0.9`;
- `terminal_reward` is false.

With `gamma < 1`, the dense reward no longer telescopes to the terminal product improvement. Early improvements receive greater weight, and delayed improvements or sequences that temporarily worsen QoR are discouraged. GFlowNet, in contrast, can assign high terminal reward to such a sequence if it finishes well. This difference can plausibly explain why GFlowNet discovers rare deep or wide endpoints while REINFORCE concentrates on consistently good trajectories.

The baseline also changes the numerical training signal. Although a state-independent baseline should ideally reduce variance without changing the expected policy-gradient optimum, the current dense per-step subtraction, discounting, critic approximation, and sequential per-step updates make the effective training dynamics different from the GFlowNet objective.

### 3.2 First recommendation: perform an objective-matched algorithm ablation

To isolate the algorithm, give both methods exactly the same terminal scalar objective:

- `terminal_reward: true` for REINFORCE;
- `gamma: 1.0`;
- identical normalized product improvement before the GFlowNet exponential transformation;
- a critic or running baseline for variance reduction, but no action-dependent reward shaping;
- identical reward clipping and normalization policy;
- identical action set and horizon.

GFlowNet requires a strictly positive reward, so it can use `exp(alpha × improvement)`. REINFORCE can optimize either `improvement` or `alpha × improvement`; the monotonic exponential should also be tested because it changes how strongly both methods prioritize the high-reward tail. A useful ablation is:

```text
terminal improvement
exponential terminal improvement, alpha ∈ {1, 2, 4, 8}
dense undiscounted improvement, gamma = 1
current dense discounted reward, gamma = 0.9
```

This will separate the effect of the learning algorithm from the effect of reward timing and concentration.

### 3.3 Second recommendation: use a genuinely Pareto-aware objective

The product objective is a single scalarization. It can prefer a balanced product improvement, but it does not explicitly ask the model to cover the size-depth Pareto front. Different tradeoffs can have similar products, and extreme size-only or depth-only solutions may receive insufficient reward.

A better approach is a preference-conditioned policy or GFlowNet. Sample a preference `λ ∈ [0,1]` and use, for example,

\[
R_\lambda(x) = \exp\left(\alpha\left[
\lambda(1-s_n) + (1-\lambda)(1-d_n)
\right]\right).
\]

The preference should be included in the policy observation. During training, deliberately sample:

- `λ = 1`, emphasizing size only;
- `λ = 0`, emphasizing depth only;
- several evenly spaced intermediate preferences;
- additional random preferences for coverage.

A weighted Tchebycheff scalarization is preferable if the true Pareto front is non-convex, because a simple weighted sum can miss non-convex regions. The policy can then be evaluated by sweeping preferences and merging the resulting endpoints into a Pareto archive.

Direct hypervolume-contribution rewards are possible, but they are nonstationary because the reward depends on the current archive. They are better used as replay priority or an exploration bonus than as the only terminal reward. A practical design is:

1. stationary preference-conditioned terminal reward;
2. persistent nondominated archive;
3. replay or priority bonus for endpoints that expand the archive;
4. final selection based on exact size and depth, not the training scalarization.

### 3.4 Controlling diversity versus quality

The present GFlowNet often finds superior rare endpoints but has lower mean rollout quality. The reward temperature `alpha` controls part of this tradeoff:

- smaller `alpha` produces a flatter distribution and broader exploration;
- larger `alpha` concentrates more probability on the best product outcomes but can reduce Pareto diversity and miss alternative tradeoffs.

Therefore `alpha` should not be selected only by mean reward. Select it using both archive hypervolume and probability of sampling an archive-quality endpoint. Preference conditioning is a cleaner way to preserve breadth while allowing higher concentration within each preference.

## 4. Other important improvements to the experimental setup

### 4.1 Match the available action space

The current Zhu comparison is confounded by different actions. The July GFlowNet runs use `available_actions: null`, exposing all seven transformations, including:

- action 5: `resub fast`;
- action 6: `resub strong`.

The compared June REINFORCE runs use only actions 0–4. The improvement on `dalu` is particularly sensitive to this change. New controlled runs should compare both algorithms under:

1. restricted actions 0–4;
2. all actions 0–6.

Action-set results should be presented as a separate ablation, not folded into the algorithm comparison.

### 4.2 Improve GFlowNet convergence and diagnose `logZ`

The latest completed GFlowNet runs end with TB losses around 1,350–1,560 and `logZ ≈ 1.98`. This is far from a balanced trajectory model. For a roughly uniform 20-step policy with seven actions,

\[
-\log P_F(\tau) \approx 20\log 7 \approx 38.9.
\]

Because `logZ` starts at zero and its default learning rate is only `10 × 0.001 = 0.01`, 200 optimizer steps move it to approximately 2. The root-mean-square TB residual remains around 37–39, consistent with the observed loss. The current policies should therefore be regarded as early, under-converged TB models.

Recommended changes:

- initialize `logZ` near an estimate of the log partition scale, such as the sum of log legal-action counts along a random trajectory;
- train for substantially more than 200 optimizer steps;
- sweep `log_z_learning_rate` separately from policy learning rate;
- log the mean, standard deviation, and quantiles of the TB residual, `log P_F`, `log R`, and gradient norms;
- use validation residual and archive quality for early stopping;
- consider SubTB or replay-buffer TB updates to obtain more learning signal from each expensive trajectory.

### 4.3 Check the backward-policy assumption

The current trajectory-balance code sets `log P_B = 0`. This is valid if the generative state space is treated as a tree of unique action prefixes with a deterministic parent. If different action sequences reach the same circuit and the intended terminal object is the unique circuit rather than the action sequence, the implementation samples endpoints in proportion to both reward and trajectory multiplicity.

That multiplicity can make an endpoint common because many sequences reach it, not because it has high reward. The intended object should be made explicit:

- if the object is the complete synthesis program, `P_B = 1` can be coherent;
- if the object is the unique terminal circuit, use a canonical trajectory, deduplicate/hash circuit states, or implement a backward policy that accounts for multiple parents.

### 4.4 Fix statistical issues in the notebook

The notebook's all-pairs comparison creates 100 pairwise values from 10 source and 10 target seeds and computes a standard error as if all 100 values were independent. They are not: each source seed and each target seed is reused ten times. This substantially understates uncertainty.

Use one of the following instead:

- paired seed differences when run IDs share seeds;
- an unpaired seed-level bootstrap when seeds are unrelated;
- a hierarchical bootstrap that resamples training seeds first and evaluation trajectories second.

The number of nondominated points should also not be interpreted as a quality measure by itself. It is affected by discrete metric resolution, duplicates, and sample count. Hypervolume, epsilon indicator, front coverage, and best-of-`N` curves are more informative.

### 4.5 Use a non-degenerate hypervolume reference

The reference `(1,1)` is intuitive, but it assigns zero hypervolume to any front that improves only one objective while leaving the other at its original value. This is exactly what happens on `adder`. It also discards otherwise interesting `dalu` tradeoffs whose depth is temporarily or finally worse than the original.

Possible fixes are:

- keep `(1,1)` but always report separate size and depth reductions alongside HV;
- predeclare a worse reference with margin, for example `(1.05,1.05)` or a domain-informed bound;
- report constrained hypervolume separately, where only solutions improving both objectives are considered feasible.

The reference must be chosen before looking at which algorithm wins and must be identical across algorithms for a circuit.

### 4.6 Measure reachability before expensive learning sweeps

For each circuit and horizon, run a large action-space diagnostic:

- uniform random trajectories;
- single-action repetitions;
- the resyn2 macro sequence and variants;
- random perturbations of strong discovered sequences;
- horizons 20, 40, and 60 where computationally feasible.

Plot the reachable size-depth cloud and its Pareto front. If random and macro searches never improve `adder` depth, no optimizer can learn such an improvement under that action set and horizon. If good endpoints are reachable but extremely rare, the problem is exploration. If they are reached often but the learned policy ignores them, the problem is the reward or optimization.

### 4.7 Improve provenance and experiment bookkeeping

Every result directory should record:

- Git commit and dirty-worktree status;
- fully resolved Hydra configuration with no duplicated active settings;
- action names as well as IDs;
- requested and completed seed counts;
- actual trajectory, transition, optimizer-step, and wall-clock budgets;
- checkpoint selection rule;
- evaluation seed and sample count;
- termination reason for partial jobs.

Run names such as `tbzhuDOP` are currently used for both conceptual families and can be confusing. Names should explicitly encode algorithm, reward, action set, horizon, and budget, for example:

```text
gfn_tb_dop_all7_h20_traj1000
reinforce_terminal_dop_all7_h20_traj1000
```

## 5. Recommended next experiment sequence

The following order would produce the most informative results with the least ambiguity:

1. **Repair the evaluation:** paired seed statistics, best-of-`N` curves, expected QoR, non-degenerate HV reporting, and explicit incomplete-run warnings.
2. **Run the controlled baseline:** both algorithms, all seven actions, horizon 20, identical terminal product objective, equal environment-trajectory budget, and at least 10 paired seeds.
3. **Run the action ablation:** actions 0–4 versus 0–6 for both algorithms, especially on `dalu`, `k2`, and `c5315`.
4. **Fix and extend TB optimization:** improved `logZ` initialization, longer training, residual diagnostics, and learning curves against trajectory budget.
5. **Test reachability and horizon:** random/macro baselines at horizons 20, 40, and 60 for `adder`, `multiplier`, and `c1355`.
6. **Introduce preference conditioning:** size-focused, depth-focused, and intermediate preferences; evaluate the merged Pareto archive.
7. **Repeat the final comparison:** complete all seeds, use equal inference budgets, and report both discovery and reliability metrics.

Under the current evidence, the most defensible conclusion is that GFlowNet is a stronger rare-candidate discovery mechanism on `max`, `bc0`, and `apex1`, with promising pooled results on `dalu` and `k2`. The experiments do not yet establish a universal GFlowNet advantage because budgets, actions, and rewards are not consistently matched, the TB models are under-converged, and several pooled improvements come from very rare samples rather than consistently better policies.
