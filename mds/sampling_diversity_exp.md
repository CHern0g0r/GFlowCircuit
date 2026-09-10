# Sampling Diversity Experiment

This is the experiment that checks the sampling diversity of different policies used to sample ABC optimization recipes. It checks both the diversity of the recipe itself, the diversity of the sampled circuits and how this diversity transforms to the quality of k-LUT transformed samples.


# Metrics

## Recipe Diversity

Let a recipe $r_i=(a_{i,1},\ldots,a_{i,T})$ be the complete ordered
sequence of ABC actions sampled in rollout $i$. Recipes are equal only when
their complete action sequences are equal, including action arguments and the
termination position.

### Unique recipe fraction curve

For the first $n$ sampled recipes, let

$$
U(n)=\left|\{r_1,\ldots,r_n\}\right|.
$$

The unique recipe fraction is $U(n)/n$. Plot it as a function of the sampling
budget $n$, from 1 through 1000. The curve describes how quickly repeated
recipes appear as the sampling budget grows. A value close to one means that
almost every additional rollout produces a previously unseen recipe; a falling
curve means that probability mass is concentrated on a smaller set of recipes.
Because the curve depends on sample order, compute its reported mean and
uncertainty by randomly permuting the 1000 recipes multiple times. The endpoint
$U(1000)/1000$ is unaffected by permutation.

### Recipe entropy

Let $c(r)$ be the number of times recipe $r$ occurs among the $N=1000$
samples and let

$$
\hat p(r)=\frac{c(r)}{N}.
$$

The empirical recipe entropy is

$$
H_{\mathrm{recipe}}=-\sum_{r:c(r)>0}\hat p(r)\log \hat p(r).
$$

Use the natural logarithm and report entropy in nats. Unlike the unique recipe
fraction, this metric accounts for the frequency distribution: it is small when
most samples are concentrated on a few recipes and large when probability mass
is distributed more evenly across many recipes. Compute it separately for every
checkpoint and circuit; do not pool recipes from different circuits.

### Action entropy

For action position $t$, let $I_t$ contain the rollouts that reach that action
position and let $N_t=|I_t|$. Estimate the marginal probability of action $a$ as

$$
\hat p_t(a)=\frac{1}{N_t}\sum_{i\in I_t}
\mathbf{1}[a_{i,t}=a],
$$

and compute

$$
H_t=-\sum_{a:\hat p_t(a)>0}\hat p_t(a)\log \hat p_t(a).
$$

Report the per-position values $H_t$ and their mean over the recipe horizon.
This shows where in a recipe the policy branches: two policies can have similar
recipe entropy while one makes diverse choices early and the other varies only
the final actions. If recipes can terminate early, treat termination as an
explicit action and exclude positions after termination from both the numerator
and denominator for that position.

## AIG and LUT Diversity

For each sampled final AIG, define its coordinate as

$$
q_i^{\mathrm{AIG}}=(S_i^{\mathrm{AIG}},D_i^{\mathrm{AIG}}),
$$

where $S_i^{\mathrm{AIG}}$ is the number of AIG nodes and
$D_i^{\mathrm{AIG}}$ is the number of AIG levels. The AIG coordinate diversity
is the number of distinct coordinate pairs:

$$
C_{\mathrm{AIG}}=
\left|\{q_i^{\mathrm{AIG}}:1\leq i\leq N\}\right|.
$$

After fixed 6-LUT mapping, define

$$
q_i^{\mathrm{LUT}}=(S_i^{\mathrm{LUT}},D_i^{\mathrm{LUT}}),
$$

where $S_i^{\mathrm{LUT}}$ is the number of LUTs and
$D_i^{\mathrm{LUT}}$ is the number of LUT levels. The LUT coordinate diversity
is

$$
C_{\mathrm{LUT}}=
\left|\{q_i^{\mathrm{LUT}}:1\leq i\leq N\}\right|.
$$

These metrics count distinct quality-of-result trade-offs, not distinct graph
structures. Several structurally different AIGs or LUT networks may share the
same size-depth pair and therefore contribute only one coordinate. The
canonical hashes recorded by the protocol identify exact duplicate networks and
must be kept alongside the coordinates so that coordinate equality is not
mistaken for structural equality.

## LUT Quality

All LUT quality metrics are computed from the $N=1000$ mapped samples for one
checkpoint and one circuit.

### Best LUT depth

The best LUT depth is

$$
D^*_{\mathrm{LUT}}=\min_i D_i^{\mathrm{LUT}}.
$$

Save a corresponding mapped circuit. If several circuits attain the minimum
depth, select the one with the smallest LUT count; if a tie remains, use the
earliest sampled circuit as a deterministic tie-breaker.

### Best LUT size

The best LUT size is

$$
S^*_{\mathrm{LUT}}=\min_i S_i^{\mathrm{LUT}}.
$$

Here, size means the number of LUTs in the mapped network. If several circuits
attain the minimum size, select the one with the smallest LUT depth and then use
sample order as the final tie-breaker.

### Best LUT size-depth product

For mapped circuit $i$, define the LUT size-depth product as

$$
P_i^{\mathrm{LUT}}=
S_i^{\mathrm{LUT}}D_i^{\mathrm{LUT}}.
$$

The best LUT size-depth product is

$$
P^*_{\mathrm{LUT}}=\min_i P_i^{\mathrm{LUT}}.
$$

This metric favors mapped circuits that jointly achieve a small LUT count and a
small LUT depth. If several circuits attain the same minimum product, select the
one with the smallest LUT count, then the smallest LUT depth, and finally the
earliest sample as a deterministic tie-breaker.

### LUT Pareto front

Construct the Pareto front from the distinct LUT coordinate pairs
$(S_i^{\mathrm{LUT}},D_i^{\mathrm{LUT}})$. A point $x=(S_x,D_x)$ dominates
$y=(S_y,D_y)$ when

$$
S_x\leq S_y,\qquad D_x\leq D_y,
$$

and at least one inequality is strict. The saved front contains exactly the
points that are not dominated by any sampled point. Save both the coordinates
and the sample identifiers or canonical hashes of every circuit that realizes
each coordinate, since multiple mapped networks may occupy the same Pareto
point.

### AIG-to-LUT rank correlation

Use Spearman rank correlation to measure whether AIG quality preserves its
ordering after LUT mapping. Compute two correlations over the paired samples:

$$
\rho_{\mathrm{size}}=
\operatorname{Spearman}(S^{\mathrm{AIG}},S^{\mathrm{LUT}}),
\qquad
\rho_{\mathrm{depth}}=
\operatorname{Spearman}(D^{\mathrm{AIG}},D^{\mathrm{LUT}}).
$$

Assign average ranks to tied values. A correlation near $1$ means that better
AIG size or depth generally predicts better mapped size or depth. A value near
zero indicates weak rank preservation, while a negative value indicates that
the ordering tends to reverse. If either input is constant across all samples,
the corresponding correlation is undefined and must be recorded as such rather
than as zero.

# Implementation details

The pipeline is implemented by `src/sampling_diversity.py` and is run from the
repository root:

~~~bash
python -m src.sampling_diversity \
  --dataset-cfg cfg/data/zhu2020/i10.yaml \
  --checkpoint outputs/<experiment> \
  --num-samples 1000 \
  --abc-path /path/to/abc \
  --output-dir outputs/sampling_diversity/<experiment> \
  --device cuda \
  --seed 42
~~~

The five required inputs are:

- `--dataset-cfg`: a circuit dataset YAML with the existing `path`, `format`,
  and `files` fields;
- `--checkpoint`: either one checkpoint file, a directory containing `.pt`
  files, a `saved_models` directory containing `run_*/last.pt`, or an
  experiment directory containing `saved_models`;
- `--num-samples`: the number of stochastic recipes sampled from every
  checkpoint for every circuit;
- `--abc-path`: the ABC executable used for mapping;
- `--output-dir`: the root of all generated artifacts.

The pipeline infers the training Hydra configuration from
`<experiment>/.hydra/config.yaml`. `--model-config` can supply it explicitly
when the checkpoints have been moved. The recipe horizon is read from that
configuration and can be overridden with `--num-steps`. The optional
`--gflownet-batch-size` limits the number of simultaneous GFlowNet rollouts
without changing the requested sample count. Existing `stats.csv` or
`metrics.json` files are not replaced unless `--overwrite` is passed.

For every checkpoint-circuit pair, the sampler returns the terminal AIG
statistics and complete action-ID sequence. The sequence is replayed in the
OpenSpiel circuit environment and the terminal network is saved with
`pyspiel.save_circuit`. This writer structurally hashes the network before
producing its AIGER representation. The replayed size and depth must exactly
match the sampler result; a mismatch terminates the run.

Each saved AIG is mapped by the supplied ABC binary with:

~~~text
read <sample.aig>; strash; if -K 6; print_stats; write_blif <sample.blif>
~~~

The LUT count and LUT depth are parsed from the post-mapping `print_stats`
output. For stable serialization, comments and blank lines are removed from the
BLIF file and its `.model` name is normalized. Both AIG and LUT files receive a
SHA-256 digest.

## Output layout

`stats.csv` and `metrics.json` are aggregate files at the output root. Circuit
artifacts are separated by circuit and checkpoint:

~~~text
<output-dir>/
  stats.csv
  metrics.json
  <circuit>/
    model_<index>_run_<run>_seed_<training-seed>/
      aig/
        000000.aig
        ...
      lut/
        000000.blif
        ...
~~~

The six-digit file stem is the `sample_index` within its checkpoint-circuit
group. `stats.csv` contains one row per sampled recipe and the following
columns:

| Column | Description |
| --- | --- |
| circuit | Absolute path of the source circuit. |
| algorithm | Algorithm name read from the training configuration. |
| run_id | Training run identifier read from the checkpoint. |
| training_seed | Training seed read from the checkpoint. |
| sampling_seed | Inference seed used for this checkpoint-circuit group. |
| checkpoint | Absolute source-checkpoint path. |
| sample_index | Zero-based sample index and generated-file stem. |
| action_sequence | Semicolon-separated ABC action names in execution order. |
| aig_path | Generated AIG path relative to the output root. |
| aig_sha256 | SHA-256 digest of the generated AIG. |
| aig_size | Number of AIG nodes. |
| aig_depth | Number of AIG levels. |
| lut_path | Generated mapped BLIF path relative to the output root. |
| lut_sha256 | SHA-256 digest of the mapped BLIF. |
| lut_size | Number of 6-LUTs. |
| lut_depth | Number of LUT levels. |

`metrics.json` has one `groups` entry for every checkpoint-circuit pair.
Metrics are not pooled across different circuits or training seeds. Each entry
contains:

- the unique-recipe fraction curve, recipe entropy, per-position action
  entropy, and mean action entropy;
- the number of distinct AIG and LUT size-depth pairs;
- the best AIG size, depth, and size-depth product;
- the best LUT size, depth, and size-depth product;
- the LUT size-depth Pareto front, including every sample index realizing each
  Pareto coordinate;
- the AIG-to-LUT Spearman size and depth correlations.

Every best-value record contains the value, all sample indices attaining it,
and the deterministically selected sample after applying the metric's
tie-breaking rule. An undefined rank correlation is serialized as JSON null.
The unique-recipe fraction curve is averaged over `--curve-permutations` random
orderings and reports its mean, 5th percentile, and 95th percentile at every
sample budget.

# Evaluation protocol

For each selected checkpoint and each circuit:

1. Sample 1000 recipes.

2. Record:

    - recipe actions;
    - final circuit;

3. Compute recipe-level diversity metrics:

    - unique recipe fraction curve;
    - recipe entropy;
    - action entropy.

4. Map every resulting AIG with fixed 6-LUT setting.

5. Canonicalize and hash both AIG and mapped networks;

6. Record:

    - AIG size and depth;
    - number of k-LUTs and number of LUT levels after transformation;

7. Measure AIG and LUT diversity:

    - Number of unique (size, depth) pairs in AIG set
    - Number of unique (size, depth) pairs in k-LUT set

8. Measure final sample quality:

    - find the best circuit in terms of AIG depth
    - find the best circuit in terms of AIG size
    - find the best circuit in terms of the AIG size-depth product
    - find the best circuit in terms of LUT depth
    - find the best circuit in terms of LUT size
    - find the best circuit in terms of the LUT size-depth product
    - save the LUT pareto front
    - measure AIG-to-LUT rank correlation.
