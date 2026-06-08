## Hardware and execution time

All experiments were conducted on the following local hardware:
* **CPU:** Intel Core i7-12700H
* **GPU:** None (Trained on CPU)
* **RAM:** 16 GB

Wall-clock training times per dataset are in the results table below. Reported times include intermittent logging I/O (negligible, <1%).

## Running

Edit `config.yaml` to choose dataset, training protocol (OCC or UAD), model architecture, and noise settings, then:

```bash
python main.py --config config.yaml
```

If `experiment.write_to_file: true` in the config, results and the full config snapshot are written under `runs/{dataset}/{training_type}_{model_type}_{timestamp}/`.

## Part 2 — Results

I evaluated on three ADBench datasets (one easy, one medium, one harder) under both protocols (OCC and UAD) and with both model architectures (VanillaMLP and ResMLP). All other settings are fixed at the paper's defaults: Gaussian noise, sigma_max=2.0, m=3, noise_ratios=[0.5, 0.8, 1.0], 500 epochs, batch_size=256, lr=1e-4 with 0.1× decay at epoch 100, AMSGrad optimizer, seed=0.

### Main results

| Dataset      | d  | Protocol | Model      | Ours       | IForest | LOF    | Paper (Ours-MLP) | Training time |
|--------------|----|----------|------------|------------|---------|--------|------------------|---------------|
| 6_cardio     | 21 | OCC | ResMLP          | **0.9582** | 0.9565  | 0.9363 | 0.8601           | 23.0s         |
| 6_cardio     | 21 | OCC | VanillaMLP      | **0.9554** | 0.9565  | 0.9363 | 0.8601           | 13.8s         |
| 6_cardio     | 21 | UAD | ResMLP          | 0.9129     | 0.9281  | 0.6305 | —                | 58.1s         |
| 6_cardio     | 21 | UAD | VanillaMLP      | 0.9116     | 0.9281  | 0.6305 | —                | 21.7s         |
| 30_satellite | 36 | OCC | ResMLP          | **0.8257** | 0.7957  | 0.8449 | n/a              | 70.9s         |
| 30_satellite | 36 | OCC | VanillaMLP      | **0.8136** | 0.7957  | 0.8449 | n/a              | 39.6s         |
| 30_satellite | 36 | UAD | ResMLP          | 0.6071     | 0.7025  | 0.5660 | 0.8206           | 189.5s        |
| 30_satellite | 36 | UAD | VanillaMLP      | 0.5842     | 0.7025  | 0.5660 | 0.8206           | 93.0s         |
| 26_optdigits | 64 | OCC | ResMLP          | 0.8239     | 0.8417  | 0.9663 | 0.9800           | 247.2s        |
| 26_optdigits | 64 | OCC | VanillaMLP      | **0.9065** | 0.8417  | 0.9663 | 0.9800           | 113.9s        |
| 26_optdigits | 64 | UAD | ResMLP          | 0.4785     | 0.7138  | 0.5611 | 0.8943           | 308.4s        |
| 26_optdigits | 64 | UAD | VanillaMLP      | 0.4392     | 0.7138  | 0.5611 | 0.8943           | 126.4s        |

### Additional optdigits UAD experiment

The default config performed poorly on optdigits UAD. With a different noise configuration — **Uniform noise, sigma_max=1.5, m=5, noise_ratios=[0.5, 0.7, 1.0]**, VanillaMLP — AUC improved from 0.44 to 0.64. Still below the paper's reported 0.89, which suggests the paper's UAD numbers may rely on per-dataset tuning or implementation details I couldn't reproduce.

### Observations

**OCC works well.** On all three datasets, my method beats at least one baseline. On cardio and satellite, I beat IForest; on optdigits OCC, I beat IForest with VanillaMLP (0.91 vs 0.84) but lose to LOF (0.97).

**LOF dominates optdigits.** Both my method (0.91) and the paper (0.98) sit below LOF (0.97). This is consistent with the paper's own Table 6, where LOF (0.97) also beats their method (0.89) under UAD. Optdigits anomalies are categorically different digit classes — not perturbations of a normal class — so density-based methods like LOF have a structural advantage.

**UAD is consistently worse than OCC.** This is expected: the method's loss treats every training sample as normal, but UAD's training set contains anomalies. Forcing the model to predict zero noise for anomalies degrades the decision boundary. The drop is especially severe on optdigits (0.91 OCC → 0.48 UAD).

**Gap with paper's optdigits OCC (0.98 vs my 0.91).** ADBench's current `26_optdigits.npz` has d=64, while the paper's Table 4 lists d=65. This puts my model on the smaller-hidden-size branch (h=64); forcing h=256 actually hurt performance further in my tests (0.91 → 0.87), likely because the OCC training set (~281 samples) is too small for the larger architecture to generalize.

## Implementation decisions

The paper's pseudocode (Algorithm 1) is precise on the high-level loop but leaves several details ambiguous. The most consequential choices I made:

1. **`noise_ratio` interpretation.** The paper varies "noise_ratio in [0.2, 0.5, 0.8, 1.0]" but doesn't formally define it relative to Algorithm 1. I interpret it as: within each of the `m` chunks, only `noise_ratio` fraction of features receive noise; the remainder have `epsilon=0`. The post-chunk shuffle (Algorithm 1 step 10) randomizes which feature positions end up corrupted. So `noise_ratio=1.0` recovers Algorithm 1 as written.

2. **Multiple noise instances per clean sample.** The paper says "we generate 3 noise-augmented instances with 3 different noise ratios 0.5, 0.8, 1.0 at each epoch." I implemented this as a deterministic outer loop: for each clean sample, generate one noisy instance per `(noise_type, noise_ratio)` pair specified in the config. I tested random per-step selection of (type, ratio) and found deterministic iteration gave consistently better AUC — likely because each ratio gets stable gradient signal every batch instead of probabilistic coverage.

3. **Multiple noise types per training run.** The paper notes that using multiple noise distributions is optional. I implemented support for arbitrary combinations via the `used_noise_distributions` list in the config, with each distribution having its own `sigma_max` and `noise_ratios`. The reported runs use Gaussian only (paper's default).

4. **Protocol scaling.** OCC uses standardization (mean/std) as described in the paper's Experimental Settings. UAD uses MinMax scaling to match ADBench's `data_generator.generator()` default, which the paper inherits when reporting UAD numbers. The two protocols are therefore not directly comparable in the same table.

5. **Hidden size threshold.** The paper specifies `h=64` for `d ≤ 64` and `h=256` otherwise. I followed this exactly. Since all three datasets have `d ≤ 64`, all runs use `h=64`. I tested `h=256` on optdigits as a sanity check (since the paper's Table 4 lists d=65 for optdigits, which would have triggered h=256 in their setup) — it performed worse, likely due to overfitting on the small OCC training set.

## AI assistance

I used Claude (Anthropic) throughout this project as a discussion partner. Specifically:

- **Paper interpretation:** discussing ambiguous sections of the paper (noise_ratio semantics, protocol differences, theoretical assumptions).
- **Code writing:** Claude wrote the code based on my design decisions and direction. I specified the structure (class-based modules, config schema, noise generator API), made architectural choices (how to handle `noise_ratio`, whether to use multiple noise types per sample, how to structure OCC vs UAD), and reviewed/iterated on each module as it was written. Several refactors came out of these iterations (e.g. removing redundant noise masking after the shuffle, fixing a stale `delta` variable, simplifying the per-chunk distribution logic).
- **Math explanations:** clarifying Theorem 1 and 2 and their proofs.

The architectural decisions (config structure, class-based modules, noise generator API, OCC vs UAD interpretation), experimental design (which datasets, which configurations to compare), and the critique/observations in this README are mine. The code itself was written by Claude based on my direction. I did not find or use any public implementation of the paper.
