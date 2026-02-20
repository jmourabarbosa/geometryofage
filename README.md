# Geometry of PFC representations across adolescent development

Code and data for the Procrustes analysis of prefrontal cortex (PFC) neural tuning geometry across development, as part of:

> Zhu, J., Garin, C.M., Qi, X.-L., Machado, A., Wang, Z., Ben Hamed, S., Stanford, T.R., Salinas, E., Whitlow, C.T., Anderson, A.W., Zhou, X.M., Calabro, F.J., Luna, B. & Constantinidis, C. **Longitudinal measures of monkey brain structure and activity through adolescence predict cognitive maturation.** *Nature Neuroscience* 28, 2344--2355 (2025). https://doi.org/10.1038/s41593-025-02076-0

The [paper PDF](paper.pdf) is included in this repository.

## Overview

This repository asks: **does the geometry of PFC neural representations change during adolescent development?**

We use Procrustes analysis to compare the tuning curve geometry of PFC neurons across 8 monkeys tracked longitudinally from adolescence to adulthood. Neurons are grouped by monkey and age group, projected into a common PCA space, and aligned using Procrustes. The resulting distances quantify how similar neural representations are across individuals and developmental stages.

## Data

Two `.mat` files in `data_raw/` contain single-neuron recordings from dlPFC during working memory tasks:

- `odr_data_both_sig_is_best_20240109.mat` -- ODR task (1.5s and 3.0s delay), 8 monkeys, 1180 + 922 neurons
- `odrd_data_sig_on_best_20231018.mat` -- ODR with distractor task, 4 monkeys, 1319 neurons (20 conditions)

## Pipeline

The analysis is in [`analyses/pipeline.ipynb`](analyses/pipeline.ipynb), which imports all functions from `analyses/functions/`:

1. **Load data** -- parse `.mat` files, extract neuron metadata (monkey ID, age, maturation date)
2. **PSTHs and tuning curves** -- bin spikes, compute mean firing rates per condition and epoch
3. **Age groups** -- assign neurons to 3 age terciles per monkey
4. **PCA + Procrustes** -- z-score, reduce to 8 PCs, compute pairwise Procrustes distances
5. **Cross-monkey analysis** -- are cross-monkey distances larger than within-monkey?
6. **Cross-age analysis** -- are same-age cross-monkey pairs more similar than different-age pairs?
7. **Temporal analysis** -- sliding-window Procrustes distances with bootstrap CIs
8. **Age-pair comparisons** -- G0-G1 vs G1-G2 vs G0-G2 temporal trajectories

## Requirements

```
numpy
scipy
matplotlib
```

## License

See [LICENSE](cog_mat_ado_release/LICENSE).
