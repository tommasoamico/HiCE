# HiCE — Hierarchical Counterfactual Explanations

HiCE computes counterfactual explanations between whole *branches* of an
agglomerative clustering dendrogram, rather than between individual points.
Given a hierarchical clustering (Ward or complete linkage) over a dataset, it
answers: what is the minimal rigid translation of a branch `A` that makes it
dominate a target branch `B` over every competing branch in the tree, under a
chosen distance metric (Euclidean or Hamming)?

The core method solves this exactly via geometric/optimization techniques
(closed-form sphere projection, SDP, MILP via Gurobi). A geometric baseline
and a [DiCE](https://github.com/interpretml/DiCE)-based black-box baseline
(genetic search) are included for comparison.

## Repository structure

- `modules/hiceCounterfactuals.py` — the `hiceCounterfactuals` class implementing the exact method.
- `modules/utilityFunctions.py` — geometry/optimization helpers (sphere projection, dominance predicate, dataset utilities).
- `modules/metrics.py` — validity and proximity metrics used to evaluate a candidate counterfactual.
- `modules/dice.py` — DiCE-based competitor baseline.
- `runHierarchical.py` — experiment runner: builds a dendrogram over a dataset, enumerates branch pairs, and evaluates the method against the baselines.
- `data/` — local datasets used for experiments (not tracked in this repo — see below).

## Usage

```bash
# On a built-in/OpenML dataset (default: iris)
python runHierarchical.py --linkage ward --dataset iris --out results.csv

# On a categorical dataset with complete linkage
python runHierarchical.py --linkage complete --dataset kr-vs-kp --out results.csv

# Offline, no OpenML download
python runHierarchical.py --linkage ward --synthetic
```

Key options: `--min-leaf` / `--max-leaf` bound the size of branches considered,
`--max-pairs` caps the number of `(A, B)` pairs evaluated, `--eps` sets the
domination margin, and `--seed` controls pair sampling.

## Requirements

Python with `numpy`, `scipy`, `pandas`, `openml`, `miniball`, `cvxpy`, and
`gurobipy` (Gurobi with a valid license) for the exact solver; `dice-ml` if
running the DiCE baseline.

## Data

The `data/` directory holds local datasets used by the experiments and is
excluded from version control (see `.gitignore`) — it is not meant to be
distributed through this repository.
