# Data generation

`build_dataset.py` downloads prices (Yahoo Finance) and macro series (FRED, ECB), builds the
features, the oracle Markowitz labels and the purged train / val / test split.

Run from the repository root:

```
python data_generation/build_dataset.py                 # base case: N = 63, no cap
python data_generation/build_dataset.py --horizon 21    # ablation: monthly horizon
python data_generation/build_dataset.py --wmax 0.25     # ablation: 25% cap per stock
python data_generation/build_dataset.py --synthetic     # offline test on simulated data
```

Output: `data/processed/N{N}_wmax{w}/` (X, Y, R_fwd, meta, returns, summary, all CSV).
Read it back with `load_dataset(folder)`, which also rebuilds `X_seq`.

To make sure the whole team uses identical data, one person runs the script and shares
`data/raw/prices.csv` and `data/raw/macro.csv`; the others put them in `data/raw/` and rerun.
