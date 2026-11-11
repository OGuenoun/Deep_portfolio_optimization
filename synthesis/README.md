# Synthesis

`compare_models.ipynb` loads every run found in `models/*/results/*/`, evaluates all of them the
same way against `R_fwd.csv`, and compares them with the benchmarks (oracle, equal weight,
100% cash, rolling Markowitz).

Saved charts go in `figures/`.
