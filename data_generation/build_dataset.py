"""
Projet 4 - Supervised learning of a Markowitz strategy with a neural network.
Dataset builder, French equity universe (15 stocks) + a cash asset.

Builds X_t (state vector), Y_t (oracle Markowitz weights on (t+1, t+N)),
R_fwd (forward compounded returns), label_end (for purging) and the
train / validation / test split.

Usage
-----
    pip install pandas numpy scikit-learn cvxpy yfinance requests truststore
    python build_dataset.py                       # base case: N = 63, no weight cap
    python build_dataset.py --horizon 21          # ablation: monthly horizon
    python build_dataset.py --wmax 0.25           # ablation: 25% cap per stock
    python build_dataset.py --synthetic           # offline test on simulated data

Outputs (in --out, default ./data), all CSV
    raw/prices.csv, raw/macro.csv                 raw downloads (re-used if present)
    raw/manual/*.csv                              optional hand-saved macro files
    processed/N{N}_wmax{w}/X.csv                  features, one row per date t
    processed/N{N}_wmax{w}/Y.csv                  oracle weights (15 stocks + CASH)
    processed/N{N}_wmax{w}/R_fwd.csv              compounded returns over (t+1, t+N)
    processed/N{N}_wmax{w}/meta.csv               label_end, label_type, split (train/val/test/none)
    processed/N{N}_wmax{w}/returns.csv            daily stock returns, to rebuild X_seq
    processed/N{N}_wmax{w}/summary.csv            sizes, split counts, checks, oracle Sharpe

Read them back with load_dataset(folder), which also rebuilds X_seq.

Labels
------
Max-Sharpe over the stocks (long-only, cap w_max per stock). When no capped stock
portfolio has a positive excess return over the window, the label is 100% cash.
Y has one column per stock plus a last column CASH.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
TICKERS = [
    "MC.PA", "OR.PA", "BN.PA",              # luxury / consumer
    "TTE.PA", "VIE.PA",                     # energy / utilities
    "SAN.PA",                               # health
    "BNP.PA", "CS.PA",                      # financials
    "AIR.PA", "SU.PA", "DG.PA", "SGO.PA",   # industrials
    "AI.PA",                                # materials
    "ORA.PA", "CAP.PA",                     # telecom / tech
]
MARKET = "^FCHI"   # CAC 40 index: market factor for features only, not tradable here
CASH = "CASH"      # earns the euro risk-free rate
ASSETS = TICKERS + [CASH]

FRED_SERIES = ["VIXCLS", "DGS10", "DGS2", "DBAA", "DAAA"]   # global risk proxies
ECB_EURIBOR3M = "https://data-api.ecb.europa.eu/service/data/FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA?format=csvdata"

MOM_WINDOWS = [5, 21, 63, 126, 252]
VOL_WINDOWS = [21, 63]
DOWNSIDE_WINDOW = 63
DD_WINDOW = 252
CORR_WINDOW = 63
N_EIG = 3
SEQ_LEN = 60
WARMUP = max(MOM_WINDOWS + [DD_WINDOW])  # 252 days before the first usable t

SPLITS = {  # split on label_end (see doc): sample belongs to the set if label_end < end
    "train": ("2006-01-01", "2017-01-01"),
    "val": ("2017-01-01", "2020-01-01"),
    "test": ("2020-01-01", "2100-01-01"),
}

# ----------------------------------------------------------------------------
# 1. Raw data
# ----------------------------------------------------------------------------
def download_prices(start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    px = yf.download(TICKERS + [MARKET], start=start, end=end, auto_adjust=True, progress=False)["Close"]
    px = px[TICKERS + [MARKET]].sort_index()
    first = px.apply(lambda s: s.first_valid_index())
    late = first[first > pd.Timestamp(start) + pd.Timedelta(days=10)]
    if len(late):
        print("WARNING - these series start after --start, check them:\n", late)
    return px


def _fetch_csv(url: str, manual_file: Path) -> io.StringIO | Path:
    """Download a CSV with the operating system's certificate store (works behind
    corporate proxies / antivirus HTTPS scanning). If the download fails, fall back
    to a file saved by hand in data/raw/manual/."""
    if manual_file.exists():
        print(f"  using manual file {manual_file}")
        return manual_file
    try:
        import truststore                     # pip install truststore (Python >= 3.10)
        truststore.inject_into_ssl()          # also covers urllib below
    except ImportError:
        pass
    import time
    import urllib.request

    def with_requests():
        import requests                       # default User-Agent: FRED drops fake browser ones
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return r.text

    def with_urllib():
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.read().decode("utf-8")

    exc = None
    for attempt in range(3):
        for method in (with_requests, with_urllib):
            try:
                text = method()
                time.sleep(0.5)               # be gentle with the servers between series
                return io.StringIO(text)
            except Exception as e:            # try the other method, then retry after a pause
                exc = e
        print(f"  retrying {url.split('?')[0]} ({exc.__class__.__name__})")
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(
        f"\nCould not download after 3 attempts:\n  {url}\nReason: {exc}\n"
        f"Fix: open the URL in your browser, save the file as\n  {manual_file.resolve()}\n"
        f"and run again (the script uses the saved file and skips the download)."
    ) from None


def download_macro(start: str, end: str, manual_dir: Path) -> pd.DataFrame:
    manual_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for code in FRED_SERIES:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}&cosd={start}&coed={end}"
        src = _fetch_csv(url, manual_dir / f"{code}.csv")
        s = pd.read_csv(src, index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
        frames.append(pd.to_numeric(s, errors="coerce").rename(code))
    src = _fetch_csv(ECB_EURIBOR3M, manual_dir / "EURIBOR3M.csv")
    ecb = pd.read_csv(src, usecols=["TIME_PERIOD", "OBS_VALUE"])
    # monthly average: dated at the START of the following month, i.e. once it is published
    idx = pd.PeriodIndex(ecb["TIME_PERIOD"], freq="M").to_timestamp() + pd.offsets.MonthBegin(1)
    frames.append(pd.Series(ecb["OBS_VALUE"].to_numpy(), index=idx, name="EURIBOR3M"))
    return pd.concat(frames, axis=1).sort_index().loc[start:end]


def synthetic_data(start: str, end: str, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlated GBM with two volatility regimes, plus fake macro series. For offline tests only."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T, n = len(dates), len(TICKERS)
    regime = (np.sin(np.arange(T) / 400) + 0.3 * rng.standard_normal(T) > 0.6).astype(float)
    base_vol = np.linspace(0.16, 0.35, n) / np.sqrt(252)
    corr = 0.6 * np.ones((n, n)) + 0.4 * np.eye(n)        # one equity market: high correlation
    z = rng.standard_normal((T, n)) @ np.linalg.cholesky(corr).T
    vol = base_vol * (1 + 1.5 * regime[:, None])
    drift = (0.08 - 0.40 * regime[:, None]) / 252
    r = drift + vol * z
    prices = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=dates, columns=TICKERS)
    prices[MARKET] = 4000 * np.exp(np.cumsum(r.mean(axis=1)))
    vix = 15 + 20 * pd.Series(regime, index=dates).rolling(10, min_periods=1).mean() + rng.normal(0, 1, T)
    macro = pd.DataFrame(
        {
            "VIXCLS": vix,
            "DGS10": 3 + 0.001 * np.cumsum(rng.standard_normal(T)),
            "DGS2": 2 + 0.001 * np.cumsum(rng.standard_normal(T)),
            "DBAA": 6 + regime + 0.1 * rng.standard_normal(T),
            "DAAA": 5 + 0.1 * rng.standard_normal(T),
            "EURIBOR3M": np.clip(1.0 + 0.001 * np.cumsum(rng.standard_normal(T)), -0.6, None),
        },
        index=dates,
    )
    return prices, macro


def load_raw(out: Path, start: str, end: str, synthetic: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if synthetic:
        return synthetic_data(start, end)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    p_file, m_file = raw / "prices.csv", raw / "macro.csv"
    old_p = raw / "prices.parquet"            # from an earlier version of this script
    if not p_file.exists() and old_p.exists():
        try:
            pd.read_parquet(old_p).to_csv(p_file)
            print(f"  converted {old_p.name} to {p_file.name} (no new download)")
        except ImportError:
            pass
    if p_file.exists():
        prices = _read_csv_dated(p_file)
    else:
        prices = download_prices(start, end)
        prices.to_csv(p_file)                # saved now, so a failed macro step does not re-download
    if m_file.exists():
        macro = _read_csv_dated(m_file)
    else:
        macro = download_macro(start, end, raw / "manual")
        macro.to_csv(m_file)
    return prices, macro


def _read_csv_dated(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


# ----------------------------------------------------------------------------
# 2. Cleaning and alignment
# ----------------------------------------------------------------------------
def clean(raw_prices: pd.DataFrame, macro: pd.DataFrame):
    """Returns stock prices, stock returns, market returns, aligned macro, daily rf."""
    px = raw_prices.dropna(how="all")                       # calendar = Euronext trading days
    px = px.ffill(limit=2).dropna(how="any")                # short gaps only
    rets_all = px.pct_change().iloc[1:]
    prices = px[TICKERS]
    rets = rets_all[TICKERS]
    mkt = rets_all[MARKET]
    macro = macro.reindex(px.index.union(macro.index)).ffill().reindex(px.index)
    macro = macro.shift(1)                                  # value for t known at close of t
    rf = (macro["EURIBOR3M"] / 100 / 252).reindex(rets.index).fillna(0.0)  # can be negative (2015-2022)
    return prices, rets, mkt, macro, rf


# ----------------------------------------------------------------------------
# 3. Features X_t  (every window ends at t; stocks only, no cash features)
# ----------------------------------------------------------------------------
def build_features(prices, rets, mkt, macro) -> pd.DataFrame:
    prices = prices.loc[rets.index[0] - pd.Timedelta(days=10):]
    r = rets.reindex(prices.index)
    m_r = mkt.reindex(prices.index)
    feats = {}
    for h in MOM_WINDOWS:
        for tk, s in (prices / prices.shift(h) - 1).items():
            feats[f"mom_{h}__{tk}"] = s
    for h in VOL_WINDOWS:
        for tk, s in (r.rolling(h).std() * np.sqrt(252)).items():
            feats[f"vol_{h}__{tk}"] = s
    down = np.sqrt((r.clip(upper=0) ** 2).rolling(DOWNSIDE_WINDOW).mean()) * np.sqrt(252)
    for tk, s in down.items():
        feats[f"dvol_{DOWNSIDE_WINDOW}__{tk}"] = s
    dd = prices / prices.rolling(DD_WINDOW).max() - 1
    for tk, s in dd.items():
        feats[f"dd_{DD_WINDOW}__{tk}"] = s
    for tk in r.columns:
        feats[f"corrmkt_{CORR_WINDOW}__{tk}"] = r[tk].rolling(CORR_WINDOW).corr(m_r)

    # top eigenvalue shares of the rolling correlation matrix of the stocks (global)
    vals = r.to_numpy()
    eig = np.full((len(r), N_EIG), np.nan)
    for i in range(CORR_WINDOW, len(r) + 1):
        w = vals[i - CORR_WINDOW:i]
        if np.isnan(w).any():
            continue
        ev = np.linalg.eigvalsh(np.corrcoef(w, rowvar=False))[::-1]
        eig[i - 1] = ev[:N_EIG] / len(ev)
    for k in range(N_EIG):
        feats[f"eig{k + 1}__global"] = pd.Series(eig[:, k], index=r.index)

    # market (CAC 40) state
    m_level = (1 + m_r.fillna(0)).cumprod()
    feats["market__mom_63"] = m_level / m_level.shift(63) - 1
    feats["market__vol_21"] = m_r.rolling(21).std() * np.sqrt(252)

    m = macro.reindex(prices.index)
    feats["macro__vix"] = m["VIXCLS"]
    feats["macro__vix_chg21"] = m["VIXCLS"] - m["VIXCLS"].shift(21)
    feats["macro__us_term_spread"] = m["DGS10"] - m["DGS2"]
    feats["macro__us_credit_spread"] = m["DBAA"] - m["DAAA"]
    feats["macro__euribor3m"] = m["EURIBOR3M"]
    return pd.DataFrame(feats).reindex(rets.index)


# ----------------------------------------------------------------------------
# 4. Labels Y_t  (oracle Markowitz on (t+1, t+N))
# ----------------------------------------------------------------------------
class MarkowitzSolver:
    """Parametrised max-Sharpe QP compiled once, re-solved for every date."""

    def __init__(self, n: int, w_max: float):
        self.n, self.w_max = n, w_max
        self.Lt = cp.Parameter((n, n))       # Sigma = L L^T  ->  w' Sigma w = ||L^T w||^2
        self.mu = cp.Parameter(n)
        self.y = cp.Variable(n)
        self.problem = cp.Problem(
            cp.Minimize(cp.sum_squares(self.Lt @ self.y)),
            [self.mu @ self.y == 1, self.y >= 0, self.y <= w_max * cp.sum(self.y)],
        )

    def best_capped_return(self, mu: np.ndarray) -> float:
        """max mu'w over the capped simplex (greedy): if <= 0 no positive Sharpe exists."""
        left, total = 1.0, 0.0
        for m in np.sort(mu)[::-1]:
            take = min(self.w_max, left)
            total += take * m
            left -= take
            if left <= 1e-12:
                break
        return total

    def solve(self, window: np.ndarray, rf_window: np.ndarray) -> tuple[np.ndarray, int]:
        """Returns (weights on stocks + cash, label_type). label_type 1 = max-Sharpe, 0 = cash."""
        cash_only = np.r_[np.zeros(self.n), 1.0]
        mu = (window - rf_window[:, None]).mean(axis=0) * 252    # annualised excess return
        if self.best_capped_return(mu) <= 1e-8:
            return cash_only, 0
        sigma = LedoitWolf().fit(window).covariance_ * 252
        self.Lt.value = np.linalg.cholesky(sigma + 1e-10 * np.eye(self.n)).T
        self.mu.value = mu
        self.problem.solve(solver=cp.CLARABEL)
        if self.problem.status not in ("optimal", "optimal_inaccurate") or self.y.value is None:
            return cash_only, 0
        y = np.clip(self.y.value, 0, None)
        return np.r_[y / y.sum(), 0.0], 1


def build_labels(rets: pd.DataFrame, rf: pd.Series, idx: np.ndarray, N: int, w_max: float):
    R, RF = rets.to_numpy(), rf.to_numpy()
    n = R.shape[1]
    solver = MarkowitzSolver(n, w_max)
    Y = np.zeros((len(idx), n + 1))
    kind = np.zeros(len(idx), dtype=np.int8)
    R_fwd = np.zeros_like(Y)
    for j, i in enumerate(idx):
        win, rf_win = R[i + 1:i + N + 1], RF[i + 1:i + N + 1]
        Y[j], kind[j] = solver.solve(win, rf_win)
        R_fwd[j, :n] = np.prod(1 + win, axis=0) - 1
        R_fwd[j, n] = np.prod(1 + rf_win) - 1                 # cash
        if j % 500 == 0:
            print(f"  labels {j:>5}/{len(idx)}")
    return Y, kind, R_fwd


# ----------------------------------------------------------------------------
# 5. Assemble, split, check, save
# ----------------------------------------------------------------------------
def build(args) -> dict:
    out = Path(args.out)
    raw_prices, raw_macro = load_raw(out, args.start, args.end, args.synthetic)
    prices, rets, mkt, macro, rf = clean(raw_prices, raw_macro)
    X_df = build_features(prices, rets, mkt, macro)

    dates = rets.index
    T, N = len(dates), args.horizon
    valid = (~X_df.isna().any(axis=1)).to_numpy().copy()
    valid[:max(WARMUP, SEQ_LEN - 1)] = False
    valid[T - N:] = False                                    # no full label window
    idx = np.flatnonzero(valid)
    print(f"{len(idx)} samples, {X_df.shape[1]} features, {len(TICKERS)} stocks + cash")

    Y, kind, R_fwd = build_labels(rets, rf, idx, N, args.wmax)
    R = rets.to_numpy()
    X_seq = np.stack([R[i - SEQ_LEN + 1:i + 1] for i in idx])  # last 60 stock returns ending at t
    d = dates[idx]
    label_end = dates[idx + N]

    split = {}
    for name, (lo, hi) in SPLITS.items():
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        split[name] = np.flatnonzero(d >= lo) if name == "test" else np.flatnonzero((d >= lo) & (label_end < hi))

    X = X_df.iloc[idx].to_numpy(dtype=np.float64)
    checks = run_checks(X, Y, R_fwd, X_seq, idx, dates, label_end, R, N, args.wmax,
                        prices, rets, mkt, macro, X_df)

    # oracle backtest, rebalancing every N days (sanity check of label alignment)
    reb = np.arange(0, len(idx), N)
    per_year = 252 / N
    sharpe = lambda x: float(x.mean() / x.std() * np.sqrt(per_year))
    cash_ret = R_fwd[reb, -1]
    oracle = (Y[reb] * R_fwd[reb]).sum(axis=1) - cash_ret
    ew = R_fwd[reb, :-1].mean(axis=1) - cash_ret
    summary = {
        "n_samples": int(len(idx)),
        "n_features": int(X.shape[1]),
        "assets": ASSETS,
        "first_date": str(d[0].date()),
        "last_date": str(d[-1].date()),
        "horizon_N": N,
        "w_max": args.wmax,
        "split_counts": {k: int(len(v)) for k, v in split.items()},
        "cash_label_share": float((kind == 0).mean()),
        "oracle_excess_sharpe": sharpe(oracle),
        "equal_weight_stocks_excess_sharpe": sharpe(ew),
        "checks": checks,
        "synthetic": bool(args.synthetic),
    }

    # ---- save: plain CSV files, one folder per (N, w_max) so ablation runs do not overwrite
    folder = out / "processed" / f"N{N}_wmax{args.wmax:g}"
    folder.mkdir(parents=True, exist_ok=True)
    fmt = "%.10g"
    idx_name = pd.Index(d, name="date")
    pd.DataFrame(X, index=idx_name, columns=X_df.columns).to_csv(folder / "X.csv", float_format=fmt)
    pd.DataFrame(Y, index=idx_name, columns=ASSETS).to_csv(folder / "Y.csv", float_format=fmt)
    pd.DataFrame(R_fwd, index=idx_name, columns=ASSETS).to_csv(folder / "R_fwd.csv", float_format=fmt)
    split_col = np.full(len(idx), "none", dtype=object)    # samples in a purge gap belong to no set
    for name, pos in split.items():
        split_col[pos] = name
    pd.DataFrame(
        {"label_end": label_end.strftime("%Y-%m-%d"), "label_type": kind, "split": split_col},
        index=idx_name,
    ).to_csv(folder / "meta.csv")
    rets.rename_axis("date").to_csv(folder / "returns.csv", float_format=fmt)   # to rebuild X_seq
    flat = {k: v for k, v in summary.items() if k not in ("checks", "split_counts", "assets")}
    flat.update({f"n_{k}": v for k, v in summary["split_counts"].items()})
    flat.update({f"check_{k}": v for k, v in checks.items()})
    flat["assets"] = " ".join(ASSETS)
    pd.Series(flat, name="value").rename_axis("key").to_csv(folder / "summary.csv")
    print(f"Saved CSV files in {folder}")
    return summary


def load_dataset(folder: str | Path, seq_len: int = SEQ_LEN) -> dict:
    """Read the CSV files written by build() back into arrays for training.

    Returns a dict with X, Y, R_fwd (DataFrames indexed by date), meta (label_end,
    label_type, split), X_seq (numpy array, samples x seq_len x stocks) and the
    integer positions of the train / val / test samples.

        from build_dataset import load_dataset
        ds = load_dataset("data/processed/N63_wmax1")
        X_train, Y_train = ds["X"].iloc[ds["train_idx"]], ds["Y"].iloc[ds["train_idx"]]
    """
    folder = Path(folder)
    read = lambda name: _read_csv_dated(folder / name)
    X, Y, R_fwd, rets = read("X.csv"), read("Y.csv"), read("R_fwd.csv"), read("returns.csv")
    meta = read("meta.csv")
    meta["label_end"] = pd.to_datetime(meta["label_end"])
    pos = rets.index.get_indexer(X.index)
    R = rets.to_numpy()
    X_seq = np.stack([R[i - seq_len + 1:i + 1] for i in pos])
    out = {"X": X, "Y": Y, "R_fwd": R_fwd, "meta": meta, "X_seq": X_seq}
    for name in ("train", "val", "test"):
        out[f"{name}_idx"] = np.flatnonzero(meta["split"].to_numpy() == name)
    return out


def run_checks(X, Y, R_fwd, X_seq, idx, dates, label_end, R, N, w_max,
               prices, rets, mkt, macro, X_df) -> dict:
    res = {}
    res["no_nan"] = bool(np.isfinite(X).all() and np.isfinite(Y).all() and np.isfinite(R_fwd).all())
    res["weights_sum_to_1"] = bool(np.allclose(Y.sum(axis=1), 1, atol=1e-6))
    res["weights_in_bounds"] = bool(
        (Y >= -1e-8).all() and (Y[:, :-1] <= w_max + 1e-4).all() and (Y[:, -1] <= 1 + 1e-8).all()
    )
    res["label_end_is_t_plus_N"] = bool((label_end == dates[idx + N]).all())

    j = len(idx) // 2  # label alignment: recompute one forward return by hand from prices
    i = idx[j]
    p = prices.reindex(rets.index).to_numpy()
    res["r_fwd_matches_prices"] = bool(np.allclose(p[i + N] / p[i] - 1, R_fwd[j, :-1], rtol=1e-8, atol=1e-10))
    res["x_seq_ends_at_t"] = bool(np.allclose(X_seq[j, -1], R[i]))

    # look-ahead: features rebuilt on data truncated at D must equal the full build for t <= D
    D = dates[idx[len(idx) // 2]]
    X_trunc = build_features(prices.loc[:D], rets.loc[:D], mkt.loc[:D], macro.loc[:D])
    common = X_trunc.dropna().index
    res["no_lookahead_in_features"] = bool(
        np.allclose(X_trunc.loc[common].to_numpy(), X_df.loc[common].to_numpy(), equal_nan=True)
    )
    failed = [k for k, v in res.items() if not v]
    if failed:
        raise RuntimeError(f"Dataset checks failed: {failed}")
    print("All checks passed:", ", ".join(res))
    return res


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2004-12-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--horizon", type=int, default=63, help="label window N in trading days (63 = 1 quarter)")
    ap.add_argument("--wmax", type=float, default=1.0, help="max weight per stock; 1.0 = no cap (cash is never capped)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--synthetic", action="store_true", help="simulated data, no download")
    return ap.parse_args()


if __name__ == "__main__":
    s = build(parse_args())
    print(json.dumps({k: v for k, v in s.items() if k != "checks"}, indent=2)) 