"""
collection of functions used to score the predictions from the trained xgboost
options models.
"""
import pandas as pd
import polars as pl
import numpy as np
from math import log, sqrt, exp 
from scipy.stats import norm 
from datetime import date 

from src.load_data import get_option_chain, list_expiries, pick_expiries_for_horizons
from src.utils import normal_params_from_quantiles, trading_days_between

def simulate_terminal_prices(
    spot: float,
    mu: float,
    sigma: float,
    T_years: float,
    n_sims: int = 50_000,
    seed: int = 7
):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_sims)
    r = mu + sigma * z
    ST = spot * np.exp(r)
    return ST 


def payoff_call(ST, K):
    return np.maximum(ST - K, 0.0)


def payoff_put(ST, K):
    return np.maximum(K - ST, 0.0)


def payoff_vertical_call_spread(ST, K_long, K_short):
    return payoff_call(ST, K_long) - payoff_call(ST, K_short)


def payoff_vertical_put_spread(ST, K_short, K_long):
    return payoff_put(ST, K_long) - payoff_put(ST, K_short)


def bs_d1_d2(S, K, T, r, q, iv):
    if T <= 0 or iv <= 0:
        return np.nan, np.nan 

    d1 = (log(S/K) + (r - q + 0.5*iv*iv) * T) / (iv*sqrt(T))
    d2 = d1 - iv*sqrt(T)
    return d1, d2


def bs_price(S, K, T, r=0.04, q=0.0, iv=0.8, opt_type="call"):
    d1, d2 = bs_d1_d2(S, K, T, r, q, iv)
    if np.isnan(d1):
        return np.nan 
    if opt_type == "call":
        return S * exp(-q*T) * norm.cdf(d1) - K * exp(-r*T) * norm.cdf(d2)
    else:
        return K * exp(-r*T) * norm.cdf(-d2) - S * exp(-q*T) * norm.cdf(-d1)


def bs_prob_itm(S, K, T, r=0.04, q=0.0, iv=0.8, opt_type="call"):
    """under BS risk-neutral, P(call ITM) ~ N(d2), P(put ITM) ~ N(-d2)"""
    d1, d2 = bs_d1_d2(S, K, T, r, q, iv)
    if np.isnan(d2):
        return np.nan 
    return float(norm.cdf(d2) if opt_type == "call" else norm.cdf(-d2))


def mid_price(row: pd.Series) -> float:
    b, a = row.get("bid", np.nan), row.get("ask", np.nan)
    if pd.notna(b) and pd.notna(a) and a > 0:
        return float((b + a) / 2)

    last = row.get("lastPrice", np.nan)
    return float(last) if pd.notna(last) else np.nan 


def score_single_leg(
    chain: pd.DataFrame,
    spot: float,
    mu: float,
    sigma: float,
    T_years: float,
    opt_type: str,
    r: float = 0.04,
    q: float = 0.0,
    n_sims: int = 50_000
) -> pd.DataFrame:
    ST = simulate_terminal_prices(spot, mu, sigma, T_years, n_sims=n_sims)

    rows = []
    for _, row in chain.iterrows():
        K = float(row["strike"])
        premium = mid_price(row)
        if not np.isfinite(premium) or premium <= 0:
            continue 

        if opt_type == "call":
            payoff = payoff_call(ST, K)
        else:
            payoff = payoff_put(ST, K)

        pnl = payoff - premium

        p_itm_model = float((payoff > 0).mean())
        pop_model = float((pnl > 0).mean())
        ev_model = float(pnl.mean())

        iv = row.get("impliedVolatility", np.nan)
        p_itm_mkt = (
            bs_prob_itm(
                spot, K, T_years, r=r, q=q, iv=float(iv), opt_type=opt_type
            )
            if pd.notna(iv) and float(iv) > 0
            else np.nan 
        )

        rows.append(
            {
                "type": opt_type,
                "strike": K,
                "premium_mid": premium,
                "iv": float(iv) if pd.notna(iv) else np.nan,
                "p_itm_model": p_itm_model,
                "p_itm_mkt_iv": p_itm_mkt,
                "edge_itm": (p_itm_model - p_itm_mkt) if np.isfinite(p_itm_mkt) else np.nan,
                "pop_model": pop_model,
                "ev_model_per_share": ev_model,
                "ev_model_per_contract": ev_model * 100.0
            }
        )

    df_out = pd.DataFrame(rows)
    if len(df_out) == 0:
        return df_out

    return df_out.sort_values(
        ["ev_model_per_contract", "pop_model"], ascending=False
    )


def score_vertical_call_spreads(
    calls: pd.DataFrame,
    spot: float,
    mu: float,
    sigma: float,
    T_years: float,
    n_sims: int = 50_000,
    max_legs: int = 40
) -> pd.DataFrame:
    ST = simulate_terminal_prices(spot, mu, sigma, T_years, n_sims=n_sims)

    calls2 = calls.copy()
    calls2["premium_mid"] = calls2.apply(mid_price, axis=1)
    calls2 = calls2.dropna(subset=["premium_mid"])
    calls2 = calls2.sort_values("strike")
    calls2 = calls2.head(max_legs)

    strikes = calls2["strike"].to_numpy(float)
    prem = calls2["premium_mid"].to_numpy(float)

    rows = []
    for i in range(len(strikes)):
        for j in range(i + 1, len(strikes)):
            K_long, K_short = strikes[i], strikes[j]
            debit = prem[i] - prem[j]
            if debit <= 0:
                continue 

            payoff = payoff_vertical_call_spread(ST, K_long, K_short)
            pnl = payoff - debit 
            rows.append(
                {
                    "spread": "bull_call",
                    "K_long": float(K_long),
                    "K_short": float(K_short),
                    "debit_mid": float(debit),
                    "pop_model": float((pnl > 0).mean()),
                    "ev_per_share": float(pnl.mean()),
                    "ev_per_contract": float(pnl.mean() * 100.0),
                    "max_profit": float((K_short - K_long - debit) * 100.0),
                    "max_loss": float(debit * 100.0)
                }
            )

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return pd.DataFrame(
            columns=[
                "spread",
                "K_long",
                "K_short",
                "debit_mid",
                "pop_model",
                "ev_per_share",
                "ev_per_contract",
                "max_profit",
                "max_loss",
            ]
        )
    return df_out.sort_values(["ev_per_contract", "pop_model"], ascending=False)


def score_vertical_put_spreads(
    puts: pd.DataFrame,
    spot: float,
    mu: float,
    sigma: float,
    T_years: float,
    n_sims: int = 50_000,
    max_legs: int = 40
) -> pd.DataFrame:
    ST = simulate_terminal_prices(spot, mu, sigma, T_years, n_sims=n_sims)

    puts2 = puts.copy()
    puts2["premium_mid"] = puts2.apply(mid_price, axis=1)
    puts2 = puts2.dropna(subset=["premium_mid"])
    puts2 = puts2.sort_values("strike")
    puts2 = puts2.head(max_legs)

    strikes = puts2["strike"].to_numpy(float)
    prem = puts2["premium_mid"].to_numpy(float)

    rows = []
    for i in range(len(strikes)):
        for j in range(i + 1, len(strikes)):
            K_short, K_long = strikes[i], strikes[j]
            if K_long <= K_short:
                continue 

            debit = prem[j] - prem[i]
            if debit <= 0:
                continue 

            payoff = payoff_vertical_put_spread(ST, K_short, K_long)
            pnl = payoff - debit 
            rows.append(
                {
                    "spread": "bear_put",
                    "K_long": float(K_long),
                    "K_short": float(K_short),
                    "debit_mid": float(debit),
                    "pop_model": float((pnl > 0).mean()),
                    "ev_per_share": float(pnl.mean()),
                    "ev_per_contract": float(pnl.mean() * 100.0),
                    "max_profit": float((K_long - K_short - debit) * 100.0),
                    "max_loss": float(debit * 100.0)
                }
            )

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return pd.DataFrame(
            columns=[
                "spread",
                "K_long",
                "K_short",
                "debit_mid",
                "pop_model",
                "ev_per_share",
                "ev_per_contract",
                "max_profit",
                "max_loss",
            ]
        )
    return df_out.sort_values(["ev_per_contract", "pop_model"], ascending=False)


def evaluate_options_across_expiries(
    ticker: str,
    quantile_forecasts: pl.DataFrame,
    r: float = 0.04,
    q: float = 0.0,
    n_sims: int = 50_000
):
    quantile_forecasts = quantile_forecasts.to_pandas()
    asof = date.today()
    out = {}

    listed_expiries = list_expiries(ticker)
    if not listed_expiries:
        raise ValueError(
            f"no listed expirations found for ticker `{ticker}`..."
        )
    listed_expiry_dates = [pd.to_datetime(s).date() for s in listed_expiries]

    for _, row in quantile_forecasts.iterrows():
        spot = float(row["spot"])
        H = int(row["horizon"])

        forecast_expiry = pd.to_datetime(row["date"]).date()
        expiry_date = min(
            listed_expiry_dates, key=lambda d: abs((d - forecast_expiry).days)
        )
        T_days = trading_days_between(asof, expiry_date)
        T_years = max(T_days / 252.0, 1e-6)

        mu, sigma = normal_params_from_quantiles(
            float(row["pred_q10_ret"]),
            float(row["pred_q50_ret"]),
            float(row["pred_q90_ret"])
        )

        # expiry_str = expiry_date.strftime("%Y-%m-%d")
        expiry_map = pick_expiries_for_horizons(
            ticker, 
            sorted(quantile_forecasts["horizon"].unique().tolist()), 
            asof=date.today()
        )
        expiry_str = expiry_map[H]
        calls, puts = get_option_chain(ticker, expiry_str)

        scored_calls = score_single_leg(
            calls, spot, mu, sigma, T_years, opt_type="call", r=r, q=q, n_sims=n_sims
        )
        scored_puts = score_single_leg(
            puts, spot, mu, sigma, T_years, opt_type="put", r=r, q=q, n_sims=n_sims
        )

        call_spreads = score_vertical_call_spreads(
            calls, spot, mu, sigma, T_years, n_sims=n_sims
        )
        put_spreads = score_vertical_put_spreads(
            puts, spot, mu, sigma, T_years, n_sims=n_sims
        )

        out[H] = {
            "expiry": expiry_str,
            "mu": mu,
            "sigma": sigma,
            "spot": spot,
            "pred_q10_px": float(row.get("pred_q10_px", float("nan"))),
            "pred_q50_px": float(row.get("pred_q50_px", float("nan"))),
            "pred_q90_px": float(row.get("pred_q90_px", float("nan"))),
            "calls": scored_calls,
            "puts": scored_puts,
            "bull_call_spreads": call_spreads,
            "bear_put_spreads": put_spreads
        }

    return out


def _normalize_results(results):
    if isinstance(results, dict):
        return {int(k): v for k, v in results.items()}


def parse_options_results(results, cols):
    r = _normalize_results(results)
    summary_rows = []
    calls_parts = []
    puts_parts = []
    bull_call_parts = []
    bear_put_parts = []

    for h, payload in sorted(r.items()):
        h = int(h)

        row = {"horizon": h}
        for c in cols:
            row[c] = payload.get(c)
        summary_rows.append(row)

        calls_df = pl.from_pandas(payload.get("calls"))
        puts_df = pl.from_pandas(payload.get("puts"))
        bull_df = pl.from_pandas(payload.get("bull_call_spreads"))
        bear_df = pl.from_pandas(payload.get("bear_put_spreads"))

        if calls_df is not None and calls_df.height > 0:
            calls_parts.append(
                calls_df.with_columns(pl.lit(h).alias("horizon"))
            )
        if puts_df is not None and puts_df.height > 0:
            puts_parts.append(puts_df.with_columns(pl.lit(h).alias("horizon")))
        if bull_df is not None and bull_df.height > 0:
            bull_call_parts.append(
                bull_df.with_columns(pl.lit(h).alias("horizon"))
            )
        if bear_df is not None and bear_df.height > 0:
            bear_put_parts.append(
                bear_df.with_columns(pl.lit(h).alias("horizon"))
            )

    summary_df = pl.DataFrame(summary_rows).select(["horizon", *cols])
    calls_out = pl.concat(calls_parts, how="vertical_relaxed") if calls_parts else pl.DataFrame({"horizon": []})
    puts_out = pl.concat(puts_parts, how="vertical_relaxed") if puts_parts else pl.DataFrame({"horizon": []})
    bull_calls_out = pl.concat(bull_call_parts, how="vertical_relaxed") if bull_call_parts else pl.DataFrame({"horizon": []})
    bear_puts_out = pl.concat(bear_put_parts, how="vertical_relaxed") if bear_put_parts else pl.DataFrame({"horizon": []})

    return summary_df, calls_out, puts_out, bull_calls_out, bear_puts_out
