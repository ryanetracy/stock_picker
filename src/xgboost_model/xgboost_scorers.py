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
from typing import Dict, Tuple

from src.load_data import get_option_chain, list_expiries, pick_expiries_for_horizons
from src.utils import normal_params_from_quantiles, trading_days_between

def simulate_terminal_prices(
    spot: float,
    mu: float,
    sigma: float,
    T_years: float,
    n_sims: int = 50_000,
    seed: int = 7
) -> np.ndarray:
    """simulate terminal underlying prices under a lognormal return assumption.

    Args:
        spot (float): current underlying spot price.
        mu (float): mean of the forecast return distribution (log-return).
        sigma (float): SD of the forecast return distribution
        T_years (float): time to expiration (years).
        n_sims (int, optional): number of monte carlo simulations. defaults to 
            50_000.
        seed (int, optional): random seed. defaults to 7.

    Returns:
        np.ndarray: simulated terminal price array of shape (n_sims).
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_sims)
    r = mu + sigma * z
    ST = spot * np.exp(r)
    return ST 


def payoff_call(ST: np.ndarray, K: float) -> np.ndarray:
    """compute payoff of long call option at expiration."""
    return np.maximum(ST - K, 0.0)


def payoff_put(ST: np.ndarray, K: float) -> np.ndarray:
    """compute payoff of long put option at expiration."""
    return np.maximum(K - ST, 0.0)


def payoff_vertical_call_spread(
    ST: np.ndarray, K_long: float, K_short: float
) -> np.ndarray:
    """compute payoff of a bull call vertical spread at expiration."""
    return payoff_call(ST, K_long) - payoff_call(ST, K_short)


def payoff_vertical_put_spread(
    ST: np.ndarray, K_short: float, K_long: float
) -> np.ndarray:
    """compute payoff of a bear put vertical spread at expiration."""
    return payoff_put(ST, K_long) - payoff_put(ST, K_short)


def bs_d1_d2(
    S: float, K: float, T: float, r: float, q: float, iv: float
) -> Tuple[float, float]:
    """compute black-scholes d1 and d2 parameters.

    Args:
        S (float): spot price.
        K (float): strike price.
        T (float): time to expiration (years).
        r (float): risk-free rate.
        q (float): dividend yield.
        iv (float): implied volatility.

    Returns:
        Tuple[float, float]: (d1, d2) parameters used in BS pricing.
    """
    if T <= 0 or iv <= 0:
        return np.nan, np.nan 

    d1 = (log(S/K) + (r - q + 0.5*iv*iv) * T) / (iv*sqrt(T))
    d2 = d1 - iv*sqrt(T)
    return d1, d2


def bs_price(
    S: float, 
    K: float, 
    T: float, 
    r: float = 0.04, 
    q: float = 0.0, 
    iv: float = 0.8, 
    opt_type: str = "call"
) -> float:
    """compute black-scholes theoretical option price.

    Args:
        S (float): spot price.
        K (float): strike price.
        T (float): time to expiration in years.
        r (float, optional): risk-free rate. defaults to 0.04.
        q (float, optional): dividend yield. defaults to 0.0.
        iv (float, optional): implied volatility. defaults to 0.8.
        opt_type (str, optional): 'call' or 'put'. defaults to 'call'.

    Returns:
        float: theoretical black-scholes option price.
    """
    d1, d2 = bs_d1_d2(S, K, T, r, q, iv)
    if np.isnan(d1):
        return np.nan 
    if opt_type == "call":
        return S * exp(-q*T) * norm.cdf(d1) - K * exp(-r*T) * norm.cdf(d2)
    else:
        return K * exp(-r*T) * norm.cdf(-d2) - S * exp(-q*T) * norm.cdf(-d1)


def bs_prob_itm(
    S: float, 
    K: float, 
    T: float, 
    r: float = 0.04, 
    q: float = 0.0, 
    iv: float = 0.8, 
    opt_type: str = "call"
) -> float:
    """estimate risk-neutral probability of finishing ITM under black-scholes.

    under BS risk-neutral, P(call ITM) ~ N(d2), P(put ITM) ~ N(-d2)

    Args:
        S (float): spot price.
        K (float): strike price.
        T (float): time to expiration in years.
        r (float, optional): risk-free rate. defaults to 0.04.
        q (float, optional): dividend yield. defaults to 0.0.
        iv (float, optional): implied volatility. defaults to 0.8.
        opt_type (str, optional): 'call' or 'put'. defaults to 'call'.

    Returns:
        float: approximate risk-neutral probability of expiring ITM.
    """
    d1, d2 = bs_d1_d2(S, K, T, r, q, iv)
    if np.isnan(d2):
        return np.nan 
    return float(norm.cdf(d2) if opt_type == "call" else norm.cdf(-d2))


def mid_price(row: pd.Series) -> float:
    """compute midpoint price from bid/ask with fallback to last trade."""
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
    """score single-leg options using monte carlo under model-implied distribution.

    simulates terminal prices, computes payoff and PnL for each strike, and
    estimates probability of ITM, probability of profit, and expected value.
    also compares model-implied ITM probability to IV-implied probability.

    Args:
        chain (pd.DataFrame): option chain (calls or puts).
        spot (float): current spot price.
        mu (float): forecast mean return.
        sigma (float): forecast return volatility.
        T_years (float): time to expiration in years.
        opt_type (str): 'call' or 'put'.
        r (float, optional): risk-free rate. defaults to 0.04.
        q (float, optional): eividend yield. defaults to 0.0.
        n_sims (int, optional): monte carlo simulations. defaults to 50_000.

    Returns:
        pd.DataFrame: scored options sorted by EV and probability of profit.
    """
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
    """score bull call spreads using monte carlo under forecast distribution.

    constructs candidate long/short call combinations and estimates expected
    value, probability of profit, max profit, and max loss.

    Args:
        calls (pd.DataFrame): call option chain.
        spot (float): current spot price.
        mu (float): forecast mean return.
        sigma (float): forecast return volatility.
        T_years (float): time to expiration in years.
        n_sims (int, optional): monte carlo simulations. defaults to 50_000.
        max_legs (int, optional): limit on strikes evaluated. defaults to 40.

    Returns:
        pd.DataFrame: scored bull call spreads sorted by EV.
    """
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
    """score bear put spreads using monte carlo under forecast distribution.

    constructs candidate long/short call combinations and estimates expected
    value, probability of profit, max profit, and max loss.

    Args:
        calls (pd.DataFrame): call option chain.
        spot (float): current spot price.
        mu (float): forecast mean return.
        sigma (float): forecast return volatility.
        T_years (float): time to expiration in years.
        n_sims (int, optional): monte carlo simulations. defaults to 50_000.
        max_legs (int, optional): limit on strikes evaluated. defaults to 40.

    Returns:
        pd.DataFrame: scored bear put spreads sorted by EV.
    """
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
) -> dict:
    """evaluate options across forecast horizons and nearest listed expiries.

    F\for each forecast horizon, maps to a listed expiration, derives the model-
    implied return distribution from quantiles, scores single-leg options and
    vertical spreads, and aggregates results into a structured dictionary.

    Args:
        ticker (str): underlying ticker symbol.
        quantile_forecasts (pl.DataFrame): forecast output containing horizon,
            spot, and return quantiles.
        r (float, optional): risk-free rate. defaults to 0.04.
        q (float, optional): dividend yield. defaults to 0.0.
        n_sims (int, optional): monte carlo simulations. defaults to 50_000.

    Returns:
        dict: nested dictionary keyed by horizon containing scored options.
    """
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


def _normalize_results(results: dict) -> dict:
    """normalize results dictionary keys to integers."""
    if isinstance(results, dict):
        return {int(k): v for k, v in results.items()}


def parse_options_results(
    results: dict, cols: list
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """flatten nested option scoring results into summary and detail tables.

    extracts summary statistics and concatenates calls, puts, and spread results
    across horizons into unified DataFrames.

    Args:
        results (dict): nested results from evaluate_options_across_expiries.
        cols (list): keys to extract for summary table.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            summary, calls, puts, bull spreads, bear spreads.
    """
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

        calls_df = payload.get("calls")
        puts_df = payload.get("puts")
        bull_df = payload.get("bull_call_spreads")
        bear_df = payload.get("bear_put_spreads")

        if isinstance(calls_df, pd.DataFrame) and not calls_df.empty:
            calls_parts.append(calls_df.assign(horizon=h))
        if isinstance(puts_df, pd.DataFrame) and not puts_df.empty:
            puts_parts.append(puts_df.assign(horizon=h))
        if isinstance(bull_df, pd.DataFrame) and not bull_df.empty:
            bull_call_parts.append(bull_df.assign(horizon=h))
        if isinstance(bear_df, pd.DataFrame) and not bear_df.empty:
            bear_put_parts.append(bear_df.assign(horizon=h))

    summary_df = pd.DataFrame(summary_rows, columns=["horizon", *cols])
    calls_out = pd.concat(calls_parts, ignore_index=True) if calls_parts else pd.DataFrame({"horizon": []})
    puts_out = pd.concat(puts_parts, ignore_index=True) if puts_parts else pd.DataFrame({"horizon": []})
    bull_calls_out = pd.concat(bull_call_parts, ignore_index=True) if bull_call_parts else pd.DataFrame({"horizon": []})
    bear_puts_out = pd.concat(bear_put_parts, ignore_index=True) if bear_put_parts else pd.DataFrame({"horizon": []})

    return summary_df, calls_out, puts_out, bull_calls_out, bear_puts_out


def option_score_single_leg(
    ev_model_per_contract: float, 
    premium_mid: float, 
    pop_model: float, 
    edge_itm: float
) -> float:
    """compute composite score for ranking single-leg options."""
    edge_wt = 1 / (1 + np.exp(-edge_itm * 10))
    return (ev_model_per_contract / (premium_mid * 100)) * pop_model * edge_wt


def make_trade_gates_single_leg(
    ev_model_per_contract: float, 
    edge_itm: float, 
    pop_model: float,
    empc_threshold: int = 10,
    ei_threshold: float = 0.03,
    pop_threshold: float = 0.45,
) -> float:
    """evaluate trade gating criteria for single-leg options."""
    passes = 0
    if ev_model_per_contract > empc_threshold:
        passes += 1
    if edge_itm > ei_threshold:
        passes += 1
    if pop_model > pop_threshold:
        passes += 1

    return passes / 3


def option_score_spread_roi(ev_per_contract: float, max_loss: float) -> float:
    """compute expected return on risk for a vertical spread."""
    return ev_per_contract / max_loss 


def option_score_spread_efficiency(
    ev_per_contract: float, max_loss: float, pop_model: float
) -> float:
    """compute risk-adjuted efficiency score for a spread."""
    return (ev_per_contract / max_loss) * pop_model 


def option_score_spread_max_ratio(max_profit: float, max_loss: float) -> float:
    """compute reward-to-risk ratio for a spread."""
    return max_profit / max_loss 


def score_options_results(
    results: dict,
    empc_threshold: int = 10,
    ei_threshold: float = 0.03,
    pop_threshold: float = 0.45,
    mr_threshold: float = 0.50,
    loss_threshold: int = 100
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """aggregate and rank scored options into decision-ready tables.

    applies trade gates and scoring functions to single-leg and spread
    results, filters by risk constraints, and returns top candidates
    per horizon and structure.

    Args:
        results (dict): nested output from evaluate_options_across_expiries.
        empc_threshold (int, optional): EV threshold for single legs.
        ei_threshold (float, optional): edge threshold for single legs.
        pop_threshold (float, optional): probability threshold.
        mr_threshold (float, optional): minimum reward-to-risk for spreads.
        loss_threshold (int, optional):maximum allowed risk per spread.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            summary table, top single-leg candidates, top spread candidates.
    """
    cols_list = [
        "expiry",
        "mu",
        "sigma",
        "spot",
        "pred_q10_px",
        "pred_q50_px",
        "pred_q90_px"
    ]

    summary_res, call_res, put_res, bull_res, bear_res = parse_options_results(
        results, cols_list
    )

    single_leg_res = pd.concat([call_res, put_res])

    single_leg_res.query(
        f"ev_model_per_contract > 0 & edge_itm > 0 & pop_model > {pop_threshold}",
        inplace=True
    )

    single_leg_res["option_score"] = single_leg_res.apply(
        lambda x: option_score_single_leg(
            x["ev_model_per_contract"], x["premium_mid"], x["pop_model"], x["edge_itm"]
        ), axis=1
    )

    single_leg_res["outcome_score"] = single_leg_res.apply(
        lambda x: make_trade_gates_single_leg(
            x["ev_model_per_contract"], x["edge_itm"], x["pop_model"]
        ), axis=1
    )

    spread_res = pd.concat([bull_res, bear_res])

    spread_res.query(
        f"ev_per_contract > 0 & pop_model > {pop_threshold}", inplace=True
    )

    spread_res["roi_ev"] = spread_res.apply(
        lambda x: option_score_spread_roi(x["ev_per_contract"], x["max_loss"]),
        axis=1
    )

    spread_res["efficiency_score"] = spread_res.apply(
        lambda x: option_score_spread_efficiency(
            x["ev_per_contract"], x["max_loss"], x["pop_model"]
        ), axis=1
    )

    spread_res["profit_ratio"] = spread_res.apply(
        lambda x: option_score_spread_max_ratio(
            x["max_profit"], x["max_loss"]
        ), axis=1
    )

    spread_res.query(
        f"profit_ratio >= {mr_threshold} & max_loss <= {loss_threshold}",
        inplace=True
    )

    return (
        summary_res,

        single_leg_res.sort_values(
            ["horizon", "option_score"],
            ascending=[True, False]
        ).groupby(
            ["horizon", "type"]
        ).head(5),

        spread_res.sort_values(
            ["horizon", "efficiency_score"], 
            ascending=[True, False]
        ).groupby(
            ["horizon", "spread"]
        ).head(5)
    )
