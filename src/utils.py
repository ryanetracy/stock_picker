"""
utility functions
    build_forecast_dates
"""
from __future__ import annotations

from datetime import timedelta, date, datetime
import pandas as pd
from typing import List, Union
import numpy as np
import math

def build_forecast_dates(last_date, horizon_days: int) -> pd.DatetimeIndex:
    """return the next `horizon_days` trading dates (mon-fri), starting AFTER `last_date`.

    Args:
        last_date (_type_): final date of the training window.
        horizon_days (int): number of days for the forecast to project.

    Returns:
        pd.DatetimeIndex: index of datetimes for the forecast.
    """
    if isinstance(last_date, str):
        last_date = pd.to_datetime(last_date)

    dates = []
    d = pd.to_datetime(last_date)

    while len(dates) < horizon_days:
        d = d + pd.Timedelta(days=1)
        if d.weekday() < 5:
            dates.append(d)

    return pd.DatetimeIndex(dates)


def next_trading_day(d: datetime) -> datetime:
    """advance to next weekday (mon-fri)"""
    d = d + timedelta(days=1)
    while d.weekday() >= 5:
        d = d + pd.Timedelta(days=1)
    return d


def build_trading_future_dates(
    last_date: datetime, n_trading_days: int
) -> List[datetime]:
    """return next N trading dates strictly after `last_date`"""
    out = []
    d = last_date
    for _ in range(n_trading_days):
        d = next_trading_day(d)
        out.append(d)
    return out


def trading_days_between(start: date, end: date) -> int:
    """count trading days from start (exclusive) to end (inclusive)"""
    if end <= start:
        return 0

    days = 0
    cur = start 
    while cur < end:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            days += 1
    return days


def normal_params_from_quantiles(
    q10: float, q50: float, q90: float
) -> List[float]:
    """
    for a normal: q(p) = mu + z_p * sigma
    z_0.10 = -1.28155, z_0.50 = 0, z_0.90 = +1.28155
    """
    z = 1.2815515655446004
    mu = q50
    sigma = (q90 - q10) / (2 * z)
    sigma = max(sigma, 1e-9)
    return mu, sigma
