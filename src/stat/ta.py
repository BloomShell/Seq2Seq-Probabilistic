#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""\
...
"""

__author__ = "BloomShell"
__version__ = "1.0.0"
__maintainer__ = "PietroAminPuddu"
__email__ = "pietroamin.puddu@gmail.com"
__status__ = "Development"


import pandas as pd
import numpy as np


def relative_strength_index(arr: pd.Series, window_length: int = 14):
    """
    Calculate the Relative Strength Index (RSI) for a given series of prices.

    Parameters:
    - prices: pd.Series
        A pandas Series containing the price data.
    - period: int, optional
        The look-back period for calculating RSI (default is 14).

    Returns:
    - pd.Series
        A pandas Series containing the RSI values.

    """
    assert isinstance(arr, pd.Series)
    # Calculate the differences in the closing prices
    delta = arr.diff()
    # Separate the positive and negative gains
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    # Calculate the average gain and loss
    avg_gain = gain.rolling(window=window_length, min_periods=1).mean()
    avg_loss = loss.rolling(window=window_length, min_periods=1).mean()
    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss
    # Calculate the RSI
    return 100 - (100 / (1 + rs))


def average_true_range(high, low, close, period=14):
    """
    Calculate the Average True Range (ATR) given high, low, and close price Series.

    Parameters:
    - high: pd.Series containing high prices
    - low: pd.Series containing low prices
    - close: pd.Series containing close prices
    - period: int, period for calculating ATR (default is 14)

    Returns:
    - pd.Series containing ATR values

    """
    assert all(isinstance(x, pd.Series) for x in (high, low, close))
    # Calculate True Range (TR)
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    # Calculate ATR
    return true_range.rolling(window=period, min_periods=1).mean()


def volume_weighted_average_price(high, low, close, volume):
    """
    Calculate the Volume Weighted Average Price (VWAP).

    Parameters:
    - high: pd.Series containing high prices
    - low: pd.Series containing low prices
    - close: pd.Series containing close prices
    - volume: pd.Series containing volume traded

    Returns:
    - pd.Series containing VWAP values

    """
    assert all(isinstance(x, pd.Series) for x in (high, low, close, volume))
    # Typical Price is calculated as the average of high, low, and close prices
    typical_price = (high + low + close) / 3
    # Calculate the cumulative values
    cumulative_price_volume = (typical_price * volume).cumsum()
    cumulative_volume = volume.cumsum()
    # Calculate VWAP
    return cumulative_price_volume / cumulative_volume
