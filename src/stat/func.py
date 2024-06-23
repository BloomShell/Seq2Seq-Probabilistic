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


def zscore(x: pd.Series, halflife: int = 25) -> pd.Series:
    """
    Calculate the z-score of a pandas Series
    using exponential weighted mean and standard deviation.

    Parameters:
    - x: pd.Series
        The input data series for which to calculate the z-score.
    - halflife: int, optional
        The halflife parameter for the exponential weighting (default is 25).

    Returns:
    - pd.Series
        A pandas Series containing the z-score values.

    """
    # Calculate the exponentially weighted moving average (EWMA)
    ewm_a = x.ewm(halflife=halflife).mean()
    # Calculate the exponentially weighted moving standard deviation (EWMSD)
    ewm_sd = x.ewm(halflife=halflife).std()
    # Calculate the z-score
    return (x - ewm_a) / ewm_sd
