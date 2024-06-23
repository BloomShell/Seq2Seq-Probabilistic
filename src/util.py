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


import re
import itertools
from typing import List, Tuple, Callable


def str2num(arr):
    """
    Translate string numbers with 'k' for thousands and 'M' for millions,
    or just digits, and transform to float. Handle NaNs and both capital
    or not capital letters.

    Parameters:
    - num_series: pd.Series
        A pandas Series containing string representations of numbers.

    Returns:
    - pd.Series
        A pandas Series containing the corresponding float values.
    """
    _str2num: Callable = lambda x: re.sub(
        r"\((\d+?(?:.\d+))\)", r"-\1", str(x).replace(" ", "")
    )

    # Define mapping for suffixes and their respective multipliers
    suffixes = {"k": 1e3, "m": 1e6}

    # Convert to lowercase (to make case insensitive)
    arr = arr.astype(str).str.strip().str.lower().apply(_str2num)

    # Translate string numbers to float
    def translate_string_number(num_str):
        # Check if the last character is a suffix
        if num_str == "nan":
            return float("nan")
        if num_str[-1] in suffixes:
            suffix = num_str[-1]
            multiplier = suffixes[suffix]
            numeric_part = float(num_str[:-1])
        else:
            multiplier = 1
            numeric_part = float(num_str)
        return numeric_part * multiplier

    return arr.apply(translate_string_number)


def groupby(lst: List[Tuple], idxs: List[int]):
    """
    Group a list of tuples by specified indices.

    Parameters:
    - lst: List[Tuple[Any, ...]]
        A list of tuples to be grouped.
    - idxs: List[int]
        A list of indices specifying which elements of the tuples to use for grouping.

    Returns:
    - List[Tuple[Tuple[Any, ...], List[Tuple[Any, ...]]]]
        A list of tuples where the first element is the key (a tuple of the grouped elements)
        and the second element is a list of the original tuples that belong to that group.

    """
    assert isinstance(lst, list), "Input lst must be a list."
    # Ensure all elements in the list are tuples
    if all(isinstance(x, tuple) for x in lst):
        # Sort the list based on the specified indices
        lst.sort(key=lambda x: tuple(x[i] for i in idxs))
        # Group the list by the specified indices
        grouped = [
            (key, list(group))
            for key, group in itertools.groupby(
                lst, key=lambda x: tuple(x[i] for i in idxs)
            )
        ]
        return grouped
