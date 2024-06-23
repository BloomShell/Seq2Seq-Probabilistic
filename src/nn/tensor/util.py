#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""\
"""

__author__ = "BloomShell"
__version__ = "1.0.0"
__maintainer__ = "PietroAminPuddu"
__email__ = "pietroamin.puddu@gmail.com"
__status__ = "Development"

import torch
import numpy as np
from typing import Tuple, List, Optional

class Scaler(object):
    """
    A class used to normalize and inverse transform tensors.

    Attributes:
    -----------
    mean : torch.Tensor
        The mean values computed for normalization.
    std : torch.Tensor
        The standard deviation values computed for normalization.
    """

    def __init__(self):
        """
        Initializes the scaler object with NaN values for mean and std.
        """
        self.mean = np.nan
        self.std = np.nan

    def fit(
        self, x: torch.Tensor, skip: Optional[List[int]] = None
    ) -> "Scaler":
        """
        Compute the mean and standard deviation of the input tensor for normalization.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor to fit.

        Returns:
        --------
        Scaler
            The fitted scaler object.
        """
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        if skip:
            self.mean[skip] = 0.0
            self.std[skip] = 1.0
        return self

    def fit_transform(
        self, x: torch.Tensor, skip: Optional[List[int]] = None
    ) -> Tuple["Scaler", torch.Tensor]:
        """
        Fits the scaler to the data and then transforms it.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor to fit and transform.

        Returns:
        --------
        Tuple[Scaler, torch.Tensor]
            The fitted scaler and the transformed tensor.
        """
        self.fit(x, skip)
        return self, self.transform(x)

    def transform(
        self, x: torch.Tensor, indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Apply the normalization to the input tensor using the computed mean and std.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor to transform.
        indices : Optional[List[int]]
            Specific indices to normalize. If None, all indices are used.

        Returns:
        --------
        torch.Tensor
            The normalized tensor.
        """
        indices = range(len(self.mean)) if indices is None else indices
        return (x - self.mean[indices]) / self.std[indices]

    def inverse_transform(
        self, x_scaled: torch.Tensor, indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Apply the inverse transformation to the normalized tensor to recover original values.

        Parameters:
        -----------
        x_scaled : torch.Tensor
            The normalized tensor to inverse transform.
        indices : Optional[List[int]]
            Specific indices to inverse transform. If None, all indices are used.

        Returns:
        --------
        torch.Tensor
            The tensor with original values recovered.
        """
        indices = range(len(self.mean)) if indices is None else indices
        return (x_scaled * self.std[indices]) + self.mean[indices]


def split_train_test_val(
    tensor: np.ndarray, train_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits data into training, validation, and test sets.

    Args:
        data (np.ndarray): The input data to split.
        train_size (float): The proportion of data to use for training.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The training, validation, and test sets.

    Raises:
        AssertionError: If train_size is not within the range [0, 1].
    """
    assert 0.0 <= train_size <= 1.0, "train_size must be between 0 and 1."
    num_timesteps = tensor.shape[0]
    train_end_indx = round(train_size * num_timesteps)
    train_data = tensor[:train_end_indx]
    test_end_indx = train_end_indx + round((1.0 - train_size) / 2 * num_timesteps)
    test_data = tensor[train_end_indx:test_end_indx]
    val_data = tensor[test_end_indx:]
    return train_data, val_data, test_data
