#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module defines a Tensors class for handling sequence-to-sequence model inputs,
including methods for generating and scaling encoder/decoder inputs and targets,
and for splitting data into training, validation, and test sets.
"""

__author__ = "BloomShell"
__version__ = "1.0.0"
__maintainer__ = "PietroAminPuddu"
__email__ = "pietroamin.puddu@gmail.com"
__status__ = "Development"

import torch
import numpy as np
from dataclasses import dataclass
from src.nn.tensor.util import Scaler
from src.nn.tensor.util import split_train_test_val
from typing import List, Optional


class Tensors:
    """
    A class used to represent and process tensors for sequence-to-sequence models.
    """

    @dataclass
    class Container:
        enc_inputs: torch.Tensor
        dec_inputs: torch.Tensor
        dec_targets: torch.Tensor
        scalers: np.ndarray

        def __repr__(self) -> str:
            """
            Returns a string representation of the Container object.

            @returns {str} String representation of the Container object.
            """
            params = (
                f"{k}={repr(v)}"
                for k, v in {
                    "enc_inputs": self.enc_inputs.shape,
                    "dec_inputs": self.dec_inputs.shape,
                    "dec_targets": self.dec_targets.shape,
                    "scalers": self.scalers.shape,
                }.items()
            )

            return f"<{self.__class__.__qualname__}({', '.join(params)})>"

    def __init__(
        self,
        tensor: torch.Tensor,
        input_seq_length: int,
        output_seq_length: int,
        target_index: List[int],
        time_index: int,
        offset: int,
        train_test_val: bool = False,
        train_size: Optional[float] = None,
    ) -> None:
        """
        Initialize the Tensors object with the input sequence data and parameters.

        Parameters:
        tensor (torch.Tensor): Input tensor containing the sequence data.
        input_seq_length (int): Length of the input sequence.
        output_seq_length (int): Length of the output sequence.
        target_index (List[int]): List of indices for the target variables in the sequence.
        time_index (int): Index of the time variable in the input tensor.
        offset (int): Offset to apply when generating sequences.
        train_test_val (bool): If True, initialize training, validation, and test sets. Defaults to False.
        train_size (Optional[float]): Proportion of data to include in the training set if train_test_val is True.

        """
        # Type and value assertions for initialization parameters
        assert isinstance(
            tensor, torch.Tensor
        ), "Input tensor must be of type torch.Tensor."
        assert isinstance(
            target_index, list
        ), "Target index must be a list of integers."
        assert all(
            isinstance(x, int) for x in target_index
        ), "All elements in target_index must be integers."
        assert all(
            isinstance(x, int)
            for x in (input_seq_length, output_seq_length, time_index, offset)
        ), "input_seq_length, output_seq_length, time_index, and offset must be integers."
        assert (
            offset < output_seq_length
        ), "Offset must be less than output sequence length."

        # Find indices where the time_index column equals 1, ensuring the sequences are within bounds
        index = np.where(
            (tensor[:, time_index] == 1)
            & (np.arange(tensor.shape[0]) > (input_seq_length + offset))
            & (
                np.arange(tensor.shape[0])
                < (tensor.shape[0] - (output_seq_length - offset))
            )
        )[0]

        # Generate encoder input sequences using the found indices
        enc_inputs = torch.stack(
            [tensor[(i - offset - input_seq_length) : (i - offset)] for i in index]
        )

        # Generate decoder input sequences using the found indices
        dec_inputs = torch.stack(
            [
                tensor[(i - offset - 1) : (i - offset + output_seq_length - 1)]
                for i in index
            ]
        )

        # Generate decoder target sequences using the found indices and target_index
        dec_targets = torch.stack(
            [
                tensor[(i - offset) : (i - offset + output_seq_length), target_index]
                for i in index
            ]
        )

        # Ensure the shapes of enc_inputs, dec_inputs, and dec_targets match
        assert enc_inputs.shape[0] == dec_inputs.shape[0] == dec_targets.shape[0]

        # Scale the input sequences and store the results
        self.scalers = np.array(
            [
                Scaler().fit(enc_inputs[indx], skip=[-1])
                for indx in range(enc_inputs.shape[0])
            ]
        )
        self.enc_inputs = torch.stack(
            [
                self.scalers[indx].transform(enc_inputs[indx])
                for indx in range(enc_inputs.shape[0])
            ]
        )
        self.dec_inputs = torch.stack(
            [
                self.scalers[indx].transform(dec_inputs[indx])
                for indx in range(enc_inputs.shape[0])
            ]
        )
        self.dec_targets = torch.stack(
            [
                self.scalers[indx].transform(dec_targets[indx], indices=target_index)
                for indx in range(enc_inputs.shape[0])
            ]
        )

        # Optionally split the data into training, validation, and test sets
        if train_test_val:
            assert (
                train_size is not None
            ), "train_size must be provided if train_test_val is True."
            self.init_train_test_val(train_size)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Tensors object.

        @returns {str} String representation of the Tensors object.
        """
        params = (
            f"{k}={repr(v)}"
            for k, v in {
                "enc_inputs": self.enc_inputs.shape,
                "dec_inputs": self.dec_inputs.shape,
                "dec_targets": self.dec_targets.shape,
                "scalers": self.scalers.shape,
            }.items()
        )

        return f"<{self.__class__.__qualname__}({', '.join(params)})>"

    def init_train_test_val(self, train_size: float) -> None:
        """
        Split the data into training, validation, and test sets based on the provided training size.

        Parameters:
        train_size (float): The proportion of the data to include in the training set.
        """

        # Split the tensors and scalers into train, validation, and test sets
        train_enc_inputs, validation_enc_inputs, test_enc_inputs = split_train_test_val(
            self.enc_inputs, train_size=train_size
        )
        train_dec_inputs, validation_dec_inputs, test_dec_inputs = split_train_test_val(
            self.dec_inputs, train_size=train_size
        )
        train_dec_targets, validation_dec_targets, test_dec_targets = split_train_test_val(
            self.dec_targets, train_size=train_size
        )
        train_scalers, validation_scalers, test_scalers = split_train_test_val(
            self.scalers, train_size=train_size
        )

        # Create containers for train, validation, and test data
        self.train = self.Container(
            train_enc_inputs, train_dec_inputs, train_dec_targets, train_scalers
        )

        self.validation = self.Container(
            validation_enc_inputs, validation_dec_inputs, validation_dec_targets, validation_scalers
        )

        self.test = self.Container(
            test_enc_inputs, test_dec_inputs, test_dec_targets, test_scalers
        )
