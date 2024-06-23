#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""\
This module provides a framework for defining various utils for neural 
network architectures for financial modeling tasks. 
"""

__author__ = "BloomShell"
__version__ = "1.0.0"
__maintainer__ = "PietroAminPuddu"
__email__ = "pietroamin.puddu@gmail.com"
__status__ = "Development"

from typing import Tuple, Iterable, Generator
from copy import deepcopy
import torch.nn as nn
import torch
import math
import time

# Define ANSI escape codes for colors and formatting
GREEN_TEXT = "\033[92m"
YELLOW_TEXT = "\033[93m"
RED_TEXT = "\033[91m"
BOLD_TEXT = "\033[1m"
RESET_TEXT = "\033[0m"


def layer_init(layer, w_scale=1.0):
    """
    Initializes a layer's weights and biases.

    Args:
        layer (nn.Module): Layer to initialize.
        w_scale (float): Scaling factor for weights.

    Returns:
        nn.Module: Initialized layer.
    """
    nn.init.kaiming_uniform_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0.0)
    return layer


def calc_teacher_force_prob(indx: int, decay: float = 10.0) -> float:
    """
    Calculate the teacher forcing probability using an inverse sigmoid decay.

    Args:
        indx (int): The current index, typically representing the epoch number.
        decay (float): The decay rate for the inverse sigmoid function.

    Returns:
        float: The calculated teacher forcing probability.
    """
    # Inverse sigmoid decay formula to calculate teacher forcing probability
    return decay / (decay + math.exp(indx / decay))


def batch_generator(
    data, batch_size: int, unscale: bool = False
) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    """
    Generate batches of data from the provided dataset.

    Parameters:
    data (Namespace or similar): The data object containing `enc_inputs`, `dec_inputs`, `dec_targets`, and `scalers`.
    batch_size (int): The size of each batch to generate.

    Yields:
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A batch consisting of:
        - batch_enc_inputs: Encoder inputs for the batch.
        - batch_dec_inputs: Decoder inputs for the batch.
        - batch_dec_targets: Decoder targets for the batch.
        - batch_scalers: Scalers used for the batch data.
    """
    enc_inputs, dec_inputs, dec_targets, scalers = (
        data.enc_inputs,
        data.dec_inputs,
        data.dec_targets,
        data.scalers,
    )
    
    # Randomly permute the indices of the dataset
    indices = torch.randperm(enc_inputs.shape[0])

    for i in range(0, len(indices), batch_size):
        # Select batch indices based on the current start position and batch size
        batch_indices = indices[i : i + batch_size]

        # Extract the corresponding data for the batch
        batch_enc_inputs = enc_inputs[batch_indices]
        batch_dec_inputs = dec_inputs[batch_indices]
        batch_dec_targets = dec_targets[batch_indices]
        batch_scalers = scalers[batch_indices]

        # Break the loop if the last batch is smaller than the specified batch size
        if batch_enc_inputs.shape[0] < batch_size:
            break

        # Yield the batch as a tuple of encoder inputs, decoder inputs, decoder targets, and scalers
        yield batch_enc_inputs, batch_dec_inputs, batch_dec_targets, batch_scalers
            

def train(
    model: nn.Module,
    train_data: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, any]],
    batch_size: int,
    teacher_force_prob: float,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_data (Iterable): The training data generator or loader, which yields batches.
        batch_size (int): The size of each training batch.
        teacher_force_prob (float): The probability of using teacher forcing during training.

    Returns:
        float: The average loss over all batches for this epoch.
    """
    model.train()  # Set the model to training mode
    epoch_loss = 0.0
    num_batches = 0

    for batch_enc_inputs, batch_dec_inputs, batch_dec_targets, _ in batch_generator(
        train_data, batch_size
    ):
        # Forward pass through the model
        output = model(batch_enc_inputs, batch_dec_inputs, teacher_force_prob)
        # Compute and apply the optimization step
        loss = model.optimize(output, batch_dec_targets)
        epoch_loss += loss
        num_batches += 1

    # Return the average loss for the epoch
    return epoch_loss / num_batches


def evaluate(
    model: nn.Module,
    val_data: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, any]],
    batch_size: int,
) -> float:
    """
    Evaluate the model on validation data.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        val_data (Iterable): The validation data generator or loader, which yields batches.
        batch_size (int): The size of each validation batch.

    Returns:
        float: The average loss over all validation batches.
    """
    model.eval()  # Set the model to evaluation mode
    epoch_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # Disable gradient calculation
        for (
            batch_enc_inputs,
            batch_dec_inputs,
            batch_dec_targets,
            _,
        ) in batch_generator(val_data, batch_size):
            # Forward pass through the model
            output = model(batch_enc_inputs, batch_dec_inputs)
            # Compute the loss
            loss = model.compute_loss(output, batch_dec_targets)
            epoch_loss += loss
            num_batches += 1

    # Return the average loss for the evaluation
    return epoch_loss / num_batches


def fit(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    num_epochs: int,
    batch_size: int,
) -> nn.Module:
    """
    Train and validate the model over multiple epochs.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_data (torch.Tensor): The training dataset.
        val_data (torch.Tensor): The validation dataset.
        num_epochs (int): The number of epochs to train.
        batch_size (int): The size of each batch.

    Returns:
        nn.Module: The best model based on validation loss.
    """
    assert isinstance(model, nn.Module), "model must be an instance of nn.Module"
    assert all(
        isinstance(x, int) for x in (num_epochs, batch_size)
    ), "num_epochs and batch_size must be integers"

    best_val_loss = float("inf")
    best_model = None

    for epoch in range(num_epochs):
        start_time = time.time()  # Track time for each epoch
        teacher_force_prob = calc_teacher_force_prob(epoch)

        # Training step
        train_loss = train(model, train_data, batch_size, teacher_force_prob)

        # Evaluation step
        val_loss = evaluate(model, val_data, batch_size)

        # Check if the current model is the best based on validation loss
        new_best_val = False
        if val_loss < best_val_loss:
            new_best_val = True
            best_val_loss = val_loss
            best_model = deepcopy(model)

        # Log the epoch results
        log_color = (
            GREEN_TEXT
            if new_best_val
            else YELLOW_TEXT
            if val_loss < 1.2 * best_val_loss
            else RED_TEXT
        )
        print(
            f"{BOLD_TEXT}║ EPOCH {epoch + 1:02d}\t ║"
            f" TRAIN LOSS: {log_color}{train_loss:.5f}{RESET_TEXT}\t ║"
            f" VAL LOSS: {log_color}{val_loss:.5f}{RESET_TEXT}\t ║"
            f" TEACH: {GREEN_TEXT}{teacher_force_prob:.2f}{RESET_TEXT}\t ║"
            f" TIME: {YELLOW_TEXT}{(time.time() - start_time):.1f}s{RESET_TEXT}\t ║"
            f"{' (NEW BEST)' if new_best_val else ''}"
        )

    return best_model
