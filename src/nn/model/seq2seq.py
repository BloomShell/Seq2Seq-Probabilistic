#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""\
This module provides a framework for defining various neural network architectures
for financial modeling tasks. It contains base classes and specific implementations
for different components of neural networks used in financial modeling.
"""

__author__ = "BloomShell"
__version__ = "1.0.0"
__maintainer__ = "PietroAminPuddu"
__email__ = "pietroamin.puddu@gmail.com"
__status__ = "Development"

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.nn.util import layer_init
import random


class Encoder(nn.Module):
    """
    The Encoder class processes the input sequence and returns the hidden states.

    Args:
        enc_feature_size (int): Number of features in the input sequence.
        hidden_size (int): Number of features in the hidden state of the GRU.
        num_gru_layers (int): Number of GRU layers.
        dropout (float): Dropout probability.
    """

    def __init__(self, enc_feature_size, hidden_size, num_gru_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            enc_feature_size,
            hidden_size,
            num_gru_layers,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, inputs):
        """
        Forward pass through the encoder.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch size, input seq len, num enc features).

        Returns:
            torch.Tensor: Output tensor of shape (batch size, input seq len, hidden size).
            torch.Tensor: Hidden states of shape (num gru layers, batch size, hidden size).
        """
        output, hidden = self.gru(inputs)
        return output, hidden


class DecoderBase(nn.Module):
    """
    Base class for decoders.

    Args:
        dec_target_size (int): Number of target features.
        target_indices (List[int]): Indices of the target features in the input.
        dist_size (int): Size of the distribution parameters.
        probabilistic (bool): Whether the decoder is probabilistic.
    """

    def __init__(self, dec_target_size, target_indices, dist_size, probabilistic):
        super().__init__()
        self.target_indices = target_indices
        self.target_size = dec_target_size
        self.dist_size = dist_size
        self.probabilistic = probabilistic

    def run_single_recurrent_step(self, inputs, hidden, enc_outputs):
        """
        Run a single recurrent step. This method should be implemented by subclasses.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch size, 1, num dec features).
            hidden (torch.Tensor): Hidden states of shape (num gru layers, batch size, hidden size).
            enc_outputs (torch.Tensor): Encoder outputs of shape (batch size, input seq len, hidden size).

        Returns:
            torch.Tensor: Output tensor of shape (batch size, 1, num targets, num dist params).
            torch.Tensor: Updated hidden states.
        """
        raise NotImplementedError()

    def forward(self, inputs, hidden, enc_outputs, teacher_force_prob=None):
        """
        Forward pass through the decoder.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch size, output seq length, num dec features).
            hidden (torch.Tensor): Hidden states of shape (num gru layers, batch size, hidden dim).
            enc_outputs (torch.Tensor): Encoder outputs of shape (batch size, input seq len, hidden size).
            teacher_force_prob (float, optional): Probability of using teacher forcing.

        Returns:
            torch.Tensor: Output tensor of shape (batch size, output seq len, num targets, num dist params).
        """
        batch_size, dec_output_seq_length, _ = inputs.shape

        # Store decoder outputs
        outputs = torch.zeros(
            batch_size,
            dec_output_seq_length,
            self.target_size,
            self.dist_size,
            dtype=torch.float,
        )

        curr_input = inputs[:, 0:1, :]

        for t in range(dec_output_seq_length):
            dec_output, hidden = self.run_single_recurrent_step(
                curr_input, hidden, enc_outputs
            )
            outputs[:, t : t + 1, :, :] = dec_output
            dec_output = Seq2Seq.sample_from_output(dec_output)

            teacher_force = (
                random.random() < teacher_force_prob
                if teacher_force_prob is not None
                else False
            )

            curr_input = inputs[:, t : t + 1, :].clone()
            if not teacher_force:
                curr_input[:, :, self.target_indices] = dec_output
        return outputs


class DecoderVanilla(DecoderBase):
    """
    A vanilla GRU decoder.

    Args:
        dec_feature_size (int): Number of features in the decoder input.
        dec_target_size (int): Number of target features.
        hidden_size (int): Number of features in the hidden state of the GRU.
        num_gru_layers (int): Number of GRU layers.
        target_indices (List[int]): Indices of the target features in the input.
        dropout (float): Dropout probability.
        dist_size (int): Size of the distribution parameters.
        probabilistic (bool): Whether the decoder is probabilistic.
    """

    def __init__(
        self,
        dec_feature_size,
        dec_target_size,
        hidden_size,
        num_gru_layers,
        target_indices,
        dropout,
        dist_size,
        probabilistic,
    ):
        super().__init__(dec_target_size, target_indices, dist_size, probabilistic)
        self.gru = nn.GRU(
            dec_feature_size,
            hidden_size,
            num_gru_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.out = layer_init(
            nn.Linear(hidden_size + dec_feature_size, dec_target_size * dist_size)
        )

    def run_single_recurrent_step(self, inputs, hidden, enc_outputs):
        """
        Run a single recurrent step of the vanilla GRU decoder.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch size, 1, num dec features).
            hidden (torch.Tensor): Hidden states of shape (num gru layers, batch size, hidden size).
            enc_outputs (torch.Tensor): Encoder outputs of shape (batch size, input seq len, hidden size).

        Returns:
            torch.Tensor: Output tensor of shape (batch size, 1, num targets, num dist params).
            torch.Tensor: Updated hidden states.
        """
        output, hidden = self.gru(inputs, hidden)
        output = self.out(torch.cat((output, inputs), dim=2))
        output = output.reshape(
            output.shape[0], output.shape[1], self.target_size, self.dist_size
        )
        return output, hidden


class Attention(nn.Module):
    """
    Attention mechanism for the decoder.

    Args:
        hidden_size (int): Number of features in the hidden state of the GRU.
        num_gru_layers (int): Number of GRU layers.
    """

    def __init__(self, hidden_size, num_gru_layers):
        super().__init__()
        self.attn = nn.Linear(2 * hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden_final_layer, encoder_outputs):
        """
        Forward pass through the attention mechanism.

        Args:
            decoder_hidden_final_layer (torch.Tensor): Final hidden state of the decoder (batch size, hidden size).
            encoder_outputs (torch.Tensor): Encoder outputs of shape (batch size, input seq len, hidden size).

        Returns:
            torch.Tensor: Attention weightings of shape (batch size, input seq len).
        """
        hidden = decoder_hidden_final_layer.unsqueeze(1).repeat(
            1, encoder_outputs.shape[1], 1
        )
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        weightings = F.softmax(attention, dim=1)
        return weightings


class DecoderWithAttention(DecoderBase):
    """
    A GRU decoder with attention mechanism.

    Args:
        dec_feature_size (int): Number of features in the decoder input.
        dec_target_size (int): Number of target features.
        hidden_size (int): Number of features in the hidden state of the GRU.
        num_gru_layers (int): Number of GRU layers.
        target_indices (List[int]): Indices of the target features in the input.
        dropout (float): Dropout probability.
        dist_size (int): Size of the distribution parameters.
        probabilistic (bool): Whether the decoder is probabilistic.
    """

    def __init__(
        self,
        dec_feature_size,
        dec_target_size,
        hidden_size,
        num_gru_layers,
        target_indices,
        dropout,
        dist_size,
        probabilistic,
    ):
        super().__init__(dec_target_size, target_indices, dist_size, probabilistic)
        self.attention_model = Attention(hidden_size, num_gru_layers)
        self.gru = nn.GRU(
            dec_feature_size + hidden_size,
            hidden_size,
            num_gru_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.out = layer_init(
            nn.Linear(
                hidden_size + hidden_size + dec_feature_size,
                dec_target_size * dist_size,
            )
        )

    def run_single_recurrent_step(self, inputs, hidden, enc_outputs):
        """
        Run a single recurrent step of the GRU decoder with attention.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch size, 1, num dec features).
            hidden (torch.Tensor): Hidden states of shape (num gru layers, batch size, hidden size).
            enc_outputs (torch.Tensor): Encoder outputs of shape (batch size, input seq len, hidden size).

        Returns:
            torch.Tensor: Output tensor of shape (batch size, 1, num targets, num dist params).
            torch.Tensor: Updated hidden states.
        """
        weightings = self.attention_model(hidden[-1], enc_outputs)
        weighted_sum = torch.bmm(weightings.unsqueeze(1), enc_outputs)
        output, hidden = self.gru(torch.cat((inputs, weighted_sum), dim=2), hidden)
        output = self.out(torch.cat((output, weighted_sum, inputs), dim=2))
        output = output.reshape(
            output.shape[0], output.shape[1], self.target_size, self.dist_size
        )
        return output, hidden


class Seq2Seq(nn.Module):
    """
    Sequence-to-sequence model.

    Args:
        encoder (nn.Module): Encoder module.
        decoder (nn.Module): Decoder module.
        lr (float): Learning rate.
        grad_clip (float): Gradient clipping threshold.
        probabilistic (bool): Whether the model is probabilistic.
    """

    def __init__(self, encoder, decoder, lr, grad_clip, probabilistic):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.opt = torch.optim.Adam(self.parameters(), lr)
        self.loss_func = nn.GaussianNLLLoss() if probabilistic else nn.L1Loss()
        self.grad_clip = grad_clip
        self.probabilistic = probabilistic

    @staticmethod
    def compute_smape(prediction, target):
        """
        Compute Symmetric Mean Absolute Percentage Error (sMAPE).

        Args:
            prediction (torch.Tensor): Predicted values.
            target (torch.Tensor): Target values.

        Returns:
            torch.Tensor: sMAPE value.
        """
        return (
            torch.mean(
                torch.abs(prediction - target)
                / ((torch.abs(target) + torch.abs(prediction)) / 2.0 + 1e-8)
            )
            * 100.0
        )

    @staticmethod
    def get_dist_params(output):
        """
        Get distribution parameters (mean and standard deviation) from the output.

        Args:
            output (torch.Tensor): Output tensor of shape (batch size, dec seq len, num targets, num dist params).

        Returns:
            torch.Tensor: Mean values.
            torch.Tensor: Standard deviation values.
        """
        mu = output[:, :, :, 0]
        sigma = F.softplus(output[:, :, :, 1])
        return mu, sigma

    @staticmethod
    def sample_from_output(output):
        """
        Sample from the output distribution.

        Args:
            output (torch.Tensor): Output tensor of shape (batch size, dec seq len, num targets, num dist params).

        Returns:
            torch.Tensor: Sampled output of shape (batch size, dec seq len, num targets).
        """
        if output.shape[-1] > 1:
            mu, sigma = Seq2Seq.get_dist_params(output)
            return torch.normal(mu, sigma)
        return output.squeeze(-1)

    def forward(self, enc_inputs, dec_inputs, teacher_force_prob=None):
        """
        Forward pass through the sequence-to-sequence model.

        Args:
            enc_inputs (torch.Tensor): Encoder input tensor of shape (batch size, input seq length, num enc features).
            dec_inputs (torch.Tensor): Decoder input tensor of shape (batch size, output seq length, num dec features).
            teacher_force_prob (float, optional): Probability of using teacher forcing.

        Returns:
            torch.Tensor: Output tensor of shape (batch size, output seq len, num targets, num dist params).
        """
        enc_outputs, hidden = self.encoder(enc_inputs)
        outputs = self.decoder(dec_inputs, hidden, enc_outputs, teacher_force_prob)
        return outputs

    def compute_loss(self, prediction, target, override_func=None):
        """
        Compute the loss.

        Args:
            prediction (torch.Tensor): Predicted values.
            target (torch.Tensor): Target values.
            override_func (callable, optional): Custom loss function.

        Returns:
            torch.Tensor: Loss value.
        """
        if self.probabilistic:
            mu, sigma = Seq2Seq.get_dist_params(prediction)
            var = sigma**2
            loss = self.loss_func(mu, target, var)
        else:
            loss = self.loss_func(prediction.squeeze(-1), target)
        return loss if self.training else loss.item()

    def optimize(self, prediction, target):
        """
        Optimize the model parameters.

        Args:
            prediction (torch.Tensor): Predicted values.
            target (torch.Tensor): Target values.

        Returns:
            float: Loss value.
        """
        self.opt.zero_grad()
        loss = self.compute_loss(prediction, target)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.opt.step()
        return loss.item()
