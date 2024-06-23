# Seq2Seq-Probabilistic-Time-Series-Prediction-
 
## Project Overview
This repository contains a Seq2Seq (Sequence to Sequence) model designed for probabilistic forecasting. The Seq2Seq model is a type of neural network designed for sequence prediction tasks. This implementation extends the basic Seq2Seq model to support probabilistic forecasting, providing not only point predictions but also confidence intervals. This is particularly useful in applications like weather forecasting, realized volatility prediction, and other time series analysis tasks where it's important to estimate the range of possible future outcomes.

## Usage - Model Description
The model consists of several components:
- **Encoder**: Processes the input sequence and encodes it into a fixed-size context vector.
- **Decoder**: Takes the context vector and generates the output sequence.
- **Attention Mechanism**: (Optional) Focuses on relevant parts of the input sequence during decoding.

## Visualize the Predictions
<p align="center">
    <img src="https://i.imgur.com/NppBsrc.png" alt="Seq2Seq Model Visualization" width="900"/>
</p>
