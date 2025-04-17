# IMDB Sentiment Analysis with RNN

## Overview
This project performs sentiment analysis on the IMDB dataset using a Recurrent Neural Network (RNN) with GRU layers. The goal is to classify movie reviews as positive (1) or negative (0). The model is optimized using hyperparameter tuning, and performance is evaluated with various metrics and visualizations.

## Dataset
* IMDB Dataset: 50,000 movie reviews (25,000 for training, 25,000 for testing).
* Reviews are preprocessed with a vocabulary of the 10,000 most frequent words and padded to a maximum length of 200.

## Model
* Architecture:
  * Embedding layer for word representations.
  * GRU layer for sequence modeling.
  * Dropout for regularization.
  * Dense layer with sigmoid activation for binary classification.
* Hyperparameter Tuning: RandomSearch with Keras Tuner to optimize embedding size, GRU units, dropout rate, and optimizer.
* Training: Early stopping to prevent overfitting, 10 epochs with 20% validation split.

## Evaluation
* Metrics:
  * Accuracy: 0.8849
  * AUC (ROC): 0.9533
  * Precision/Recall/F1-Score: ~0.89 for both classes
* Visualizations:
  * ROC Curve (AUC: 0.95)
  * Precision-Recall Curve
  * Confusion Matrix
  * Error Analysis: Examples of incorrect predictions

## Requirements
- Python 3.8+
- TensorFlow
- Keras Tuner
- Scikit-learn
- Matplotlib
- Seaborn
- NumPy
