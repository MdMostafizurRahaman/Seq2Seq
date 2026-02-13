# Text-to-Python Code Generation Using Seq2Seq Models

## Overview
This project implements and compares three different sequence-to-sequence models for automatic Python code generation from natural language descriptions (docstrings):

1. **Vanilla RNN-based Seq2Seq** - Baseline model with fixed-length context vector
2. **LSTM-based Seq2Seq** - Improved model handling long-term dependencies
3. **LSTM with Attention** - Advanced model with attention mechanism to overcome fixed-length bottleneck

## Dataset
- **Source**: CodeSearchNet Python Dataset from HuggingFace
- **Task**: English docstrings → Python code
- **Training samples**: 5,000-10,000 examples
- **Max lengths**: Docstring: 50 tokens, Code: 80 tokens

## Project Structure
```
seq2seq/
├── README.md
├── requirements.txt
├── text_to_python_colab.ipynb    # Main notebook for Google Colab
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Dataset loading and preprocessing
│   ├── tokenizer.py               # Vocabulary and tokenization
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vanilla_rnn.py         # Vanilla RNN Seq2Seq
│   │   ├── lstm_seq2seq.py        # LSTM Seq2Seq
│   │   └── lstm_attention.py      # LSTM with Attention
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py               # Training logic
│   │   └── evaluate.py            # Evaluation metrics
│   └── utils/
│       ├── __init__.py
│       └── helpers.py             # Utility functions
└── results/                        # Saved models and plots
```

## How to Run in Google Colab

1. **Upload to Google Drive or Clone from GitHub**
2. **Open `text_to_python_colab.ipynb` in Google Colab**
3. **Run all cells** - The notebook handles:
   - Installing dependencies
   - Loading and preprocessing data
   - Training all three models
   - Evaluating and comparing results
   - Visualizing performance metrics

## Features
- Complete data preprocessing pipeline
- Three model implementations with detailed comments
- Training with progress tracking
- Comprehensive evaluation metrics (BLEU, Accuracy, Loss)
- Visualization of results
- Generated code examples comparison

## Expected Results
- Vanilla RNN: Baseline performance, struggles with long sequences
- LSTM: Better handling of long-term dependencies
- LSTM + Attention: Best performance with attention mechanism

## Author
Assignment: Text-to-Python Code Generation Using Seq2Seq Models (RNNs)
