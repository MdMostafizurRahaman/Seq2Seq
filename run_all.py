"""
Run all models training and evaluation
This script can be run directly with: python run_all.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_codesearchnet_data, create_dataloaders
from tokenizer import build_vocab_from_data
from models.vanilla_rnn import VanillaRNNSeq2Seq
from models.lstm_seq2seq import LSTMSeq2Seq
from models.lstm_attention import LSTMAttentionSeq2Seq
from training.train import train_model
from training.evaluate import evaluate_model_outputs
from utils.helpers import plot_training_history, plot_metrics_comparison, count_parameters

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Create results directory
os.makedirs('results', exist_ok=True)

def main():
    print("="*80)
    print("TEXT-TO-PYTHON CODE GENERATION USING SEQ2SEQ MODELS")
    print("="*80 + "\n")
    
    # 1. Load data
    print("Step 1: Loading data...")
    train_data, val_data, test_data = load_codesearchnet_data(
        num_train=10000, num_val=1000, num_test=1000
    )
    
    # 2. Build vocabulary
    print("\nStep 2: Building vocabulary...")
    vocab = build_vocab_from_data(train_data, val_data, min_freq=2)
    
    # 3. Create dataloaders
    print("\nStep 3: Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, vocab, batch_size=32,
        max_docstring_len=50, max_code_len=80
    )
    
    # Hyperparameters
    vocab_size = len(vocab)
    embed_size = 256
    hidden_size = 512
    num_epochs = 10
    learning_rate = 0.001
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    
    print(f"\nHyperparameters:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding size: {embed_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}\n")
    
    # Storage for results
    histories = []
    model_names = []
    all_metrics = {}
    
    # 4. Train Vanilla RNN
    print("\n" + "="*80)
    print("TRAINING MODEL 1/3: VANILLA RNN SEQ2SEQ")
    print("="*80)
    
    vanilla_rnn = VanillaRNNSeq2Seq(
        vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1
    ).to(device)
    print(f"Parameters: {count_parameters(vanilla_rnn):,}")
    
    vanilla_rnn_optimizer = optim.Adam(vanilla_rnn.parameters(), lr=learning_rate)
    vanilla_rnn_history = train_model(
        vanilla_rnn, train_loader, val_loader, vanilla_rnn_optimizer,
        criterion, device, num_epochs=num_epochs, model_name="Vanilla RNN",
        save_path="results/vanilla_rnn_best.pt", pad_idx=vocab.pad_idx
    )
    
    histories.append(vanilla_rnn_history)
    model_names.append("Vanilla RNN")
    
    vanilla_metrics, vanilla_examples = evaluate_model_outputs(
        vanilla_rnn, test_loader, vocab, device, num_samples=100, model_name="Vanilla RNN"
    )
    all_metrics["Vanilla RNN"] = vanilla_metrics
    
    # 5. Train LSTM
    print("\n" + "="*80)
    print("TRAINING MODEL 2/3: LSTM SEQ2SEQ")
    print("="*80)
    
    lstm_model = LSTMSeq2Seq(
        vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.1
    ).to(device)
    print(f"Parameters: {count_parameters(lstm_model):,}")
    
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
    lstm_history = train_model(
        lstm_model, train_loader, val_loader, lstm_optimizer,
        criterion, device, num_epochs=num_epochs, model_name="LSTM",
        save_path="results/lstm_best.pt", pad_idx=vocab.pad_idx
    )
    
    histories.append(lstm_history)
    model_names.append("LSTM")
    
    lstm_metrics, lstm_examples = evaluate_model_outputs(
        lstm_model, test_loader, vocab, device, num_samples=100, model_name="LSTM"
    )
    all_metrics["LSTM"] = lstm_metrics
    
    # 6. Train LSTM with Attention
    print("\n" + "="*80)
    print("TRAINING MODEL 3/3: LSTM WITH ATTENTION")
    print("="*80)
    
    lstm_attention = LSTMAttentionSeq2Seq(
        vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.1
    ).to(device)
    print(f"Parameters: {count_parameters(lstm_attention):,}")
    
    lstm_attention_optimizer = optim.Adam(lstm_attention.parameters(), lr=learning_rate)
    lstm_attention_history = train_model(
        lstm_attention, train_loader, val_loader, lstm_attention_optimizer,
        criterion, device, num_epochs=num_epochs, model_name="LSTM Attention",
        is_attention_model=True, save_path="results/lstm_attention_best.pt",
        pad_idx=vocab.pad_idx
    )
    
    histories.append(lstm_attention_history)
    model_names.append("LSTM Attention")
    
    lstm_attention_metrics, lstm_attention_examples = evaluate_model_outputs(
        lstm_attention, test_loader, vocab, device, num_samples=100,
        is_attention_model=True, model_name="LSTM Attention"
    )
    all_metrics["LSTM Attention"] = lstm_attention_metrics
    
    # 7. Plot comparisons
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    plot_training_history(histories, model_names, save_path="results/training_curves.png")
    plot_metrics_comparison(all_metrics, save_path="results/metrics_comparison.png")
    
    # 8. Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print("\nModel Performance Comparison:")
    print(f"{'Model':<20} {'BLEU':>10} {'Exact Match':>15} {'Parameters':>15}")
    print("-" * 65)
    
    models = [
        ("Vanilla RNN", vanilla_rnn, vanilla_metrics),
        ("LSTM", lstm_model, lstm_metrics),
        ("LSTM Attention", lstm_attention, lstm_attention_metrics)
    ]
    
    for name, model, metrics in models:
        params = count_parameters(model)
        print(f"{name:<20} {metrics['bleu']:>10.4f} {metrics['exact_match']:>15.4f} {params:>15,}")
    
    print("\n" + "="*80)
    print("Training complete! Results saved to 'results/' directory")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
