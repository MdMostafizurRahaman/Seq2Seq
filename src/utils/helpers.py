"""
Helper utilities for training and visualization
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_training_history(histories, model_names, save_path=None):
    """
    Plot training curves for multiple models
    
    Args:
        histories: List of history dictionaries
        model_names: List of model names
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot training loss
    for history, name in zip(histories, model_names):
        axes[0].plot(history['epochs'], history['train_loss'], marker='o', label=name)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot validation loss
    for history, name in zip(histories, model_names):
        axes[1].plot(history['epochs'], history['val_loss'], marker='o', label=name)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Validation Loss Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    
    plt.show()


def plot_metrics_comparison(metrics_dict, save_path=None):
    """
    Plot metrics comparison across models
    
    Args:
        metrics_dict: Dictionary mapping model names to metrics
        save_path: Path to save figure
    """
    model_names = list(metrics_dict.keys())
    bleu_scores = [metrics_dict[name]['bleu'] for name in model_names]
    exact_matches = [metrics_dict[name]['exact_match'] for name in model_names]
    token_accs = [metrics_dict[name]['token_accuracy'] for name in model_names]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # BLEU scores
    axes[0].bar(model_names, bleu_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0].set_ylabel('BLEU Score')
    axes[0].set_title('BLEU Score Comparison')
    axes[0].set_ylim([0, max(bleu_scores) * 1.2])
    for i, v in enumerate(bleu_scores):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    # Exact match
    axes[1].bar(model_names, exact_matches, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1].set_ylabel('Exact Match Accuracy')
    axes[1].set_title('Exact Match Comparison')
    axes[1].set_ylim([0, max(exact_matches) * 1.2 if max(exact_matches) > 0 else 0.1])
    for i, v in enumerate(exact_matches):
        axes[1].text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom')
    
    # Token accuracy
    axes[2].bar(model_names, token_accs, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[2].set_ylabel('Token Accuracy')
    axes[2].set_title('Token Accuracy Comparison')
    axes[2].set_ylim([0, max(token_accs) * 1.2])
    for i, v in enumerate(token_accs):
        axes[2].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics comparison plot to {save_path}")
    
    plt.show()


def print_examples(examples, num_examples=5):
    """
    Print example predictions
    
    Args:
        examples: List of example dictionaries
        num_examples: Number of examples to print
    """
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS")
    print("="*80 + "\n")
    
    for i, ex in enumerate(examples[:num_examples]):
        print(f"Example {i+1}:")
        print(f"  Input:      {ex['input']}")
        print(f"  Reference:  {ex['reference']}")
        print(f"  Prediction: {ex['prediction']}")
        print(f"  BLEU: {ex['bleu']:.4f} | Exact Match: {ex['exact_match']:.0f}")
        print()


def count_parameters(model):
    """
    Count trainable parameters in model
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_attention_heatmap(attention_weights, source_tokens, target_tokens, save_path=None):
    """
    Plot attention heatmap for a single example
    
    Args:
        attention_weights: Attention weights [tgt_len, src_len]
        source_tokens: List of source tokens
        target_tokens: List of target tokens
        save_path: Path to save figure
    """
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot heatmap
    sns.heatmap(
        attention_weights,
        xticklabels=source_tokens,
        yticklabels=target_tokens,
        cmap='YlOrRd',
        cbar_kws={'label': 'Attention Weight'},
        ax=ax,
        vmin=0,
        vmax=1
    )
    
    ax.set_xlabel('Source (Docstring)', fontsize=12)
    ax.set_ylabel('Target (Generated Code)', fontsize=12)
    ax.set_title('Attention Weights Heatmap', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention heatmap to {save_path}")
    
    plt.show()


def plot_multiple_attention_heatmaps(examples_with_attention, vocab, save_path=None, num_examples=3):
    """
    Plot multiple attention heatmaps in a grid
    
    Args:
        examples_with_attention: List of dicts with 'attention', 'source', 'target'
        vocab: Vocabulary object
        save_path: Path to save figure
        num_examples: Number of examples to plot
    """
    import seaborn as sns
    
    num_examples = min(num_examples, len(examples_with_attention))
    fig, axes = plt.subplots(1, num_examples, figsize=(6*num_examples, 5))
    
    if num_examples == 1:
        axes = [axes]
    
    for idx, (ax, example) in enumerate(zip(axes, examples_with_attention[:num_examples])):
        attention = example['attention']  # [tgt_len, src_len]
        src_tokens = example['source_tokens']
        tgt_tokens = example['target_tokens']
        
        # Limit tokens for readability
        max_tokens = 15
        src_display = src_tokens[:max_tokens]
        tgt_display = tgt_tokens[:max_tokens]
        attention_display = attention[:len(tgt_display), :len(src_display)]
        
        sns.heatmap(
            attention_display,
            xticklabels=src_display,
            yticklabels=tgt_display,
            cmap='YlOrRd',
            cbar=True,
            ax=ax,
            vmin=0,
            vmax=1
        )
        
        ax.set_xlabel('Source', fontsize=10)
        ax.set_ylabel('Target', fontsize=10)
        ax.set_title(f'Example {idx+1}', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention heatmaps to {save_path}")
    
    plt.show()


def analyze_errors_by_type(examples, vocab):
    """
    Analyze errors by type: syntax, indentation, operators, etc.
    
    Args:
        examples: List of example predictions with reference and prediction
        vocab: Vocabulary object
    
    Returns:
        error_stats: Dictionary with error type counts
    """
    error_stats = {
        'syntax_errors': 0,
        'indentation_errors': 0,
        'operator_errors': 0,
        'variable_name_errors': 0,
        'keyword_errors': 0,
        'bracket_mismatch': 0,
        'total_errors': 0
    }
    
    python_keywords = {'def', 'return', 'if', 'else', 'elif', 'for', 'while', 'in', 'not', 'and', 'or'}
    operators = {'+', '-', '*', '/', '=', '==', '!=', '<', '>', '<=', '>='}
    brackets = {'(', ')', '[', ']', '{', '}'}
    
    for ex in examples:
        ref_tokens = ex['reference'].split()
        pred_tokens = ex['prediction'].split()
        
        if ref_tokens != pred_tokens:
            error_stats['total_errors'] += 1
            
            # Check syntax errors (def, return, etc.)
            ref_keywords = [t for t in ref_tokens if t in python_keywords]
            pred_keywords = [t for t in pred_tokens if t in python_keywords]
            if ref_keywords != pred_keywords:
                error_stats['keyword_errors'] += 1
            
            # Check operator errors
            ref_ops = [t for t in ref_tokens if t in operators]
            pred_ops = [t for t in pred_tokens if t in operators]
            if ref_ops != pred_ops:
                error_stats['operator_errors'] += 1
            
            # Check bracket matching
            ref_brackets = [t for t in ref_tokens if t in brackets]
            pred_brackets = [t for t in pred_tokens if t in brackets]
            if ref_brackets != pred_brackets:
                error_stats['bracket_mismatch'] += 1
            
            # Check indentation (colon indicates indentation needed)
            if ':' in ref_tokens and ':' not in pred_tokens:
                error_stats['indentation_errors'] += 1
            
            # Variable name errors (alphanumeric tokens)
            ref_vars = [t for t in ref_tokens if t.isalnum() and t not in python_keywords]
            pred_vars = [t for t in pred_tokens if t.isalnum() and t not in python_keywords]
            if ref_vars != pred_vars:
                error_stats['variable_name_errors'] += 1
    
    return error_stats


def plot_error_analysis(error_stats, save_path=None):
    """
    Plot error type distribution
    
    Args:
        error_stats: Dictionary with error counts
        save_path: Path to save figure
    """
    # Remove total_errors for plotting
    plot_stats = {k: v for k, v in error_stats.items() if k != 'total_errors'}
    
    labels = [k.replace('_', ' ').title() for k in plot_stats.keys()]
    values = list(plot_stats.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(labels, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F'])
    
    ax.set_ylabel('Number of Errors', fontsize=12)
    ax.set_title('Error Type Analysis', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved error analysis to {save_path}")
    
    plt.show()


def plot_performance_vs_length(examples, save_path=None):
    """
    Plot model performance vs docstring length
    
    Args:
        examples: List of examples with 'input', 'bleu', 'reference'
        save_path: Path to save figure
    """
    # Group by length
    length_groups = {}
    for ex in examples:
        length = len(ex['input'].split())
        if length not in length_groups:
            length_groups[length] = []
        length_groups[length].append(ex['bleu'])
    
    # Calculate average BLEU for each length
    lengths = sorted(length_groups.keys())
    avg_bleu = [np.mean(length_groups[l]) for l in lengths]
    counts = [len(length_groups[l]) for l in lengths]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Performance vs Length
    ax1.scatter(lengths, avg_bleu, s=100, alpha=0.6, c=avg_bleu, cmap='viridis')
    ax1.plot(lengths, avg_bleu, 'r--', alpha=0.5)
    ax1.set_xlabel('Docstring Length (tokens)', fontsize=12)
    ax1.set_ylabel('Average BLEU Score', fontsize=12)
    ax1.set_title('Performance vs Docstring Length', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sample count distribution
    ax2.bar(lengths, counts, color='#4ECDC4', alpha=0.7)
    ax2.set_xlabel('Docstring Length (tokens)', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Sample Distribution by Length', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance vs length plot to {save_path}")
    
    plt.show()


def save_model(model, optimizer, epoch, loss, save_path):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
    """
    import torch
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved to {save_path}")


def load_model(model, optimizer, load_path, device):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        load_path: Path to checkpoint
        device: Device to load model on
    
    Returns:
        epoch, loss: Loaded epoch and loss
    """
    import torch
    
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Model checkpoint loaded from {load_path}")
    print(f"  Epoch: {epoch}, Loss: {loss:.4f}")
    
    return epoch, loss
