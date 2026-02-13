"""
Evaluation metrics for code generation
"""
import torch
import numpy as np
from collections import Counter
import re


def calculate_bleu(reference, candidate, max_n=4):
    """
    Calculate BLEU score
    
    Args:
        reference: Reference tokens (list)
        candidate: Candidate tokens (list)
        max_n: Maximum n-gram size
    
    Returns:
        BLEU score (0-1)
    """
    if len(candidate) == 0:
        return 0.0
    
    # Calculate n-gram precisions
    precisions = []
    
    for n in range(1, max_n + 1):
        ref_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference) - n + 1)])
        cand_ngrams = Counter([tuple(candidate[i:i+n]) for i in range(len(candidate) - n + 1)])
        
        overlap = sum((ref_ngrams & cand_ngrams).values())
        total = sum(cand_ngrams.values())
        
        if total == 0:
            precision = 0
        else:
            precision = overlap / total
        
        precisions.append(precision)
    
    # Brevity penalty
    bp = 1.0
    if len(candidate) < len(reference):
        bp = np.exp(1 - len(reference) / len(candidate))
    
    # Geometric mean of precisions
    if min(precisions) > 0:
        log_precisions = [np.log(p) for p in precisions]
        geo_mean = np.exp(sum(log_precisions) / len(log_precisions))
        bleu = bp * geo_mean
    else:
        bleu = 0.0
    
    return bleu


def calculate_exact_match(reference, candidate):
    """
    Calculate exact match accuracy
    
    Args:
        reference: Reference tokens (list)
        candidate: Candidate tokens (list)
    
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return 1.0 if reference == candidate else 0.0


def calculate_token_accuracy(reference, candidate):
    """
    Calculate token-level accuracy
    
    Args:
        reference: Reference tokens (list)
        candidate: Candidate tokens (list)
    
    Returns:
        Token accuracy (0-1)
    """
    if len(reference) == 0 or len(candidate) == 0:
        return 0.0
    
    # Pad to same length
    max_len = max(len(reference), len(candidate))
    ref_padded = reference + [''] * (max_len - len(reference))
    cand_padded = candidate + [''] * (max_len - len(candidate))
    
    # Count matches
    matches = sum(1 for r, c in zip(ref_padded, cand_padded) if r == c)
    
    return matches / max_len


def evaluate_model_outputs(model, test_loader, vocab, device, num_samples=100, 
                           is_attention_model=False, model_name="Model"):
    """
    Evaluate model and calculate metrics
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        vocab: Vocabulary object
        device: Device
        num_samples: Number of samples to evaluate
        is_attention_model: Whether model has attention
        model_name: Model name for logging
    
    Returns:
        metrics: Dictionary of evaluation metrics
        examples: List of example predictions (with attention if applicable)
    """
    model.eval()
    
    bleu_scores = []
    exact_matches = []
    token_accuracies = []
    examples = []
    
    count = 0
    
    with torch.no_grad():
        for src, tgt in test_loader:
            if count >= num_samples:
                break
            
            src = src.to(device)
            tgt = tgt.to(device)
            
            for i in range(src.size(0)):
                if count >= num_samples:
                    break
                
                src_seq = src[i:i+1]
                tgt_seq = tgt[i]
                
                # Generate prediction
                if is_attention_model:
                    predicted, attention_weights = model.generate(
                        src_seq, 
                        max_len=80,
                        sos_idx=vocab.sos_idx,
                        eos_idx=vocab.eos_idx
                    )
                else:
                    predicted = model.generate(
                        src_seq,
                        max_len=80,
                        sos_idx=vocab.sos_idx,
                        eos_idx=vocab.eos_idx
                    )
                    attention_weights = None
                
                # Convert to tokens (remove special tokens)
                reference_tokens = []
                for idx in tgt_seq.cpu().numpy():
                    if idx == vocab.eos_idx:
                        break
                    if idx not in [vocab.pad_idx, vocab.sos_idx]:
                        reference_tokens.append(vocab.idx2word[idx])
                
                predicted_tokens = []
                for idx in predicted:
                    if idx == vocab.eos_idx:
                        break
                    if idx not in [vocab.pad_idx, vocab.sos_idx]:
                        word = vocab.idx2word.get(idx, vocab.UNK_TOKEN)
                        predicted_tokens.append(word)
                
                # Calculate metrics
                bleu = calculate_bleu(reference_tokens, predicted_tokens)
                exact = calculate_exact_match(reference_tokens, predicted_tokens)
                token_acc = calculate_token_accuracy(reference_tokens, predicted_tokens)
                
                bleu_scores.append(bleu)
                exact_matches.append(exact)
                token_accuracies.append(token_acc)
                
                # Store examples (first 10)
                if count < 10:
                    src_text = vocab.decode(src_seq[0].cpu().numpy())
                    ref_text = ' '.join(reference_tokens)
                    pred_text = ' '.join(predicted_tokens)
                    
                    # Get source tokens for attention visualization
                    src_tokens = []
                    for idx in src_seq[0].cpu().numpy():
                        if idx == vocab.eos_idx:
                            break
                        if idx not in [vocab.pad_idx, vocab.sos_idx]:
                            src_tokens.append(vocab.idx2word[idx])
                    
                    example = {
                        'input': src_text,
                        'reference': ref_text,
                        'prediction': pred_text,
                        'bleu': bleu,
                        'exact_match': exact,
                        'token_accuracy': token_acc,
                        'source_tokens': src_tokens,
                        'target_tokens': predicted_tokens,
                    }
                    
                    if is_attention_model and attention_weights is not None:
                        example['attention'] = attention_weights
                    
                    examples.append(example)
                
                count += 1
    
    # Aggregate metrics
    metrics = {
        'bleu': np.mean(bleu_scores),
        'bleu_std': np.std(bleu_scores),
        'exact_match': np.mean(exact_matches),
        'token_accuracy': np.mean(token_accuracies),
        'num_samples': count
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"{model_name} - Evaluation Results")
    print(f"{'='*60}")
    print(f"Samples evaluated: {metrics['num_samples']}")
    print(f"BLEU Score: {metrics['bleu']:.4f} (±{metrics['bleu_std']:.4f})")
    print(f"Exact Match: {metrics['exact_match']:.4f}")
    print(f"Token Accuracy: {metrics['token_accuracy']:.4f}")
    print(f"{'='*60}\n")
    
    return metrics, examples
