"""
Training module for Seq2Seq models
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import time


def train_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio=0.5, 
                is_attention_model=False, pad_idx=0):
    """
    Train for one epoch
    
    Args:
        model: Seq2Seq model
        dataloader: Training DataLoader
        optimizer: Optimizer
        criterion: Loss function
        device: Device (cuda/cpu)
        teacher_forcing_ratio: Probability of teacher forcing
        is_attention_model: Whether model has attention (returns extra outputs)
        pad_idx: Padding token index
    
    Returns:
        avg_loss: Average loss for the epoch
    """
    model.train()
    epoch_loss = 0
    
    for src, tgt in tqdm(dataloader, desc="Training", leave=False):
        src = src.to(device)
        tgt = tgt.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if is_attention_model:
            output, _ = model(src, tgt, teacher_forcing_ratio)
        else:
            output = model(src, tgt, teacher_forcing_ratio)
        
        # Reshape for loss calculation
        # output: [batch_size, tgt_len, vocab_size]
        # tgt: [batch_size, tgt_len]
        output_dim = output.shape[-1]
        
        output = output[:, 1:].contiguous().view(-1, output_dim)  # Skip <SOS>
        tgt = tgt[:, 1:].contiguous().view(-1)  # Skip <SOS>
        
        # Calculate loss (ignoring padding)
        loss = criterion(output, tgt)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, is_attention_model=False, pad_idx=0):
    """
    Evaluate model on validation/test set
    
    Args:
        model: Seq2Seq model
        dataloader: Validation/Test DataLoader
        criterion: Loss function
        device: Device (cuda/cpu)
        is_attention_model: Whether model has attention
        pad_idx: Padding token index
    
    Returns:
        avg_loss: Average loss
    """
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Forward pass (no teacher forcing during evaluation)
            if is_attention_model:
                output, _ = model(src, tgt, teacher_forcing_ratio=0)
            else:
                output = model(src, tgt, teacher_forcing_ratio=0)
            
            # Reshape for loss calculation
            output_dim = output.shape[-1]
            
            output = output[:, 1:].contiguous().view(-1, output_dim)
            tgt = tgt[:, 1:].contiguous().view(-1)
            
            # Calculate loss
            loss = criterion(output, tgt)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def train_model(model, train_loader, val_loader, optimizer, criterion, device, 
                num_epochs=10, teacher_forcing_ratio=0.5, is_attention_model=False,
                model_name="model", save_path=None, pad_idx=0):
    """
    Complete training loop for a model
    
    Args:
        model: Seq2Seq model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: Optimizer
        criterion: Loss function
        device: Device (cuda/cpu)
        num_epochs: Number of training epochs
        teacher_forcing_ratio: Teacher forcing ratio
        is_attention_model: Whether model has attention
        model_name: Name for logging
        save_path: Path to save best model
        pad_idx: Padding token index
    
    Returns:
        history: Dictionary with training history
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            teacher_forcing_ratio, is_attention_model, pad_idx
        )
        
        # Evaluate
        val_loss = evaluate(
            model, val_loader, criterion, device, is_attention_model, pad_idx
        )
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch + 1)
        
        # Time
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                # Save full checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'history': history
                }
                torch.save(checkpoint, save_path)
                print(f"  → Saved best model (val_loss: {val_loss:.4f})")
        
        print()
    
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}\n")
    
    return history
