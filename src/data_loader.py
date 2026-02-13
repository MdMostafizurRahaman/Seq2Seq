"""
Data loading and preprocessing for CodeSearchNet Python dataset
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import re
from typing import List, Tuple, Dict
import numpy as np


class CodeSearchNetDataset(Dataset):
    """Custom Dataset for CodeSearchNet Python data"""
    
    def __init__(self, data, vocab, max_docstring_len=50, max_code_len=80):
        """
        Args:
            data: List of dictionaries with 'docstring' and 'code' keys
            vocab: Vocabulary object
            max_docstring_len: Maximum length for docstrings
            max_code_len: Maximum length for code
        """
        self.data = data
        self.vocab = vocab
        self.max_docstring_len = max_docstring_len
        self.max_code_len = max_code_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            src: Source sequence (docstring) as tensor of indices
            tgt: Target sequence (code) as tensor of indices
            src_len: Actual length of source sequence
            tgt_len: Actual length of target sequence
        """
        docstring = self.data[idx]['docstring']
        code = self.data[idx]['code']
        
        # Tokenize and convert to indices
        src_tokens = self.vocab.encode_docstring(docstring, self.max_docstring_len)
        tgt_tokens = self.vocab.encode_code(code, self.max_code_len)
        
        src = torch.tensor(src_tokens, dtype=torch.long)
        tgt = torch.tensor(tgt_tokens, dtype=torch.long)
        
        return src, tgt


def clean_docstring(text: str) -> str:
    """Clean and normalize docstring text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Convert to lowercase
    text = text.lower().strip()
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\-\_]', '', text)
    
    return text


def clean_code(code: str) -> str:
    """Clean and normalize Python code"""
    if not code:
        return ""
    
    # Remove extra blank lines
    lines = [line.rstrip() for line in code.split('\n') if line.strip()]
    code = '\n'.join(lines)
    
    # Remove excessive whitespace but preserve indentation
    code = re.sub(r' +', ' ', code)
    
    return code.strip()


def load_codesearchnet_data(num_train=10000, num_val=1000, num_test=1000):
    """
    Load and preprocess CodeSearchNet Python dataset
    
    Args:
        num_train: Number of training examples
        num_val: Number of validation examples
        num_test: Number of test examples
    
    Returns:
        train_data, val_data, test_data: Lists of dictionaries
    """
    print("Loading CodeSearchNet Python dataset...")
    
    # Load dataset from HuggingFace
    dataset = load_dataset("Nan-Do/code-search-net-python", split="train", trust_remote_code=True)
    
    # Process and filter data
    processed_data = []
    
    for i, example in enumerate(dataset):
        if len(processed_data) >= (num_train + num_val + num_test):
            break
            
        # Extract docstring and code
        docstring = example.get('docstring', '') or example.get('func_documentation_string', '')
        code = example.get('code', '') or example.get('func_code_string', '')
        
        if not docstring or not code:
            continue
        
        # Clean and filter
        docstring = clean_docstring(docstring)
        code = clean_code(code)
        
        # Filter by length (rough token estimate)
        if len(docstring.split()) > 60 or len(docstring.split()) < 3:
            continue
        if len(code.split()) > 100 or len(code.split()) < 3:
            continue
        
        processed_data.append({
            'docstring': docstring,
            'code': code
        })
    
    # Split into train/val/test
    train_data = processed_data[:num_train]
    val_data = processed_data[num_train:num_train + num_val]
    test_data = processed_data[num_train + num_val:num_train + num_val + num_test]
    
    print(f"Loaded {len(train_data)} training examples")
    print(f"Loaded {len(val_data)} validation examples")
    print(f"Loaded {len(test_data)} test examples")
    
    return train_data, val_data, test_data


def create_dataloaders(train_data, val_data, test_data, vocab, batch_size=32, 
                       max_docstring_len=50, max_code_len=80):
    """
    Create PyTorch DataLoaders for training, validation, and testing
    
    Args:
        train_data, val_data, test_data: Lists of data dictionaries
        vocab: Vocabulary object
        batch_size: Batch size for training
        max_docstring_len: Maximum docstring length
        max_code_len: Maximum code length
    
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
    """
    train_dataset = CodeSearchNetDataset(train_data, vocab, max_docstring_len, max_code_len)
    val_dataset = CodeSearchNetDataset(val_data, vocab, max_docstring_len, max_code_len)
    test_dataset = CodeSearchNetDataset(test_data, vocab, max_docstring_len, max_code_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
