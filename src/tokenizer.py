"""
Vocabulary and tokenization for docstrings and Python code
"""
import re
from typing import List, Dict
from collections import Counter


class Vocabulary:
    """Vocabulary class for managing word-to-index mappings"""
    
    PAD_TOKEN = '<PAD>'
    SOS_TOKEN = '<SOS>'
    EOS_TOKEN = '<EOS>'
    UNK_TOKEN = '<UNK>'
    
    def __init__(self, min_freq=2):
        """
        Args:
            min_freq: Minimum frequency for a token to be included in vocabulary
        """
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        
        # Special tokens
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        
        # Initialize with special tokens
        self._add_special_tokens()
        
    def _add_special_tokens(self):
        """Add special tokens to vocabulary"""
        self.word2idx[self.PAD_TOKEN] = self.pad_idx
        self.word2idx[self.SOS_TOKEN] = self.sos_idx
        self.word2idx[self.EOS_TOKEN] = self.eos_idx
        self.word2idx[self.UNK_TOKEN] = self.unk_idx
        
        self.idx2word[self.pad_idx] = self.PAD_TOKEN
        self.idx2word[self.sos_idx] = self.SOS_TOKEN
        self.idx2word[self.eos_idx] = self.EOS_TOKEN
        self.idx2word[self.unk_idx] = self.UNK_TOKEN
        
    def build_vocabulary(self, docstrings: List[str], codes: List[str]):
        """
        Build vocabulary from docstrings and code
        
        Args:
            docstrings: List of docstring texts
            codes: List of code texts
        """
        # Count tokens in docstrings
        for doc in docstrings:
            tokens = self._tokenize_docstring(doc)
            self.word_counts.update(tokens)
        
        # Count tokens in code
        for code in codes:
            tokens = self._tokenize_code(code)
            self.word_counts.update(tokens)
        
        # Add words that meet minimum frequency
        idx = len(self.word2idx)
        for word, count in self.word_counts.items():
            if count >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        print(f"Vocabulary size: {len(self.word2idx)}")
        
    def _tokenize_docstring(self, text: str) -> List[str]:
        """Tokenize docstring text"""
        # Simple word-level tokenization
        tokens = text.lower().split()
        return tokens
    
    def _tokenize_code(self, code: str) -> List[str]:
        """Tokenize Python code"""
        # Split on whitespace and common code delimiters
        code = re.sub(r'([(){}[\],.:=+\-*/%<>!&|])', r' \1 ', code)
        tokens = code.split()
        return tokens
    
    def encode_docstring(self, text: str, max_len: int) -> List[int]:
        """
        Convert docstring to list of indices
        
        Args:
            text: Docstring text
            max_len: Maximum sequence length
        
        Returns:
            List of token indices with <SOS>, <EOS>, and padding
        """
        tokens = self._tokenize_docstring(text)
        indices = [self.sos_idx]
        
        for token in tokens[:max_len - 2]:  # Reserve space for SOS and EOS
            idx = self.word2idx.get(token, self.unk_idx)
            indices.append(idx)
        
        indices.append(self.eos_idx)
        
        # Pad to max_len
        while len(indices) < max_len:
            indices.append(self.pad_idx)
        
        return indices[:max_len]
    
    def encode_code(self, code: str, max_len: int) -> List[int]:
        """
        Convert code to list of indices
        
        Args:
            code: Python code
            max_len: Maximum sequence length
        
        Returns:
            List of token indices with <SOS>, <EOS>, and padding
        """
        tokens = self._tokenize_code(code)
        indices = [self.sos_idx]
        
        for token in tokens[:max_len - 2]:  # Reserve space for SOS and EOS
            idx = self.word2idx.get(token, self.unk_idx)
            indices.append(idx)
        
        indices.append(self.eos_idx)
        
        # Pad to max_len
        while len(indices) < max_len:
            indices.append(self.pad_idx)
        
        return indices[:max_len]
    
    def decode(self, indices: List[int], skip_special_tokens=True) -> str:
        """
        Convert list of indices back to text
        
        Args:
            indices: List of token indices
            skip_special_tokens: Whether to skip special tokens in output
        
        Returns:
            Decoded text string
        """
        tokens = []
        for idx in indices:
            if idx == self.eos_idx and skip_special_tokens:
                break
            
            word = self.idx2word.get(idx, self.UNK_TOKEN)
            
            if skip_special_tokens and word in [self.PAD_TOKEN, self.SOS_TOKEN]:
                continue
            
            tokens.append(word)
        
        return ' '.join(tokens)
    
    def __len__(self):
        return len(self.word2idx)


def build_vocab_from_data(train_data, val_data, min_freq=2):
    """
    Build vocabulary from training and validation data
    
    Args:
        train_data: Training data list
        val_data: Validation data list
        min_freq: Minimum frequency threshold
    
    Returns:
        Vocabulary object
    """
    print("Building vocabulary...")
    
    vocab = Vocabulary(min_freq=min_freq)
    
    # Extract docstrings and code
    docstrings = [item['docstring'] for item in train_data + val_data]
    codes = [item['code'] for item in train_data + val_data]
    
    # Build vocabulary
    vocab.build_vocabulary(docstrings, codes)
    
    return vocab
