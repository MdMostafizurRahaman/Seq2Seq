"""
Vanilla RNN-based Seq2Seq Model
Baseline model with fixed-length context vector and no attention mechanism
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaRNNEncoder(nn.Module):
    """
    Vanilla RNN Encoder
    Encodes input sequence into a fixed-length context vector
    """
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_size: Embedding dimension
            hidden_size: Hidden state dimension
            num_layers: Number of RNN layers
            dropout: Dropout probability
        """
        super(VanillaRNNEncoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # Vanilla RNN
        self.rnn = nn.RNN(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        """
        Args:
            src: Source sequences [batch_size, seq_len]
        
        Returns:
            outputs: All hidden states [batch_size, seq_len, hidden_size]
            hidden: Final hidden state [num_layers, batch_size, hidden_size]
        """
        # Embed input
        embedded = self.dropout(self.embedding(src))  # [batch_size, seq_len, embed_size]
        
        # Pass through RNN
        outputs, hidden = self.rnn(embedded)
        # outputs: [batch_size, seq_len, hidden_size]
        # hidden: [num_layers, batch_size, hidden_size]
        
        return outputs, hidden


class VanillaRNNDecoder(nn.Module):
    """
    Vanilla RNN Decoder
    Decodes from fixed-length context vector to output sequence
    """
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_size: Embedding dimension
            hidden_size: Hidden state dimension
            num_layers: Number of RNN layers
            dropout: Dropout probability
        """
        super(VanillaRNNDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # Vanilla RNN
        self.rnn = nn.RNN(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, hidden):
        """
        Args:
            tgt: Target sequence [batch_size, seq_len]
            hidden: Hidden state from encoder [num_layers, batch_size, hidden_size]
        
        Returns:
            output: Output logits [batch_size, seq_len, vocab_size]
            hidden: Updated hidden state [num_layers, batch_size, hidden_size]
        """
        # Embed input
        embedded = self.dropout(self.embedding(tgt))  # [batch_size, seq_len, embed_size]
        
        # Pass through RNN
        outputs, hidden = self.rnn(embedded, hidden)
        # outputs: [batch_size, seq_len, hidden_size]
        
        # Project to vocabulary
        predictions = self.fc_out(outputs)  # [batch_size, seq_len, vocab_size]
        
        return predictions, hidden


class VanillaRNNSeq2Seq(nn.Module):
    """
    Vanilla RNN Seq2Seq Model
    Complete encoder-decoder architecture with fixed-length context vector
    
    Limitations:
    - Fixed-length context vector is a bottleneck
    - Struggles with long sequences
    - Information loss for longer inputs
    """
    
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=1, dropout=0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_size: Embedding dimension
            hidden_size: Hidden state dimension
            num_layers: Number of RNN layers
            dropout: Dropout probability
        """
        super(VanillaRNNSeq2Seq, self).__init__()
        
        self.encoder = VanillaRNNEncoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.decoder = VanillaRNNDecoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        Args:
            src: Source sequences [batch_size, src_len]
            tgt: Target sequences [batch_size, tgt_len]
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            outputs: Output logits [batch_size, tgt_len, vocab_size]
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.vocab_size
        
        # Encode source sequence
        encoder_outputs, hidden = self.encoder(src)
        
        # Prepare decoder input (start with <SOS> token)
        decoder_input = tgt[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        # Store outputs
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(src.device)
        
        # Decode step-by-step
        for t in range(tgt_len):
            # Decode one step
            output, hidden = self.decoder(decoder_input, hidden)
            
            # Store output
            outputs[:, t:t+1, :] = output
            
            # Teacher forcing: use actual target as next input
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            if teacher_force and t < tgt_len - 1:
                decoder_input = tgt[:, t+1].unsqueeze(1)
            else:
                # Use predicted token as next input
                decoder_input = output.argmax(dim=-1)
        
        return outputs
    
    def generate(self, src, max_len=80, sos_idx=1, eos_idx=2):
        """
        Generate output sequence using greedy decoding
        
        Args:
            src: Source sequence [1, src_len]
            max_len: Maximum output length
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
        
        Returns:
            generated: Generated sequence indices
        """
        self.eval()
        with torch.no_grad():
            # Encode
            encoder_outputs, hidden = self.encoder(src)
            
            # Start with SOS token
            decoder_input = torch.tensor([[sos_idx]]).to(src.device)
            
            generated = []
            
            for _ in range(max_len):
                # Decode one step
                output, hidden = self.decoder(decoder_input, hidden)
                
                # Get predicted token
                predicted = output.argmax(dim=-1).item()
                generated.append(predicted)
                
                # Stop if EOS token is generated
                if predicted == eos_idx:
                    break
                
                # Use predicted token as next input
                decoder_input = torch.tensor([[predicted]]).to(src.device)
            
            return generated
