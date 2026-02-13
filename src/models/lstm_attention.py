"""
LSTM with Attention Mechanism
Advanced model that overcomes the fixed-length context vector bottleneck
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism
    Computes attention weights over encoder hidden states
    Now handles bidirectional encoder outputs
    """
    
    def __init__(self, hidden_size, encoder_hidden_size):
        """
        Args:
            hidden_size: Decoder hidden state dimension
            encoder_hidden_size: Encoder output dimension (hidden_size * 2 for bidirectional)
        """
        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        
        self.attn = nn.Linear(hidden_size + encoder_hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        """
        Compute attention weights and context vector
        
        Args:
            hidden: Current decoder hidden state [batch_size, hidden_size]
            encoder_outputs: All encoder outputs [batch_size, src_len, encoder_hidden_size]
        
        Returns:
            attention_weights: Attention distribution [batch_size, src_len]
            context: Weighted sum of encoder outputs [batch_size, encoder_hidden_size]
        """
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  
        
        energy = torch.cat((hidden, encoder_outputs), dim=2)  
        
        energy = torch.tanh(self.attn(energy))  
        attention = self.v(energy).squeeze(2) 
        
        attention_weights = F.softmax(attention, dim=1)  
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  
        context = context.squeeze(1)          
        return attention_weights, context


class AttentionLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM Encoder for Attention-based Seq2Seq
    Uses bidirectional LSTM to capture context from both directions
    """
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_size: Embedding dimension
            hidden_size: Hidden state dimension (per direction)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(AttentionLSTMEncoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidirectional for better context
        )
        
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        """
        Args:
            src: Source sequences [batch_size, seq_len]
        
        Returns:
            outputs: All hidden states [batch_size, seq_len, hidden_size * 2]
            hidden: Tuple of (h_n, c_n) - [num_layers, batch_size, hidden_size]
        """
        embedded = self.dropout(self.embedding(src))
        
        # Bidirectional LSTM outputs
        outputs, (hidden, cell) = self.lstm(embedded)
        
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
        cell = cell.view(self.num_layers, 2, -1, self.hidden_size)
        
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
        
        # Project to hidden_size: [num_layers, batch_size, hidden_size]
        hidden = torch.tanh(self.fc_hidden(hidden))
        cell = torch.tanh(self.fc_cell(cell))
        
        return outputs, (hidden, cell)


class AttentionLSTMDecoder(nn.Module):
    """
    LSTM Decoder with Attention Mechanism
    Uses attention to focus on relevant parts of input sequence
    Now handles bidirectional encoder outputs
    """
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_size: Embedding dimension
            hidden_size: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(AttentionLSTMDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.encoder_hidden_size = hidden_size * 2
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        self.attention = Attention(hidden_size, self.encoder_hidden_size)
        
        self.lstm = nn.LSTM(
            embed_size + self.encoder_hidden_size,  # Concatenate embedding with bidirectional context
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection (hidden + context)
        self.fc_out = nn.Linear(hidden_size + self.encoder_hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, hidden, cell, encoder_outputs):
        """
        Args:
            tgt: Target token [batch_size, 1]
            hidden: Hidden state [num_layers, batch_size, hidden_size]
            cell: Cell state [num_layers, batch_size, hidden_size]
            encoder_outputs: All encoder outputs [batch_size, src_len, hidden_size * 2]
        
        Returns:
            prediction: Output logits [batch_size, vocab_size]
            hidden: Updated hidden state
            cell: Updated cell state
            attention_weights: Attention distribution [batch_size, src_len]
        """
        # Embed input
        embedded = self.dropout(self.embedding(tgt))  # [batch_size, 1, embed_size]
        
        attention_weights, context = self.attention(hidden[-1], encoder_outputs)
        
        context = context.unsqueeze(1) 
        rnn_input = torch.cat([embedded, context], dim=2)  
        
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        # output: [batch_size, 1, hidden_size]
        
        output = output.squeeze(1)  # [batch_size, hidden_size]
        context = context.squeeze(1)  # [batch_size, encoder_hidden_size]
        
        # Concatenate LSTM output and context for final prediction
        prediction_input = torch.cat([output, context], dim=1)  # [batch_size, hidden_size + encoder_hidden_size]
        prediction = self.fc_out(prediction_input)  # [batch_size, vocab_size]
        
        return prediction, hidden, cell, attention_weights


class LSTMAttentionSeq2Seq(nn.Module):
    """
    LSTM Seq2Seq with Attention Mechanism
    
    Key Advantages:
    - Attention mechanism allows decoder to focus on relevant input tokens
    - Overcomes fixed-length context vector bottleneck
    - Better handling of long sequences
    - Interpretable attention weights show which input tokens are important
    """
    
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=2, dropout=0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_size: Embedding dimension
            hidden_size: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(LSTMAttentionSeq2Seq, self).__init__()
        
        self.encoder = AttentionLSTMEncoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.decoder = AttentionLSTMDecoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        Args:
            src: Source sequences [batch_size, src_len]
            tgt: Target sequences [batch_size, tgt_len]
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            outputs: Output logits [batch_size, tgt_len, vocab_size]
            attention_weights: All attention weights [batch_size, tgt_len, src_len]
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.vocab_size
        
        # Encode source sequence
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        # Prepare decoder input (start with <SOS> token)
        decoder_input = tgt[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        # Store outputs and attention weights
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(src.device)
        attentions = torch.zeros(batch_size, tgt_len, encoder_outputs.size(1)).to(src.device)
        
        # Decode step-by-step
        for t in range(tgt_len):
            # Decode one step with attention
            output, hidden, cell, attention_weights = self.decoder(
                decoder_input, hidden, cell, encoder_outputs
            )
            
            # Store output and attention
            outputs[:, t, :] = output
            attentions[:, t, :] = attention_weights
            
            # Teacher forcing: use actual target as next input
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            if teacher_force and t < tgt_len - 1:
                decoder_input = tgt[:, t+1].unsqueeze(1)
            else:
                # Use predicted token as next input
                decoder_input = output.argmax(dim=-1).unsqueeze(1)
        
        return outputs, attentions
    
    def generate(self, src, max_len=80, sos_idx=1, eos_idx=2):
        """
        Generate output sequence using greedy decoding with attention
        
        Args:
            src: Source sequence [1, src_len]
            max_len: Maximum output length
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
        
        Returns:
            generated: Generated sequence indices
            attention_weights: Attention weights for each step [tgt_len, src_len]
        """
        self.eval()
        with torch.no_grad():
            # Encode
            encoder_outputs, (hidden, cell) = self.encoder(src)
            
            # Start with SOS token
            decoder_input = torch.tensor([[sos_idx]]).to(src.device)
            
            generated = []
            all_attention_weights = []
            
            for _ in range(max_len):
                # Decode one step
                output, hidden, cell, attention_weights = self.decoder(
                    decoder_input, hidden, cell, encoder_outputs
                )
                
                # Get predicted token
                predicted = output.argmax(dim=-1).item()
                generated.append(predicted)
                all_attention_weights.append(attention_weights.squeeze(0).cpu().numpy())
                
                # Stop if EOS token is generated
                if predicted == eos_idx:
                    break
                
                # Use predicted token as next input
                decoder_input = torch.tensor([[predicted]]).to(src.device)
            
            # Stack attention weights: [tgt_len, src_len]
            if all_attention_weights:
                all_attention_weights = np.vstack(all_attention_weights)
            
            return generated, all_attention_weights
