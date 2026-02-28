# Text-to-Python Code Generation Using Seq2Seq Models

## Overview
This project implements and compares three different sequence-to-sequence (Seq2Seq) models for automatically generating Python code from English natural language descriptions (docstrings):

1. **Vanilla RNN-based Seq2Seq** — Baseline model; compresses the entire input into one fixed-length vector
2. **LSTM-based Seq2Seq** — Better at remembering long sentences thanks to LSTM's memory cells
3. **LSTM with Attention** — Best model; the decoder can "look back" at any part of the input while generating each output token

---

## Assignment Goal

Given an English description (docstring) like:
> *"returns the sum of two numbers"*

The model should automatically generate the corresponding Python code:
```python
def add(a, b):
    return a + b
```

---

## Dataset
- **Source**: [CodeSearchNet Python Dataset](https://huggingface.co/datasets/Nan-Do/code-search-net-python) (HuggingFace)
- **Task**: English docstrings → Python code
- **Training samples**: up to 10,000 examples
- **Max lengths**: Docstring: 50 tokens, Code: 80 tokens

---

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
│   │   ├── train.py               # Training loop
│   │   └── evaluate.py            # Evaluation metrics
│   └── utils/
│       ├── __init__.py
│       └── helpers.py             # Plotting and utility functions
└── results/                        # Saved model checkpoints and plots
```

---

## Full Pipeline — Step by Step

```
Raw Dataset (HuggingFace)
        ↓
  clean_docstring()         ← cleans the English description
  clean_code()              ← cleans the Python code
        ↓
  build_vocab_from_data()   ← builds a word ↔ integer dictionary
        ↓
  CodeSearchNetDataset      ← converts text to tensors
  create_dataloaders()      ← batches the tensors for training
        ↓
  Model: Encoder → Decoder  ← learns to translate English → Python
        ↓
  train_model()             ← runs training epochs, saves best model
        ↓
  evaluate_model_outputs()  ← measures BLEU, exact match, token accuracy
        ↓
  Visualization helpers     ← plots losses, metrics, attention heatmaps
```

---

## Detailed Code Explanation

### `src/data_loader.py` — Data Loading & Preprocessing

#### `clean_docstring(text)`
**What it does:** Cleans the English description text before training.
- Removes extra spaces/newlines (replaces multiple whitespace with a single space)
- Converts everything to lowercase
- Removes special characters, keeping only letters, numbers, and basic punctuation (`.`, `,`, `-`, `_`)

```python
clean_docstring("  Returns the SUM of A and B!!  ")
# → "returns the sum of a and b"
```

#### `clean_code(code)`
**What it does:** Cleans Python code.
- Removes blank lines
- Strips trailing whitespace from each line
- Collapses multiple spaces into one (but **preserves indentation**)

#### `load_codesearchnet_data(num_train, num_val, num_test)`
**What it does:** Loads the dataset from HuggingFace and splits it.
1. Downloads the CodeSearchNet Python dataset
2. Loops through examples, extracts `docstring` and `code` for each
3. Calls `clean_docstring()` and `clean_code()` to clean both
4. Filters out examples that are too short (< 3 words) or too long (> 60/100 words) to keep training efficient
5. Splits the cleaned data into train / validation / test sets and returns them

#### `CodeSearchNetDataset` (class)
**What it does:** Wraps the data so PyTorch can load it in batches.
- `__len__()` — returns how many examples exist
- `__getitem__(idx)` — for a given index, converts the docstring and code to integer tensors using `vocab.encode_docstring()` and `vocab.encode_code()`, then returns `(src_tensor, tgt_tensor)`

#### `create_dataloaders(train_data, val_data, test_data, vocab, batch_size, ...)`
**What it does:** Wraps each dataset in a `DataLoader` so training can iterate over mini-batches automatically.
- `shuffle=True` for training (randomises order each epoch)
- `shuffle=False` for validation/test (consistent order for evaluation)

---

### `src/tokenizer.py` — Vocabulary & Tokenization

#### `Vocabulary` (class)
**What it does:** Manages the mapping between words and integers.

Special tokens:
| Token | Index | Meaning |
|-------|-------|---------|
| `<PAD>` | 0 | Padding (fills shorter sequences to fixed length) |
| `<SOS>` | 1 | Start-of-sequence marker |
| `<EOS>` | 2 | End-of-sequence marker |
| `<UNK>` | 3 | Unknown word (not in vocabulary) |

#### `Vocabulary.build_vocabulary(docstrings, codes)`
**What it does:** Builds the word-to-integer dictionary from training data.
1. Tokenizes every docstring and code snippet
2. Counts how often each token appears (`Counter`)
3. Adds tokens that appear **at least `min_freq` times** (default: 2) to the vocabulary
4. Assigns a unique integer index to each token

Why `min_freq=2`? Words that appear only once are likely typos or rare names. Ignoring them keeps the vocabulary small and training fast.

#### `Vocabulary._tokenize_docstring(text)`
Simple whitespace split: `"returns sum of two"` → `["returns", "sum", "of", "two"]`

#### `Vocabulary._tokenize_code(code)`
Splits on whitespace **and** inserts spaces around code symbols like `(`, `)`, `=`, `+`, etc., so each operator becomes its own token:
`"return a+b"` → `["return", "a", "+", "b"]`

#### `Vocabulary.encode_docstring(text, max_len)`
Converts a docstring to a list of integers:
1. Tokenizes the text
2. Adds `<SOS>` at the start and `<EOS>` at the end
3. Looks up each token in `word2idx` (uses `<UNK>` if not found)
4. Pads with `<PAD>` tokens until the list is exactly `max_len` long

#### `Vocabulary.encode_code(code, max_len)`
Same as `encode_docstring`, but uses the code tokenizer.

#### `Vocabulary.decode(indices, skip_special_tokens=True)`
Converts a list of integers back to a human-readable string.
Stops at `<EOS>`, skips `<PAD>` and `<SOS>` if `skip_special_tokens=True`.

#### `build_vocab_from_data(train_data, val_data, min_freq)`
**What it does:** Convenience function — extracts all docstrings and code from the data and calls `Vocabulary.build_vocabulary()`.

---

### `src/models/vanilla_rnn.py` — Model 1: Vanilla RNN

#### Architecture concept
```
Input docstring → [Encoder RNN] → hidden state (context vector)
                                          ↓
                               [Decoder RNN] → output code tokens
```
The **entire input** is squeezed into a single hidden state vector, which is then used to generate the output. This is the **bottleneck** — long inputs lose information.

#### `VanillaRNNEncoder`
- `embedding`: Converts integer token IDs to dense vectors of size `embed_size`
- `rnn`: Vanilla RNN layer. Processes the embedded tokens one by one, updating a hidden state at each step
- `forward(src)`:
  - Embeds the source tokens
  - Runs them through the RNN
  - Returns **all hidden states** (one per input token) and the **final hidden state** (the context vector)

#### `VanillaRNNDecoder`
- Same structure as the encoder but generates output tokens one at a time
- `fc_out`: A linear layer that maps the hidden state to a probability over the entire vocabulary
- `forward(tgt, hidden)`:
  - Takes the current target token and the hidden state
  - Returns predicted logits (scores for each vocabulary word) and the updated hidden state

#### `VanillaRNNSeq2Seq`
Combines encoder and decoder.
- `forward(src, tgt, teacher_forcing_ratio=0.5)`:
  1. Encodes the full source sequence → gets `hidden`
  2. Starts decoding with `<SOS>` token
  3. At each step, runs the decoder to get a prediction
  4. **Teacher forcing**: with probability `teacher_forcing_ratio`, feeds the *real* next token instead of the predicted one. This speeds up training.
  5. Returns all output logits `[batch_size, tgt_len, vocab_size]`

- `generate(src, max_len, sos_idx, eos_idx)`:
  - Used at inference time (no teacher forcing)
  - Greedily picks the highest-probability token at each step
  - Stops when `<EOS>` is generated or `max_len` is reached

---

### `src/models/lstm_seq2seq.py` — Model 2: LSTM

#### Why LSTM over Vanilla RNN?
Vanilla RNN suffers from the **vanishing gradient problem** — gradients shrink during backpropagation through many time steps, making it hard to learn long-range dependencies.

LSTM adds:
- **Cell state (`c`)**: Long-term memory
- **Forget gate**: Decides what to erase from memory
- **Input gate**: Decides what new information to store
- **Output gate**: Decides what to output from memory

#### `LSTMEncoder`
Same idea as `VanillaRNNEncoder` but uses `nn.LSTM` instead of `nn.RNN`.
- `forward(src)` returns `(outputs, (hidden, cell))` — both the hidden state AND the cell state are passed to the decoder

#### `LSTMDecoder`
- `forward(tgt, hidden, cell)`:
  - Takes the current token, hidden state, and cell state
  - Returns predictions, updated hidden, and updated cell

#### `LSTMSeq2Seq`
Same structure as `VanillaRNNSeq2Seq` but threads both `hidden` and `cell` through the decoder at each step.

---

### `src/models/lstm_attention.py` — Model 3: LSTM with Attention

#### Why Attention?
Even LSTM compresses the entire input into a fixed-size vector. With attention, the decoder can **dynamically focus on different parts of the input** for each output token — like a human re-reading different parts of a description while writing code.

#### `Attention` (class)
Implements **Bahdanau (additive) attention**.
- `forward(hidden, encoder_outputs)`:
  1. Expands the current decoder hidden state to match the length of the encoder outputs
  2. Concatenates them and passes through a small neural network (`attn` + `v`)
  3. Applies `softmax` to get **attention weights** — a probability distribution over input tokens
  4. Computes a **context vector** as the weighted sum of encoder outputs
  5. Returns `(attention_weights, context)`

The attention weights tell you *which input words the model is paying attention to* for this output token.

#### `AttentionLSTMEncoder`
Uses a **bidirectional** LSTM — processes the input sequence **left-to-right AND right-to-left** simultaneously, giving each token context from both sides.
- Output size is `hidden_size * 2` (forward + backward directions concatenated)
- `fc_hidden` and `fc_cell`: Linear layers that project the bidirectional states back down to `hidden_size` for the decoder

#### `AttentionLSTMDecoder`
- `forward(tgt, hidden, cell, encoder_outputs)`:
  1. Embeds the current token
  2. Calls `self.attention(hidden[-1], encoder_outputs)` to get attention weights and context vector
  3. Concatenates the embedded token with the context vector → feeds into LSTM
  4. Concatenates LSTM output with context vector → feeds into `fc_out` for final prediction
  5. Returns `(prediction, hidden, cell, attention_weights)`

#### `LSTMAttentionSeq2Seq`
- `forward(src, tgt, teacher_forcing_ratio)`:
  - Same as other models, but also collects `attention_weights` at each step
  - Returns `(outputs, attentions)` — attentions can be visualised as a heatmap

- `generate(src, max_len, sos_idx, eos_idx)`:
  - Returns both the generated token sequence and the attention weights for every step

---

### `src/training/train.py` — Training Logic

#### `train_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio, is_attention_model, pad_idx)`
**What it does:** Runs one full pass over the training data.
1. Sets model to training mode (`model.train()`)
2. For each batch `(src, tgt)`:
   a. Moves tensors to GPU/CPU
   b. Zeros gradients (`optimizer.zero_grad()`)
   c. Runs the forward pass
   d. Reshapes output and target, skipping the `<SOS>` token (index 0), for loss calculation
   e. Computes cross-entropy loss (ignores `<PAD>` tokens via `ignore_index`)
   f. Runs backpropagation (`loss.backward()`)
   g. Clips gradients to max norm 1.0 (prevents exploding gradients)
   h. Updates weights (`optimizer.step()`)
3. Returns the average loss over all batches

`is_attention_model=True` tells the function to unpack the extra `attention` output that the attention model returns.

#### `evaluate(model, dataloader, criterion, device, is_attention_model, pad_idx)`
**What it does:** Measures loss on the validation/test set without updating weights.
- Sets model to eval mode and wraps in `torch.no_grad()` (no gradient tracking needed)
- `teacher_forcing_ratio=0` — uses only model predictions, not ground truth tokens

#### `train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, ...)`
**What it does:** The complete training loop.
1. Runs `train_epoch()` and `evaluate()` for each epoch
2. Records `train_loss` and `val_loss` in a `history` dictionary
3. If `val_loss` improves, saves a checkpoint with:
   - Model weights (`model_state_dict`)
   - Optimiser state (`optimizer_state_dict`)
   - Epoch number and losses
4. Returns `history` (used to plot training curves)

---

### `src/training/evaluate.py` — Evaluation Metrics

#### `calculate_bleu(reference, candidate, max_n=4)`
**What it does:** Calculates the **BLEU score** — a standard metric for measuring how similar a generated sequence is to the reference.
- Counts n-gram overlaps (1-gram, 2-gram, 3-gram, 4-gram) between reference and candidate
- Applies a **brevity penalty** if the generated sequence is shorter than the reference
- Returns a score between 0 (no match) and 1 (perfect match)

#### `calculate_exact_match(reference, candidate)`
Returns `1.0` if the generated sequence is *exactly* the same as the reference, `0.0` otherwise.

#### `calculate_token_accuracy(reference, candidate)`
Calculates what fraction of tokens match at the same position (after padding both to the same length).

#### `evaluate_model_outputs(model, test_loader, vocab, device, num_samples, is_attention_model, model_name)`
**What it does:** Full evaluation pipeline.
1. Loops over the test set
2. For each example, calls `model.generate()` to produce a prediction
3. Converts integer indices back to tokens using the vocabulary
4. Computes BLEU, exact match, and token accuracy for each example
5. Stores the first 10 examples (input, reference, prediction, scores, attention if applicable)
6. Prints and returns aggregated metrics

---

### `src/utils/helpers.py` — Utility & Visualization Functions

#### `plot_training_history(histories, model_names, save_path)`
Plots training and validation loss curves for all three models side-by-side. Useful for comparing convergence speed.

#### `plot_metrics_comparison(metrics_dict, save_path)`
Creates three bar charts comparing BLEU score, exact match, and token accuracy across all models.

#### `print_examples(examples, num_examples)`
Prints sample predictions in a readable format:
```
Input:      returns the absolute value of x
Reference:  def abs_val ( x ) : return abs ( x )
Prediction: def abs_val ( x ) : return x
BLEU: 0.7500 | Exact Match: 0
```

#### `count_parameters(model)`
Returns the total number of trainable parameters in a model. Useful for understanding model complexity.

#### `plot_attention_heatmap(attention_weights, source_tokens, target_tokens, save_path)`
Creates a heatmap where:
- **X-axis** = source (input docstring) tokens
- **Y-axis** = target (generated code) tokens
- **Cell colour** = how much attention the model paid to that input token when generating that output token

#### `plot_multiple_attention_heatmaps(examples_with_attention, vocab, save_path, num_examples)`
Shows multiple attention heatmaps side-by-side for comparison.

#### `analyze_errors_by_type(examples, vocab)`
Categorises prediction errors by type:
- **Keyword errors**: wrong Python keywords (`def`, `return`, `if`, etc.)
- **Operator errors**: wrong operators (`+`, `=`, `==`, etc.)
- **Bracket mismatch**: unmatched parentheses/brackets
- **Indentation errors**: missing colons
- **Variable name errors**: wrong variable names

#### `plot_error_analysis(error_stats, save_path)`
Bar chart of the error type counts.

#### `plot_performance_vs_length(examples, save_path)`
Scatter plot showing how BLEU score changes as input docstring length increases. Longer inputs are generally harder.

#### `save_model(model, optimizer, epoch, loss, save_path)`
Saves a model checkpoint (weights + optimiser state + epoch + loss) to disk.

#### `load_model(model, optimizer, load_path, device)`
Loads a previously saved checkpoint back into a model and optimiser.

---

## How the Three Models Compare

| Feature | Vanilla RNN | LSTM | LSTM + Attention |
|---------|-------------|------|-----------------|
| Memory mechanism | Simple hidden state | Hidden + cell state | Hidden + cell + dynamic attention |
| Handles long sequences | ✗ Poor | ✓ Better | ✓✓ Best |
| Fixed-length bottleneck | Yes | Yes | No — attention removes it |
| Interpretability | None | None | Attention heatmaps |
| Expected performance | Lowest BLEU | Medium BLEU | Highest BLEU |

---

## How to Run in Google Colab

1. **Upload to Google Drive or clone from GitHub**
2. **Open `text_to_python_colab.ipynb` in Google Colab**
3. **Run all cells** — the notebook automatically:
   - Installs dependencies
   - Loads and preprocesses data
   - Builds the vocabulary
   - Trains all three models
   - Evaluates and compares results
   - Visualises performance metrics and attention maps

---

## Key Concepts Glossary

| Term | Meaning |
|------|---------|
| **Seq2Seq** | Encoder–Decoder architecture that maps one sequence to another |
| **Encoder** | Reads the input and compresses it into a hidden state |
| **Decoder** | Uses the hidden state to generate the output one token at a time |
| **Teacher forcing** | During training, feed the real target token as the next input instead of the model's prediction — speeds up learning |
| **BLEU score** | Measures n-gram overlap between generated and reference text (0 = worst, 1 = best) |
| **Attention** | Mechanism allowing the decoder to focus on different input tokens at each output step |
| **Bidirectional LSTM** | Processes input both left-to-right and right-to-left for richer representations |
| **Gradient clipping** | Caps gradient magnitudes to prevent exploding gradients during training |
| **Vocabulary** | Dictionary mapping every unique token to an integer index |
| `<PAD>` | Padding token to make all sequences the same length in a batch |
| `<SOS>` | Start-of-sequence token — the first input to the decoder |
| `<EOS>` | End-of-sequence token — signals the decoder to stop generating |
| `<UNK>` | Unknown token — used when a word is not in the vocabulary |

---

## Author
Assignment: Text-to-Python Code Generation Using Seq2Seq Models (RNNs)
