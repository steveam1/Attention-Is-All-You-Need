# Attention Is All You Need

**DS 5690 – Paper Project (Fall 2025)**

**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain & Google Research, 2017)

---

## Overview

### The Problem

Before 2017, NLP models relied on RNNs or CNNs both with serious limitations. RNNs processed sequences one token at a time, making them painfully slow to train and prone to forgetting long-range dependencies (vanishing gradients). CNNs were faster but couldn't effectively capture relationships between distant words. The core issue? Neither architecture could parallelize well which bottlenecked training on large datasets.

### The Approach

Vaswani proposed something radical which was to ditch recurrence and convolution entirely. The **Transformer** uses pure attention mechanisms to process all tokens simultaneously. Self-attention computes relationships between every pair of words in a sentence regardless of distance. Multiple attention heads run in parallel, each learning different linguistic patterns. Stack this 6 layers deep in both encoder and decoder, add positional encodings (since attention doesn't inherently understand order), and you've got a model that's both more powerful and more parallelizable.

### Results

**28.4 BLEU** on WMT 2014 English-German (beating previous SOTA by 2+ BLEU)  
**41.8 BLEU** on WMT 2014 English-French (new single-model record)  
**12 hours** to train the base model on 8 GPUs (vs. days or weeks for previous models)

The speed improvement was the real game-changer. Faster training → bigger models → better performance → the foundation for GPT, BERT, and everything after.

---

## Model Architecture

### Core Components

**Multi-Head Self-Attention:** Computes attention 8 times in parallel with different learned projections. Each head focuses on different aspects of word relationships (syntax, semantics, etc.). Complexity is O(n²) in sequence length but O(1) in sequential operations.

**Position-wise Feed-Forward:** Two-layer MLP applied identically at each position. First layer expands from d_model=512 to d_ff=2048, applies ReLU, then projects back down.

**Positional Encoding:** Sine/cosine functions of varying frequencies added to input embeddings. Allows the model to use word position information since attention is permutation-invariant.

**Residual + LayerNorm:** Every sub-layer wrapped with `LayerNorm(x + Sublayer(x))` for stable training and gradient flow.

### Why This Beats RNNs/CNNs

| Model | Sequential Ops | Max Path Length | Parallelization |
|-------|---------------|-----------------|-----------------|
| RNN | O(n) | O(n) | None within sequence |
| CNN | O(1) | O(log_k(n)) | Full |
| Transformer | O(1) | O(1) | Full |

Transformers connect any two positions in constant time, enabling better long-range dependency learning.

### Detailed Pseudocode
```python
def Transformer(src, tgt):
    """
    Full seq2seq transformer
    Args: src (source tokens), tgt (target tokens)
    Returns: output probabilities
    """
    # Embed and add positional encoding
    src_embed = Embedding(src) + PositionalEncoding(len(src))
    tgt_embed = Embedding(tgt) + PositionalEncoding(len(tgt))
    
    # Encode source
    memory = Encoder(src_embed, num_layers=6)
    
    # Decode to target
    output = Decoder(tgt_embed, memory, num_layers=6)
    
    # Project to vocab
    return Softmax(Linear(output))


def Encoder(x, num_layers=6):
    """
    Stacked encoder with self-attention
    Args: x (seq_len, d_model=512)
    Returns: encoded representation
    """
    for _ in range(num_layers):
        # Self-attention
        attn = MultiHeadAttention(query=x, key=x, value=x, heads=8)
        x = LayerNorm(x + Dropout(attn))
        
        # Feed-forward
        ff = FeedForward(x, d_ff=2048)
        x = LayerNorm(x + Dropout(ff))
    
    return x


def Decoder(x, memory, num_layers=6):
    """
    Stacked decoder with masked self-attention and cross-attention
    Args: x (target embeddings), memory (encoder output)
    Returns: decoded representation
    """
    for _ in range(num_layers):
        # Masked self-attention (can't look at future tokens)
        self_attn = MultiHeadAttention(
            query=x, key=x, value=x, heads=8, mask=causal_mask
        )
        x = LayerNorm(x + Dropout(self_attn))
        
        # Cross-attention to encoder output
        cross_attn = MultiHeadAttention(
            query=x, key=memory, value=memory, heads=8
        )
        x = LayerNorm(x + Dropout(cross_attn))
        
        # Feed-forward
        ff = FeedForward(x, d_ff=2048)
        x = LayerNorm(x + Dropout(ff))
    
    return x


def MultiHeadAttention(query, key, value, heads=8, mask=None):
    """
    Parallel attention with multiple heads
    Args: query, key, value (seq_len, d_model)
    Returns: multi-head attention output
    """
    d_k = d_model // heads  # 512 / 8 = 64 per head
    
    # Linear projections split across heads
    Q = Linear(query).split_heads(heads, d_k)    # (heads, seq_len, d_k)
    K = Linear(key).split_heads(heads, d_k)
    V = Linear(value).split_heads(heads, d_k)
    
    # Scaled dot-product attention per head
    scores = (Q @ K.T) / sqrt(d_k)
    if mask:
        scores.masked_fill(mask, -inf)
    
    attn_weights = Softmax(scores)
    head_outputs = attn_weights @ V
    
    # Concatenate heads and final projection
    concat = head_outputs.concat_heads()
    return Linear(concat)


def FeedForward(x, d_ff=2048):
    """
    Position-wise FFN: two linear layers with ReLU
    """
    return Linear(ReLU(Linear(x, out=d_ff)), out=d_model)


def PositionalEncoding(seq_len, d_model=512):
    """
    Sinusoidal position encoding
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pos_enc = zeros(seq_len, d_model)
    position = arange(seq_len).unsqueeze(1)
    div_term = exp(arange(0, d_model, 2) * -(log(10000.0) / d_model))
    
    pos_enc[:, 0::2] = sin(position * div_term)
    pos_enc[:, 1::2] = cos(position * div_term)
    
    return pos_enc
```

Key innovation: Everything happens in parallel. No sequential dependencies means you can throw massive compute at training.

---

## Discussion Questions

### Question 1: Why Remove Recurrence and Convolution?

**Consider:**
- RNNs require n sequential operations for a sequence of length n. What does this mean for parallelization?
- How does path length between distant positions affect learning?
- What's the tradeoff between O(n²) complexity and O(1) sequential operations?

**Answer:** RNNs can't parallelize within a training example because each step depends on the previous one. For long sequences, this is brutally slow. Self-attention trades spatial complexity (O(n²)) for temporal parallelism (O(1) sequential steps), which works because n is usually smaller than d_model (512), and GPUs handle matrix operations efficiently. The O(1) path length also makes learning long-range dependencies way easier.

### Question 2: What Does Positional Encoding Solve?

**Consider:**
- Self-attention computes weighted sums based on content similarity. Does this use position information?
- What happens if we permute the input sequence?
- Why use sin/cos instead of learned embeddings?

**Answer:** Self-attention is permutation-invariant—swap word order and you get the same output. "Dog bites man" = "Man bites dog" without positional info. Sinusoidal encoding injects position information while allowing the model to generalize to longer sequences than seen during training (learned embeddings don't extrapolate as well). The different frequencies let the model learn to attend by relative position.

---

## Critical Analysis

### Major Limitations

**1. Quadratic Memory/Compute with Sequence Length**

O(n²) attention is brutal for long sequences. The paper mentions this but doesn't explore solutions. For a 1000-token sequence, that's 1M attention weights per head per layer. This became a huge bottleneck for documents, high-res images, etc. Spawned a whole subfield: Longformer (sparse attention), Linformer (low-rank approximation), BigBird (random + local + global attention).

**2. Interpretability Hand-waving**

The paper shows cool attention visualizations but doesn't rigorously analyze what the model learns. Recent work shows attention weights aren't always interpretable as "importance." Some heads learn syntactic patterns, others seem random. The feed-forward layers actually do most of the heavy lifting for many tasks.

**3. No Inductive Bias = Data Hungry**

Removing architectural assumptions means the model must learn everything from data. This requires massive datasets. The paper doesn't explore low-data regimes or whether some structural bias might help. Compare to CNNs, which have translation equivariance baked in and work well with less data.

**4. Positional Encoding Could Be Better**

Additive sinusoidal encoding is elegant but not optimal. Learned embeddings can work better for specific sequence lengths. Relative position representations (Transformer-XL) or rotary embeddings (RoFormer) outperform fixed sinusoidal encoding. The paper doesn't explore these alternatives.

### What Subsequent Work Corrected

**"Attention is all you need"** is overstated. Research shows:
- Feed-forward layers crucial (not just attention)
- Layer norm placement matters more than initially thought
- Many attention heads are redundant (can prune 40%+ without performance loss)
- Initialization and learning rate schedules are critical

### Should Have Explored

- **Pretraining paradigms:** The paper trains from scratch. BERT/GPT showed pretraining on unlabeled data is huge.
- **Scaling laws:** How does performance scale with model/data/compute? GPT-3 systematically studied this.
- **Other modalities:** Took 3 years for Vision Transformers. Could have been explored sooner.

---

## Impact

This paper fundamentally changed AI and is the foundation of modern deep learning.

### Paradigm Shifts

**1. Task-Agnostic Architectures**

Pre-2017: Custom architecture for each task (different models for translation, classification, QA, etc.)  
Post-Transformer: One architecture for everything. Just change the task head.

**2. Scale is All You Need**

Transformers parallelize so well that we could suddenly train 100B+ parameter models. This unlocked emergent capabilities:
- **GPT-2** (2019): 1.5B params, coherent long-form text
- **GPT-3** (2020): 175B params, few-shot learning without fine-tuning
- **GPT-4** (2023): Multimodal, reasoning, near-human performance on many tasks

**3. Attention Everywhere**

Once proven for NLP, Transformers spread to every domain:
- **Vision:** ViT (2020) matches/beats CNNs for image classification
- **Multimodal:** CLIP, DALL-E, GPT-4V combine vision + language
- **Biology:** AlphaFold 2 uses Transformers for protein structure prediction
- **RL:** Decision Transformer treats RL as sequence prediction
- **Audio:** Whisper (speech recognition), Jukebox (music generation)

### Built On / Influenced

**Foundations from prior work:**
- Attention mechanism (Bahdanau et al., 2014)
- Residual connections (ResNet, 2015)
- Layer normalization (2016)

**Directly enabled:**
- BERT (2018): Bidirectional pretraining, crushed 11 NLP benchmarks
- GPT series (2018-2023): Autoregressive language modeling at scale
- T5 (2019): Text-to-text unified framework
- Vision Transformers (2020): Pure attention for images
- ChatGPT/Claude (2022+): Conversational AI

### Real-World Products

Transformers power things you use daily:
- ChatGPT, Claude, Gemini (LLM chatbots)
- Google Translate (switched to Transformers in 2016-2017)
- GitHub Copilot (code completion)
- DALL-E, Midjourney, Stable Diffusion (image generation)
- Grammarly (writing assistance)
- YouTube/Netflix recommendations
- Google Search (BERT for query understanding)

### Why It Matters

100,000+ citations and counting. The title became a meme, but it's actually true. Transformers are the default architecture for basically everything now. They showed that:
1. Architecture matters more than task-specific engineering
2. Parallelization enables scale, scale enables capabilities
3. Attention is a universal computation primitive

This paper didn't just improve BLEU scores—it redefined what's possible with AI.

---

## Code Demo

### Translation with T5
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Translate
text = "translate English to German: Transformers changed everything."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Output: "Transformatoren haben alles verändert."
```

### Visualizing Attention
```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)

text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Get attention weights from layer 6, head 3
attention = outputs.attentions[5][0, 2]  # (seq_len, seq_len)

tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print(f"Tokens: {tokens}")
print(f"Attention from 'cat' to other words:")
cat_idx = tokens.index('cat')
for token, weight in zip(tokens, attention[cat_idx]):
    print(f"  {token}: {weight:.3f}")
```

### Fine-Tuning for Classification
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load data
dataset = load_dataset("imdb", split="train[:1000]")

# Model setup
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Tokenize
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize, batched=True)

# Train
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
trainer.train()
```

### Building from Scratch
```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Project and split heads
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x

# Test it
block = TransformerBlock()
x = torch.randn(2, 10, 512)  # (batch, seq_len, d_model)
output = block(x)
print(f"Input: {x.shape}, Output: {output.shape}")
```

---

## Citation
```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is All You Need. *Advances in Neural Information Processing Systems*, 30. https://arxiv.org/abs/1706.03762

---

## Resource Links

1. **[Original Paper (arXiv)](https://arxiv.org/abs/1706.03762)** - The paper that started it all
2. **[Tensor2Tensor GitHub](https://github.com/tensorflow/tensor2tensor)** - Original TensorFlow implementation
3. **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)** - Best visual explanation, read this first
4. **[Hugging Face Docs](https://huggingface.co/docs/transformers/)** - Industry standard library for transformers
5. **[Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)** - Line-by-line PyTorch implementation with explanations

