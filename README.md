# Attention Is All You Need

**DS 5690 – Paper Project (Fall 2025)**

**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain & Google Research, 2017)

---

## 📘 Overview

### The Problem

Before 2017, if you wanted to build a model that could translate languages or understand text, you were basically stuck with two main options: Recurrent Neural Networks (RNNs) or Convolutional Neural Networks (CNNs). Both had some pretty serious issues. RNNs processed words one at a time in sequence, which meant they were super slow and had a hard time remembering information from earlier in a sentence (the infamous vanishing gradient problem). CNNs were faster but struggled to capture long-range dependencies between words that were far apart.

The real bottleneck was that these models couldn't be easily parallelized during training. RNNs had to wait for each time step to finish before moving to the next one, which made training painfully slow on long sequences. This sequential processing was a fundamental constraint that limited both the size of models we could train and how much data we could feed them.

### The Approach

Vaswani and his team at Google asked a bold question: what if we got rid of recurrence and convolution entirely? Instead, they proposed the **Transformer**, a model architecture built purely on **attention mechanisms**. The key insight was that attention allows the model to look at all words in a sentence simultaneously and figure out which words are relevant to each other, regardless of their distance in the sequence.

Think of it like this: when you read a sentence, you don't just process word by word. Your brain can jump around and connect "it" to "the cat" even if they're separated by several other words. That's basically what attention does.

### How It Works

The Transformer uses something called **self-attention** to compute representations of input sequences. For each word, it calculates how much attention to pay to every other word in the sentence. Then it uses these attention weights to create a weighted combination of all the word representations. By doing this multiple times in parallel (multi-head attention) and stacking several layers, the model builds up increasingly sophisticated representations.

Because there's no recurrence, the entire input can be processed in parallel, making training way faster. The authors added positional encodings to give the model information about word order, since attention itself doesn't care about sequence position.

### Results

The results were honestly kind of crazy. On the WMT 2014 English-to-German translation benchmark, the Transformer achieved **28.4 BLEU**, beating all previous models including ensembles by more than 2 BLEU points. On English-to-French, it hit **41.8 BLEU**, setting a new state-of-the-art.

But what really made people pay attention (pun intended) was the training time. The big model trained in just **3.5 days on 8 GPUs**. Previous state-of-the-art models took way longer. The base model trained in only **12 hours**, which was absolutely unheard of at the time for a model of this quality.

---

## ⚙️ Model Architecture

The Transformer uses an encoder-decoder architecture, but unlike previous sequence-to-sequence models, both the encoder and decoder are built entirely from attention layers and feed-forward networks. No recurrence, no convolution.

### Core Components

**1. Multi-Head Self-Attention**

This is the heart of the Transformer. Instead of having a single attention mechanism, the model runs several attention operations in parallel (8 heads in the base model). Each head can learn to focus on different aspects of the relationships between words.

The attention mechanism itself works by computing three vectors for each word: Query (Q), Key (K), and Value (V). The attention weight between two words is computed as the dot product of their Q and K vectors (scaled and normalized with softmax), and these weights are used to take a weighted sum of the V vectors.

**2. Position-wise Feed-Forward Networks**

After attention, each position goes through a two-layer fully connected network with a ReLU activation. This is applied identically at each position but with different parameters across layers. It gives the model more capacity to transform representations.

**3. Positional Encoding**

Since the model has no built-in notion of sequence order, positional encodings are added to the input embeddings. The authors use sine and cosine functions of different frequencies, which allows the model to learn to attend to relative positions.

**4. Residual Connections and Layer Normalization**

Each sub-layer (attention or feed-forward) is wrapped with a residual connection and layer normalization. This helps with training stability and allows gradients to flow more easily through the network.

### Differences from Previous Models

**vs. RNNs:** No sequential processing means full parallelization and O(1) path length between any two positions (RNNs have O(n) path length)

**vs. CNNs:** Can capture long-range dependencies in constant layers (CNNs need O(log n) layers with dilated convolutions)

**vs. Previous Attention Models:** First model to rely *entirely* on self-attention without recurrence

### Detailed Pseudocode

```python
# Main Transformer Architecture
def Transformer(input_seq, target_seq):
    """
    Full transformer model for sequence-to-sequence tasks
    
    Args:
        input_seq: Source sequence (e.g., English sentence)
        target_seq: Target sequence (e.g., German sentence)
    Returns:
        Output probabilities over vocabulary for each target position
    """
    # Embed input tokens and add positional encoding
    input_embedded = Embedding(input_seq)  # Shape: (seq_len, d_model)
    input_pos_encoded = input_embedded + PositionalEncoding(seq_len)
    
    # Encode input sequence
    encoder_output = Encoder(input_pos_encoded, num_layers=6)
    
    # Embed target tokens and add positional encoding
    target_embedded = Embedding(target_seq)
    target_pos_encoded = target_embedded + PositionalEncoding(target_len)
    
    # Decode with attention to encoder output
    decoder_output = Decoder(target_pos_encoded, encoder_output, num_layers=6)
    
    # Project to vocabulary and get probabilities
    logits = Linear(decoder_output, d_model -> vocab_size)
    output_probs = Softmax(logits)
    
    return output_probs


# Encoder Stack
def Encoder(x, num_layers=6):
    """
    Stack of identical encoder layers with self-attention
    
    Args:
        x: Input embeddings with positional encoding (seq_len, d_model)
        num_layers: Number of encoder layers (default: 6)
    Returns:
        Encoded representations (seq_len, d_model)
    """
    for layer in range(num_layers):
        # Self-attention: each position attends to all positions in input
        attn_output = MultiHeadAttention(
            query=x, 
            key=x, 
            value=x,
            num_heads=8,
            d_k=64,  # dimension per head
            d_v=64
        )
        # Add & Norm
        x = LayerNorm(x + attn_output)
        
        # Position-wise feed-forward network
        ffn_output = FeedForward(x, d_ff=2048)
        # Add & Norm
        x = LayerNorm(x + ffn_output)
    
    return x


# Decoder Stack
def Decoder(y, encoder_output, num_layers=6):
    """
    Stack of identical decoder layers with masked self-attention 
    and encoder-decoder attention
    
    Args:
        y: Target embeddings with positional encoding (target_len, d_model)
        encoder_output: Output from encoder (seq_len, d_model)
        num_layers: Number of decoder layers (default: 6)
    Returns:
        Decoded representations (target_len, d_model)
    """
    for layer in range(num_layers):
        # Masked self-attention: each position attends only to earlier positions
        # This prevents the decoder from "cheating" by looking at future tokens
        masked_attn_output = MultiHeadAttention(
            query=y,
            key=y,
            value=y,
            num_heads=8,
            mask=True  # Mask out future positions
        )
        # Add & Norm
        y = LayerNorm(y + masked_attn_output)
        
        # Encoder-decoder attention: decoder attends to encoder output
        cross_attn_output = MultiHeadAttention(
            query=y,  # From decoder
            key=encoder_output,  # From encoder
            value=encoder_output,  # From encoder
            num_heads=8
        )
        # Add & Norm
        y = LayerNorm(y + cross_attn_output)
        
        # Position-wise feed-forward network
        ffn_output = FeedForward(y, d_ff=2048)
        # Add & Norm
        y = LayerNorm(y + ffn_output)
    
    return y


# Multi-Head Attention Mechanism
def MultiHeadAttention(query, key, value, num_heads=8, d_k=64, d_v=64, mask=False):
    """
    Apply multiple attention heads in parallel
    
    Args:
        query: Query vectors (seq_len, d_model)
        key: Key vectors (seq_len, d_model)
        value: Value vectors (seq_len, d_model)
        num_heads: Number of parallel attention heads
        d_k: Dimension of queries and keys per head
        d_v: Dimension of values per head
        mask: Whether to apply masking (for decoder self-attention)
    Returns:
        Multi-head attention output (seq_len, d_model)
    """
    d_model = query.shape[-1]
    heads = []
    
    # Run attention in parallel for each head
    for i in range(num_heads):
        # Linear projections for this head
        Q_i = Linear(query, d_model -> d_k)
        K_i = Linear(key, d_model -> d_k)
        V_i = Linear(value, d_model -> d_v)
        
        # Scaled dot-product attention
        head_i = ScaledDotProductAttention(Q_i, K_i, V_i, mask)
        heads.append(head_i)
    
    # Concatenate all heads
    multi_head = Concat(heads)  # Shape: (seq_len, num_heads * d_v)
    
    # Final linear projection
    output = Linear(multi_head, num_heads * d_v -> d_model)
    
    return output


# Scaled Dot-Product Attention
def ScaledDotProductAttention(Q, K, V, mask=False):
    """
    Core attention mechanism using scaled dot-product
    
    Args:
        Q: Query matrix (seq_len, d_k)
        K: Key matrix (seq_len, d_k)
        V: Value matrix (seq_len, d_v)
        mask: Whether to mask future positions
    Returns:
        Attention output (seq_len, d_v)
    """
    d_k = Q.shape[-1]
    
    # Compute attention scores
    scores = MatMul(Q, Transpose(K))  # (seq_len, seq_len)
    scores = scores / sqrt(d_k)  # Scale by sqrt of dimension
    
    # Apply mask if needed (set future positions to -inf)
    if mask:
        scores = MaskFuture(scores, value=-inf)
    
    # Convert to probabilities
    attention_weights = Softmax(scores, dim=-1)
    
    # Apply attention to values
    output = MatMul(attention_weights, V)  # (seq_len, d_v)
    
    return output


# Position-wise Feed-Forward Network
def FeedForward(x, d_ff=2048):
    """
    Two-layer fully connected network applied to each position
    
    Args:
        x: Input (seq_len, d_model=512)
        d_ff: Hidden dimension (default: 2048)
    Returns:
        Output (seq_len, d_model=512)
    """
    hidden = Linear(x, d_model -> d_ff)
    hidden = ReLU(hidden)
    output = Linear(hidden, d_ff -> d_model)
    return output


# Positional Encoding
def PositionalEncoding(seq_len, d_model=512):
    """
    Generate sinusoidal positional encodings
    
    Args:
        seq_len: Length of sequence
        d_model: Model dimension (must match embedding size)
    Returns:
        Positional encodings (seq_len, d_model)
    """
    pos_encoding = zeros(seq_len, d_model)
    
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # Even dimensions: sine
            pos_encoding[pos, i] = sin(pos / 10000^(2i/d_model))
            # Odd dimensions: cosine
            if i + 1 < d_model:
                pos_encoding[pos, i+1] = cos(pos / 10000^(2i/d_model))
    
    return pos_encoding
```

This architecture enables parallel processing of all positions and constant-time communication between any pair of positions, which are the key innovations that made Transformers so much more efficient than previous models.

---

## 💬 Discussion Questions

### Question 1: Why Did the Authors Remove Recurrence and Convolution?

**Think about these points:**
- What computational bottlenecks did RNNs have during training?
- How does the sequential nature of RNNs limit parallelization?
- What advantages does self-attention provide in terms of learning long-range dependencies?
- Consider the path length between distant positions in the sequence

**Key insight:** The sequential computation in RNNs means you can't parallelize within a single training example. For a sequence of length n, you need n sequential operations. With self-attention, you only need 1 operation (though it has O(n²) complexity in terms of memory and computation). This tradeoff favors self-attention for sequences shorter than the model dimension, which is typically the case.

### Question 2: What Problem Does Positional Encoding Solve?

**Consider:**
- What information does self-attention inherently lack?
- Why is word order important for language understanding?
- How do the sine/cosine functions encode position?
- What would happen if we didn't include positional information?

**Key insight:** Self-attention is actually permutation-invariant—it treats the input as a set, not a sequence. Without positional encoding, "The cat ate the mouse" would be identical to "The mouse ate the cat" as far as the model is concerned. Positional encodings inject sequence order information while still allowing the model to easily learn to attend to relative positions.

---

## 🔍 Critical Analysis

While "Attention Is All You Need" was groundbreaking, it's worth examining what the paper overlooked or where subsequent research has improved on the original design.

### What the Authors Overlooked or Underexplored

**1. Quadratic Complexity for Long Sequences**

The biggest practical limitation is that self-attention has O(n²) time and memory complexity with respect to sequence length. For each position, you need to compute attention over all other positions, which means the cost grows quadratically. The paper briefly mentions this but doesn't explore it deeply. This became a major bottleneck when people tried to apply Transformers to really long sequences like entire documents or high-resolution images. Later work like Longformer, BigBird, and Linformer specifically addressed this with sparse attention patterns.

**2. Limited Interpretability Analysis**

The paper shows some cool attention visualizations in the appendix, but doesn't really dig into what the model is learning or why it works so well. Are the attention heads learning linguistic features like syntax? How much does each component contribute? Recent work has shown that attention weights don't always correspond to what we'd call "importance" in an interpretable sense, and that the model's behavior is more complex than just "attending to relevant words."

**3. Lack of Inductive Biases**

By removing all architectural assumptions (no recurrence for sequences, no convolution for locality), the Transformer relies entirely on learning everything from data. This is powerful but also means you need a LOT of data. The paper doesn't fully explore whether some architectural inductive biases might actually help, especially in low-data settings. This is why models like BERT needed massive pretraining datasets.

**4. Positional Encoding Limitations**

The sinusoidal positional encoding is clever, but it's additive rather than multiplicative, and it's the same across all layers. Some research has shown that learned positional embeddings or relative position representations (like in Transformer-XL) can work better. The paper doesn't explore these alternatives much.

### Errors or Disputed Findings

There aren't really any major errors in the paper—the math checks out and the results have been replicated many times. However, some claims have been refined by later work:

- The claim that "attention is all you need" is a bit overstated. Subsequent research showed that other components (like the feed-forward layers, layer norms, and residual connections) are actually crucial for performance.
- The paper suggests that multi-head attention helps because different heads can focus on different relationships. While this is true to some extent, research has shown that many heads learn redundant patterns, and you can actually prune a lot of heads without hurting performance much.

### What Could Have Been Developed Further

**Multimodal Extensions:** The paper focuses exclusively on text. It would have been interesting to see how the architecture handles other modalities like images or audio. Vision Transformers (ViT) didn't come until 2020, and they showed that with some modifications, Transformers work great for vision too.

**Analysis of Scaling Laws:** The paper shows that bigger models work better but doesn't systematically study how performance scales with model size, data size, and compute. This became a huge area of research with GPT-3 and other large language models.

**Pretraining Strategies:** The Transformer was trained from scratch on translation tasks. The paper doesn't explore pretraining on large unlabeled datasets, which turned out to be massive for BERT and GPT.

---

## 🌍 Impact

This paper didn't just advance the state-of-the-art—it fundamentally changed how we think about and build AI systems. It's not an exaggeration to say that the Transformer architecture is the foundation of modern AI.

### How It Changed the AI Landscape

**1. From Task-Specific to General-Purpose Architectures**

Before Transformers, researchers would design custom architectures for each task (one for translation, one for text classification, one for question answering, etc.). The Transformer showed that a single architecture could work across many tasks with minimal modifications. This shift toward general-purpose architectures accelerated progress massively because improvements to the core architecture benefited everyone working on any NLP task.

**2. Enabled Large-Scale Pretraining**

The parallelization benefits of Transformers made it feasible to train truly massive models on huge datasets. This led to the "pretrain then fine-tune" paradigm that dominates modern NLP:
- **BERT (2018):** Bidirectional Transformer pretrained on masked language modeling, set new records on 11 NLP tasks
- **GPT-2 (2019):** 1.5B parameter Transformer, showed impressive zero-shot capabilities
- **GPT-3 (2020):** 175B parameters, demonstrated few-shot learning and broad task generalization
- **GPT-4 (2023):** Multimodal capabilities, reasoning improvements

**3. Sparked the "Attention Revolution" Beyond NLP**

Once people saw how well Transformers worked for language, they started applying them everywhere:
- **Vision Transformers (ViT, 2020):** Showed that pure Transformers can match or beat CNNs for image classification
- **Audio Models:** Whisper for speech recognition, MusicGen for music generation
- **Multimodal Models:** CLIP (vision + language), Flamingo, GPT-4V
- **Reinforcement Learning:** Decision Transformer treats RL as a sequence modeling problem
- **Protein Folding:** AlphaFold 2 uses Transformers for 3D structure prediction
- **Time Series:** Transformers for forecasting, anomaly detection

### Intersection with Other Work

**Past Work It Built Upon:**
- Attention mechanisms from Bahdanau et al. (2014)
- Residual connections from ResNet (He et al., 2016)
- Layer normalization (Ba et al., 2016)
- Ideas about parallel computation from ByteNet (Kalchbrenner et al., 2016)

**Foundational for Later Work:**
- **BERT & Masked Language Modeling:** Used Transformer encoder for bidirectional pretraining
- **GPT Series:** Used Transformer decoder for autoregressive language modeling
- **T5:** "Text-to-Text Transfer Transformer" unified many NLP tasks
- **Switch Transformers & Mixture of Experts:** Scaled to trillions of parameters
- **Efficient Transformers:** Linformer, Performer, Reformer tackled the quadratic complexity issue
- **Retrieval-Augmented Generation:** Combined Transformers with external knowledge bases

### Real-World Applications Today

The impact isn't just academic—Transformers power tons of products and services you probably use:

- **ChatGPT, Claude, Gemini:** All built on Transformer architectures
- **Google Translate:** Switched to Transformer-based models in 2016-2017
- **GitHub Copilot:** Uses GPT models (Transformers) for code completion
- **Grammarly:** Transformer-based grammar and style checking
- **Recommendation Systems:** YouTube, Netflix use Transformers for personalization
- **Search Engines:** Google BERT helps understand search queries
- **Content Moderation:** Detecting harmful content on social media platforms

### Why This Paper Matters

In retrospect, "Attention Is All You Need" is one of those rare papers that defines an era. It's cited over 100,000 times (as of 2024) and counting. The title itself became iconic—a bold claim that turned out to be mostly true.

The Transformer didn't just improve translation scores. It provided a scalable, flexible, parallelizable architecture that could be trained on massive datasets and adapted to virtually any task. It opened the door to the current era of large language models and has fundamentally changed what we think is possible with AI.

---

## 💻 Code Demo

Let's see the Transformer in action! Here's a practical example using Hugging Face's `transformers` library, which has become the de facto standard for working with Transformer models.

### Basic Translation Example

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load a pretrained T5 model (based on the Transformer architecture)
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Translate English to German
text = "translate English to German: The Transformer architecture revolutionized NLP."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)

translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Translation: {translation}")
# Output: "Die Transformator-Architektur revolutionierte NLP."
```

### Exploring Attention Weights

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load BERT (uses Transformer encoder)
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

# Encode a sentence
text = "The cat sat on the mat."
inputs = tokenizer(text, return_tensors="pt")

# Get model outputs with attention weights
with torch.no_grad():
    outputs = model(**inputs)

# Access attention weights
# Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
attentions = outputs.attentions

# Look at attention from layer 6, head 3
layer_6_head_3 = attentions[5][0, 2]  # Layer 6, head 3
print(f"Attention shape: {layer_6_head_3.shape}")
print(f"Attention weights:\n{layer_6_head_3}")

# You can visualize which words "cat" attends to
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
cat_idx = tokens.index('cat')
cat_attention = layer_6_head_3[cat_idx]
for token, weight in zip(tokens, cat_attention):
    print(f"{token}: {weight:.4f}")
```

### Fine-Tuning a Transformer

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load a sentiment analysis dataset
dataset = load_dataset("imdb", split="train[:1000]")  # Small subset for demo

# Load pretrained model
model_name = "distilbert-base-uncased"  # Smaller, faster version of BERT
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set up training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=100,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Use the fine-tuned model
test_text = "This movie was absolutely fantastic! I loved every minute of it."
inputs = tokenizer(test_text, return_tensors="pt")
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1)
print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
```

### Building a Mini-Transformer from Scratch

Here's a simplified implementation to understand the core concepts:

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        return output
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        x = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and apply final linear
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.W_o(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# Example usage
d_model = 512
num_heads = 8
d_ff = 2048
batch_size = 2
seq_len = 10

# Create sample input
x = torch.randn(batch_size, seq_len, d_model)

# Create a single transformer block
transformer_block = TransformerBlock(d_model, num_heads, d_ff)

# Forward pass
output = transformer_block(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

These examples show how accessible Transformer models have become. You can use state-of-the-art models with just a few lines of code, or build your own from scratch to understand the internals!

---

## 📚 Citation

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

**Full Citation:**  
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is All You Need. *Advances in Neural Information Processing Systems*, 30. https://arxiv.org/abs/1706.03762

---

## 🔗 Resource Links

1. **[Original Paper (arXiv)](https://arxiv.org/abs/1706.03762)** - The original paper with all the mathematical details

2. **[Tensor2Tensor GitHub Repository](https://github.com/tensorflow/tensor2tensor)** - Official implementation from Google

3. **[The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/)** - Fantastic visual guide to understanding Transformers

4. **[Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)** - Industry-standard library for using pretrained Transformers

5. **[Annotated Transformer by Harvard NLP](http://nlp.seas.harvard.edu/annotated-transformer/)** - Line-by-line implementation guide with explanations

---

## 📊 Presentation Information

**Course:** DS 5690 – Topics in Data Science  
**Semester:** Fall 2025  
**Institution:** Vanderbilt University 
**Presenter:** Ashley Stevens
