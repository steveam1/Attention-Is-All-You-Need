# Attention Is All You Need

**DS 5690 – Paper Project (Fall 2025)**

**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain & Google Research, 2017)

---

## Overview
Before the Transformer, most language models relied on Recurrent Neural Networks (RNNs) or Convolutional Neural Networks (CNNs) to handle sequential text. RNNs processed sentences strictly one word at a time which made training slow and impossible to parallelize. They also struggled to retain information from earlier in long sentences because of vanishing gradients. CNNs were faster since they could process multiple positions in parallel but they mainly captured local patterns (like short phrases) and needed many stacked layers to model long-range relationships. As datasets grew and tasks demanded deeper contextual understanding these limitations became major bottlenecks. Researchers began looking for an approach that could model global relationships efficiently without relying on recurrence or deep convolution.

### The Problem
The authors set out to solve this challenge by asking a key question:

Can we design a model that understands relationships between words, no matter how far apart, without relying on recurrence or convolution?

### The Approach
To solve this challenge, the author's proposed remove recurrence and convolution entirely and build a model that relies only on attention. Instead of reading sentences word by word like RNNs, the Transformer looks at all words at the same time and learns which ones should pay attention to each other. This makes training massively faster and helps the model capture long-distance relationships that older architectures struggled with.

The Transformer uses an encoder–decoder structure made of repeated blocks that include:

- Multi-Head Self-Attention: lets the model look at different types of word relationships in parallel.
- Feed-Forward Networks: apply a small neural network to each position to refine the representation.
- Positional Encoding: tells the model the order of the words since it doesn’t process them sequentially.
- Residual Connections + Layer Norm: help stabilize training and prevent information from being lost as it passes through layers.

Together, these components allow the Transformer to learn globally, train quickly, and scale to large datasets. In the original paper, this architecture set new records on machine translation tasks, showing that attention alone was not just simpler—it was better.

---

## Quesiton 1
What do you think the authors meant when they said ‘attention is all you need’? What makes attention powerful enough to replace recurrence and convolution?

Answer: Processing text one word at a time makes training very slow and limits how much data the model can handle efficiently. RNNs also struggle to capture long-range dependencies, meaning they forget earlier words as sentences get longer. The Transformer solved this by allowing all words to be processed at once through self-attention, making training faster and improving the model’s ability to understand context across the entire sentence.

## Model Architecture
The Transformer follows an encoder decoder structure but unlike older models, it removes recurrence and convolution completely. Instead, it relies entirely on self attention to understand relationships between words. This design allows every word in a sentence to look at all other words at once, capturing both nearby and distant dependencies efficiently.

## Pseudocode Description
Input: Sequence of tokens (words) → X = [x1, x2, ..., xn]

**Encoder:**
1. Convert tokens to embeddings  
2. Add positional encodings to preserve word order  
3. For each of the 6 encoder layers:  
    a. Apply multi-head self-attention to X  
    b. Apply feed-forward network  
    c. Add residual connection and layer normalization  
4. Output encoded representation **Z**

**Decoder:**

5. Take previous outputs (shifted right) as input **Y**  
6. Add positional encodings  
7. For each of the 6 decoder layers:  
    a. Apply masked multi-head self-attention to Y  
    b. Apply encoder–decoder attention using Z  
    c. Apply feed-forward network  
    d. Add residual connection and layer normalization  
8. Apply linear layer + softmax to produce next-word probabilities  

**Output:**  
Translated or generated sequence


## How It Differs from Previous Models

Traditional RNNs processed one word at a time, passing information step by step through a hidden state. This made training slow and caused long sentences to lose context. CNNs improved parallel processing but still struggled with long distance dependencies.

The Transformer changed this by replacing recurrence and convolution with attention. Self attention allows the model to find connections between all words at once, regardless of their position in the sentence. Multi head attention goes further by letting different heads focus on different types of relationships, such as syntax, meaning, or context. This architecture made training dramatically faster and more scalable while also improving accuracy. The key idea is that instead of remembering through sequence order, the model learns to focus on what matters most in every layer.

## Question 2
Why does the Transformer use multi head attention instead of a single attention mechanism? What advantage does it provide?

Answer: Multi head attention allows the model to learn multiple types of relationships between words at the same time. Each head focuses on a different aspect of the sentence, such as meaning, grammar, or context. When the outputs from all heads are combined, the model builds a richer understanding of how words relate to one another. This design makes the Transformer more accurate and expressive than using a single attention layer.

## Critical Analysis
This paper completely reshaped how researchers think about sequence modeling but it also introduced new questions that were not fully explored. One major issue the authors did not address in depth is the computational cost of attention. Self attention compares every token with every other token which grows quadratically with sequence length. This limits the model’s efficiency when dealing with very long text or high-resolution data. Later work, such as Longformer, Performer, and FlashAttention, focused on solving this by designing more memory-efficient attention mechanisms.

Another point that could have been developed further is the scope of experimentation. The paper mainly focused on machine translation, where it showed excellent results. However, the authors did not test its generality across other natural language tasks or modalities like vision or speech. Future research, including BERT, GPT, and Vision Transformers, showed just how adaptable this architecture really was.

The authors also offered limited analysis of how attention actually works. They demonstrated that the model performs well but not why. Later visualization studies revealed that attention heads often align with syntactic or semantic structures, providing interpretability the original paper lacked.

## Impact 
The Transformer completely changed the landscape of artificial intelligence by introducing a new foundation for how models process information. Before this paper, deep learning models relied heavily on recurrence or convolution which limited their ability to scale and slowed training. The Transformer removed those barriers, showing that attention alone could handle complex relationships across sequences while training faster and more efficiently.

Its impact was immediate and far-reaching. The architecture became the blueprint for nearly every major model that followed. BERT used the encoder portion of the Transformer to understand context in both directions, revolutionizing natural language understanding. GPT built on the decoder side to generate text, leading to modern large language models that power tools like ChatGPT. Beyond language, the same architecture inspired Vision Transformers (ViT) for image recognition, Audio Spectrogram Transformers for sound, and even applications in protein folding and genomics.

The Transformer also shifted how the AI community thinks about scale. It proved that larger models trained on massive datasets could continue improving with more computation—a trend that has defined modern AI research. From a historical standpoint, it marked the transition from traditional sequence models to the era of foundation models that can be fine-tuned for countless tasks. Attention Is All You Need did not just improve translation, it redefined how machines learn. It bridged the gap between efficiency and performance, inspiring a generation of architectures that continue to shape AI research and real-world applications today.

## Resource Links

Attention Is All You Need (Vaswani et al., 2017)
Official Implementation (Tensor2Tensor by Google Brain) – GitHub Repository
Illustrated Guide to Transformers – Jay Alammar’s Visual Explanation
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding – arXiv Link
The Annotated Transformer (Harvard NLP) – Walkthrough Code & Explanation

## Citation

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems (NeurIPS 2017). https://arxiv.org/abs/1706.03762
