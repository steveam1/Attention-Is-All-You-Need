# Attention Is All You Need

**DS 5690 – Paper Project (Fall 2025)**

**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain & Google Research, 2017)

---

## Overview
Before the Transformer, most language models used Recurrent Neural Networks (RNNs) or Convolutional Neural Networks (CNNs) to process sequential data like text. These architectures had a major limitation because they processed words one at a time, making training slow, memory-intensive, and difficult to process in parallel. They also struggled to capture long-range dependencies, meaning they often forgot earlier words in long sentences. As datasets and model sizes grew, these limitations became a major bottleneck for progress in natural language processing. Researchers began looking for a new approach that could handle sequences more efficiently while still understanding complex word relationships across an entire sentence.

### The Problem
The authors set out to solve this challenge by asking a key question:

Can we design a model that understands relationships between words, no matter how far apart, without relying on recurrence or convolution?

### The Approach
To solve this challenge, the authors introduced the Transformer, a model built entirely on attention mechanisms. Instead of reading text one word at a time like earlier models, the Transformer looks at every word in a sentence at once and learns which words are most connected to each other. This design makes training much faster and allows the model to capture relationships across the entire sentence, even between words that are far apart.

The Transformer follows an encoder-decoder structure made up of repeating layers that include:

- Multi-Head Self-Attention: lets the model focus on different types of word relationships, such as meaning, grammar, or context.
- Feed Forward Networks: refine and transform the information for each word position.
- Positional Encoding: adds information about word order since the model itself does not read sequentially.
- Residual Connections and Layer Normalization: keep learning stable and prevent information loss between layers.

This combination allowed the Transformer to train much more efficiently while improving translation accuracy. It was able to achieve record-breaking results on English-to-German and English-to-French translation tasks, proving that attention alone could outperform more complex and slower architectures.

---

## Quesiton 1
What do you think the authors meant when they said ‘attention is all you need’? What makes attention powerful enough to replace recurrence and convolution?

Answer: Processing text one word at a time makes training very slow and limits how much data the model can handle efficiently. RNNs also struggle to capture long-range dependencies, meaning they forget earlier words as sentences get longer. The Transformer solved this by allowing all words to be processed at once through self-attention, making training faster and improving the model’s ability to understand context across the entire sentence.

## Model Architecture
The Transformer follows an encoder decoder structure but unlike older models, it removes recurrence and convolution completely. Instead, it relies entirely on self attention to understand relationships between words. This design allows every word in a sentence to look at all other words at once, capturing both nearby and distant dependencies efficiently.

## Pseudocode Description
Input: Sequence of tokens (words) → X = [x1, x2, ..., xn]

# Encoder
1. Convert tokens to embeddings
2. Add positional encodings to preserve word order
3. For each of the 6 encoder layers:
      a. Apply multi head self attention to X
      b. Apply feed forward network to the result
      c. Add residual connection and layer normalization
4. Output encoded representation Z

# Decoder
5. Take previous outputs (shifted right) as input Y
6. Add positional encodings
7. For each of the 6 decoder layers:
      a. Apply masked multi head self attention to Y 
         (so the model cannot look ahead)
      b. Apply encoder decoder attention using Z
         (connects understanding from encoder to decoder)
      c. Apply feed forward network
      d. Add residual connection and layer normalization
8. Apply linear layer and softmax to produce probabilities for next word

Output: Translated or generated sequence

##How It Differs from Previous Models

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
