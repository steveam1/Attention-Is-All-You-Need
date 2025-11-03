# Attention Is All You Need

**DS 5690 – Paper Project (Fall 2025)**

Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin

---

## Overview

Before Transformers, models like RNNs and CNNs dominated language processing but struggled to handle long-range dependencies and were hard to train efficiently.

In 2017, Vaswani et al. introduced the **Transformer**, a model that replaced recurrence and convolution entirely with **self-attention**. This idea changed everything about how we process language.

Instead of processing one word at a time, the Transformer looks at all words in a sentence simultaneously. This makes it much faster, easier to parallelize, and more accurate on translation tasks.

### Key Results
- **28.4 BLEU** on English→German translation
- **41.8 BLEU** on English→French translation
- Trained in only **12 hours on 8 GPUs** (much faster than previous models)

---

## Model Architecture

The Transformer follows an **encoder–decoder** structure with six layers each. Unlike RNNs, which work step by step, the Transformer uses attention to look at every word at once.

### Main Components

- **Multi-Head Self-Attention:** Lets the model focus on multiple relationships in a sentence at the same time.
- **Feed-Forward Layers:** Apply transformations at each position independently.
- **Residual Connections + Layer Normalization:** Improve training stability and convergence.
- **Positional Encoding:** Adds order information since there's no recurrence in the model.

### Architecture Pseudocode

**Transformer**
```python
def Transformer(InputSeq):
    InputEmbed = Embed(InputSeq) + PositionalEncoding()
    EncOutput = Encoder(InputEmbed)
    DecOutput = Decoder(EncOutput, TargetSeq)
    return Softmax(Linear(DecOutput))
```

**Encoder**
```python
for i in range(6):
    SelfAttn = MultiHeadAttention(X, X, X)
    FFN = FeedForward(SelfAttn)
    X = LayerNorm(X + FFN)
```

**Decoder**
```python
for i in range(6):
    MaskedAttn = MultiHeadAttention(Y, Y, Y)
    CrossAttn = MultiHeadAttention(MaskedAttn, EncOutput, EncOutput)
    FFN = FeedForward(CrossAttn)
    Y = LayerNorm(Y + FFN)
```

This structure allows the model to capture relationships between all words simultaneously instead of sequentially.

---

## Discussion Questions

1. **Why did the authors remove recurrence and convolution, and what benefits did that bring?**
   
2. **What problem does positional encoding solve in the Transformer?**

---

## Critical Analysis

While the Transformer was revolutionary, it had a few limitations:

- **Memory requirements:** It required large amounts of memory for long sequences.
- **Limited scope:** It didn't yet explore multimodal data (like combining text and images).
- **Interpretability:** The paper didn't fully explain how attention weights relate to meaning, so interpretability is still an open question.

Later models like **BERT** and **GPT** expanded on this architecture and improved scalability and performance.

Even with these limits, this paper completely redefined the standard for sequence modeling in AI.

---

## Impact

The Transformer architecture became the foundation for nearly all modern AI models. It made it possible to train much larger models that can generalize across tasks.

### Real-World Impacts

- Led to the creation of **BERT, GPT, T5, and Vision Transformers (ViT)**
- Enabled **transfer learning** and **few-shot/zero-shot** performance
- Transformed fields like **translation, summarization, question answering**, and even **image and audio modeling**

The phrase *"Attention is all you need"* basically became the motto for today's AI revolution.

---

## Code Demo

Here's a quick example showing how a small Transformer model works today using the Hugging Face `transformers` library:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Translate text
text = "Translate English to German: The cat is on the table."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)

# Print translation
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Output:** `Die Katze ist auf dem Tisch.`

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

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention is All You Need.** *Advances in Neural Information Processing Systems*, 30. https://arxiv.org/abs/1706.03762

---

## Resource Links

1. [Original Paper (arXiv)](https://arxiv.org/abs/1706.03762)
2. [Tensor2Tensor Code (GitHub)](https://github.com/tensorflow/tensor2tensor)
3. [The Illustrated Transformer (Blog)](https://jalammar.github.io/illustrated-transformer/)
4. [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
5. [Annotated Transformer (Harvard NLP)](http://nlp.seas.harvard.edu/annotated-transformer/)

---

## Presentation Information

**Presenter:** [Your Name]  
**Course:** DS 5690 – Topics in Data Science  
**Semester:** Fall 2025  
**Date:** [Presentation Date]
Annotated Transformer (Harvard NLP)

