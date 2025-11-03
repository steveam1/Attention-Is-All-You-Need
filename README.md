# 🧠 Attention Is All You Need  
### DS 5690 – Paper Project (Fall 2025)  

---

## 📘 Overview  
Before Transformers, models like RNNs and CNNs dominated language processing but struggled to handle long-range dependencies and were hard to train efficiently.  

In 2017, Vaswani et al. introduced the **Transformer**, a model that replaced recurrence and convolution entirely with **self-attention**. This idea changed everything about how we process language.  

Instead of processing one word at a time, the Transformer looks at all words in a sentence simultaneously. This makes it much faster, easier to parallelize, and more accurate on translation tasks.  

**Key Results:**  
- 28.4 BLEU on English→German translation  
- 41.8 BLEU on English→French translation  
- Trained in only 12 hours on 8 GPUs (much faster than previous models)  

---

## ⚙️ Model Architecture  
The Transformer follows an **encoder–decoder** structure with six layers each.  
Unlike RNNs, which work step by step, the Transformer uses attention to look at every word at once.  

### Main Components  
- **Multi-Head Self-Attention:** Lets the model focus on multiple relationships in a sentence at the same time.  
- **Feed-Forward Layers:** Apply transformations at each position independently.  
- **Residual Connections + Layer Normalization:** Improve training stability and convergence.  
- **Positional Encoding:** Adds order information since there’s no recurrence in the model.  

---

### 🧩 Pseudocode (Simplified)
```python
def Transformer(InputSeq):
    InputEmbed = Embed(InputSeq) + PositionalEncoding()
    EncOutput = Encoder(InputEmbed)
    DecOutput = Decoder(EncOutput, TargetSeq)
    return Softmax(Linear(DecOutput))
Encoder

python
Copy code
for i in range(6):
    SelfAttn = MultiHeadAttention(X, X, X)
    FFN = FeedForward(SelfAttn)
    X = LayerNorm(X + FFN)
Decoder

python
Copy code
for i in range(6):
    MaskedAttn = MultiHeadAttention(Y, Y, Y)
    CrossAttn = MultiHeadAttention(MaskedAttn, EncOutput, EncOutput)
    FFN = FeedForward(CrossAttn)
    Y = LayerNorm(Y + FFN)
This structure allows the model to capture relationships between all words simultaneously instead of sequentially.

💬 Discussion Questions
Why did the authors remove recurrence and convolution, and what benefits did that bring?

What problem does positional encoding solve in the Transformer?

🔍 Critical Analysis
While the Transformer was revolutionary, it had a few limitations:

It required large amounts of memory for long sequences.

It didn’t yet explore multimodal data (like combining text and images).

The paper didn’t fully explain how attention weights relate to meaning, so interpretability is still an open question.

Later models like BERT and GPT expanded on this architecture and improved scalability and performance.

Even with these limits, this paper completely redefined the standard for sequence modeling in AI.

🌍 Impact
The Transformer architecture became the foundation for nearly all modern AI models.
It made it possible to train much larger models that can generalize across tasks.

Real-World Impacts:

Led to the creation of BERT, GPT, T5, and Vision Transformers (ViT)

Enabled transfer learning and few-shot/zero-shot performance

Transformed fields like translation, summarization, question answering, and even image and audio modeling

The phrase “Attention is all you need” basically became the motto for today’s AI revolution.

💻 Code Demo (Optional)
Here’s a quick example showing how a small Transformer model works today using the Hugging Face transformers library:

python
Copy code
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

text = "Translate English to German: The cat is on the table."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
📚 Citation
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).
Attention is All You Need. Advances in Neural Information Processing Systems, 30.
https://arxiv.org/abs/1706.03762

Resource Links
Original Paper (arXiv)
Tensor2Tensor Code
The Illustrated Transformer (Blog)
Hugging Face Transformers
Annotated Transformer (Harvard NLP)

