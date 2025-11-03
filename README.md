# Attention-Is-All-You-Need
Paper project for "Attention Is All You Need"

## 📘 Overview
Before Transformers, models like RNNs and CNNs dominated NLP but struggled with **long-range dependencies** and **limited parallelization**.  
This 2017 paper by Vaswani et al. proposed a new architecture called the **Transformer**, which relies entirely on *self-attention*—removing recurrence and convolution altogether.  

### Problem
Recurrent models process words one at a time, limiting speed and scalability.

### Approach
The Transformer uses **multi-head self-attention** to look at all tokens simultaneously, learning how words relate regardless of distance.

### Results
- **28.4 BLEU** on English→German translation  
- **41.8 BLEU** on English→French  
- Trained in **12 hours on 8 GPUs**—dramatically faster than previous models:contentReference[oaicite:0]{index=0}  

## ⚙️ Architecture Overview
The model follows an **encoder–decoder** design with six stacked layers in each.

### Key Components
- **Multi-Head Attention:** lets the model focus on multiple relationships in parallel  
- **Feed-Forward Networks:** transform representations at each position  
- **Residual Connections + Layer Norm:** stabilize training  
- **Positional Encoding:** injects token order since no recurrence exists  

### ✏️ Pseudocode
```python
function Transformer(InputSeq):
    InputEmbed = Embed(InputSeq) + PositionalEncoding()
    EncOutput = Encoder(InputEmbed)
    DecOutput = Decoder(EncOutput, TargetSeq)
    return Softmax(Linear(DecOutput))

