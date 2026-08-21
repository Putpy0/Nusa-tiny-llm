# Architecture Documentation

## Overview

Nusa Tiny LLM implements a decoder-only transformer architecture designed for bilingual English-Indonesian language understanding and generation. The model targets approximately 500 million parameters with optimizations for low-resource inference.

## Decoder-Only Transformer Design

### Why Decoder-Only?

Decoder-only architectures have proven highly effective for autoregressive language modeling tasks:

1. **Simplicity**: Single stack of transformer blocks without encoder-decoder attention
2. **Efficiency**: Causal masking enables efficient training with simple cross-entropy loss
3. **Scalability**: Proven by GPT family, LLaMA, and other successful models
4. **Flexibility**: Same architecture handles various tasks through prompting

Our decoder-only design follows the standard causal language modeling approach where each token can only attend to previous tokens in the sequence.

## Hidden Size: 1280

### Rationale

The hidden size of 1280 was chosen through careful consideration:

1. **Parameter Budget**: With 24 layers, this provides good capacity within 550M budget
2. **Memory Efficiency**: Balances representational power with inference memory requirements
3. **Hardware Alignment**: Divisible by common head counts for efficient matrix operations
4. **Scaling Laws**: Follows established scaling patterns for sub-billion parameter models

Compared to common choices:
- Smaller models (512-768): Limited capacity for complex reasoning
- Larger models (1536-2048): Exceed memory budget for low-resource deployment
- **1280**: Sweet spot for our target use case

## Number of Layers: 24

### Design Decision

24 transformer layers provide:

1. **Depth for Abstraction**: Sufficient layers for hierarchical feature learning
2. **Training Stability**: Manageable gradient flow without excessive depth
3. **Inference Speed**: Reasonable latency on CPU with KV caching
4. **Parameter Distribution**: ~67% of parameters in attention + MLP layers

Layer distribution:
- Embedding: ~31M parameters (tied)
- Per Layer: ~19M parameters
- Total Layers: 24 × 19M = ~456M
- Output: Tied with embedding

## Grouped-Query Attention (GQA)

### Implementation Choice

We use Grouped-Query Attention with 20 query heads and 4 key-value heads.

**Why GQA?**

1. **Memory Reduction**: KV cache reduced by 5× compared to multi-head attention
   - MHA: 20 KV heads
   - GQA: 4 KV heads (each shared by 5 query heads)
   
2. **Inference Speed**: Lower memory bandwidth requirements during generation
3. **Quality Preservation**: Minimal quality loss compared to full MHA
4. **Low-Resource Focus**: Critical for 4GB RAM devices

**Calculation:**
- Query heads: 20
- KV heads: 4
- Groups per KV head: 20 / 4 = 5 query heads per KV head

Each group of 5 query heads shares the same key and value projections.

## Rotary Position Embedding (RoPE)

### Why RoPE?

Rotary Position Embeddings offer several advantages:

1. **Length Extrapolation**: Better performance on sequences longer than training length
2. **Relative Position Encoding**: Implicitly encodes relative positions through rotation
3. **No Additional Parameters**: Position information without extra learnable weights
4. **Rotation Property**: Preserves dot product relationships under rotation

**Implementation:**
- Applied to query and key vectors before attention
- Rotation angle based on position and dimension index
- Theta base: 10000 (standard value)
- Applied per attention head with dimension pairing

**Formula:**
```
RoPE(x, pos) = rotate(x, pos, theta)
where theta_i = 10000^(-2i/d)
```

## RMSNorm

### Normalization Choice

Root Mean Square Layer Normalization (RMSNorm) instead of LayerNorm:

1. **Computational Efficiency**: No mean centering, fewer operations
2. **Equivalent Performance**: Empirically similar results to LayerNorm
3. **Simpler Gradient Flow**: Fewer operations in backward pass
4. **Standard Practice**: Adopted by LLaMA and modern architectures

**Formula:**
```
RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ
```

Where γ is a learnable scale parameter (no bias).

**Benefits:**
- ~7-10% faster computation
- Reduced numerical instability
- Simpler implementation

## SwiGLU Activation

### Activation Function

Swish Gated Linear Unit (SwiGLU) combines Swish activation with gating:

**Structure:**
```
SwiGLU(x) = Swish(xW₁ + b₁) ⊗ (xW₂ + b₂)
```

Where ⊗ is element-wise multiplication.

**Why SwiGLU?**

1. **Better Performance**: Outperforms ReLU and GeLU in transformer models
2. **Gating Mechanism**: Allows dynamic feature selection
3. **Smooth Gradients**: Swish component provides smooth non-linearity
4. **Proven Track Record**: Used successfully in PaLM, LLaMA, etc.

**Implementation Details:**
- intermediate_size = 4096 (3.2× hidden_size for SwiGLU)
- Two projection matrices: gate and value
- Element-wise multiplication followed by output projection

## Parameter Estimation

### Detailed Calculation

**Embedding Layer:**
- vocab_size × hidden_size = 24,576 × 1,280 = 31,457,280
- Tied with output projection (not counted twice)

**Per Transformer Layer:**

1. **Attention:**
   - Q projection: hidden_size × hidden_size = 1,280 × 1,280 = 1,638,400
   - K projection: hidden_size × (num_kv_heads × head_dim) = 1,280 × 256 = 327,680
   - V projection: hidden_size × (num_kv_heads × head_dim) = 1,280 × 256 = 327,680
   - O projection: hidden_size × hidden_size = 1,280 × 1,280 = 1,638,400
   - Attention total: 3,932,160

2. **MLP (SwiGLU):**
   - Gate projection: hidden_size × intermediate_size = 1,280 × 4,096 = 5,242,880
   - Value projection: hidden_size × intermediate_size = 1,280 × 4,096 = 5,242,880
   - Output projection: intermediate_size × hidden_size = 4,096 × 1,280 = 5,242,880
   - MLP total: 15,728,640

3. **Normalization:**
   - RMSNorm (pre-attn): hidden_size = 1,280
   - RMSNorm (pre-mlp): hidden_size = 1,280
   - Norm total: 2,560

**Per Layer Total:** 3,932,160 + 15,728,640 + 2,560 = 19,663,360

**Full Model:**
- Embedding: 31,457,280
- 24 Layers: 24 × 19,663,360 = 471,920,640
- **Total: 503,377,920 parameters** (~503M)

**Within Budget:** ✓ (550M max)

## Memory Estimation for Inference

### Full Precision (FP32)
- Model weights: 503M × 4 bytes = 2,012 MB (~2 GB)
- KV Cache (1024 context): ~512 MB
- Activations: ~256 MB
- **Total: ~2.8 GB**

### Half Precision (FP16)
- Model weights: 503M × 2 bytes = 1,006 MB (~1 GB)
- KV Cache: ~256 MB
- Activations: ~128 MB
- **Total: ~1.4 GB**

### Quantized (Q4_K_M)
- Model weights: 503M × 0.5 bytes ≈ 252 MB
- KV Cache (FP16): ~256 MB
- Overhead: ~100 MB
- **Total: ~600 MB**

This makes Q4 quantization suitable for 4GB RAM devices with room for OS and application.

## Context Length: 1024 Tokens

### Design Constraint

Limited to 1024 tokens for specific reasons:

1. **Memory Constraints**: KV cache scales linearly with sequence length
2. **Training Efficiency**: Shorter sequences enable larger batch sizes
3. **Use Case Fit**: Most mobile/VPS use cases involve short interactions
4. **RoPE Extrapolation**: Can potentially handle longer contexts at inference

**Trade-offs:**
- ✓ Lower memory footprint
- ✓ Faster training iterations
- ✗ Cannot process long documents directly
- Mitigation: Chunking or summarization for longer inputs

## Bilingual Design: English-Indonesian

### Architecture Support

The model architecture inherently supports multiple languages:

1. **Shared Vocabulary**: 24,576 tokens cover both languages
2. **Unified Embeddings**: Same embedding space for EN and ID
3. **Code-Switching**: Natural handling of mixed language input
4. **Cross-Lingual Transfer**: Knowledge transfers between languages

### Tokenizer Considerations

- Byte-level BPE handles characters from both languages
- Special tokens for language identification (optional)
- Balanced training data ensures equal capability

### Training Strategy

- Mixed bilingual batches
- Language-balanced synthetic data
- Shared attention mechanisms learn cross-lingual patterns

## Summary

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Architecture | Decoder-only | Simplicity, efficiency, proven |
| Hidden Size | 1280 | Balance capacity and memory |
| Layers | 24 | Depth for abstraction |
| Attention | GQA (20Q/4KV) | 5× KV cache reduction |
| Positions | RoPE | Length extrapolation |
| Normalization | RMSNorm | Efficiency |
| Activation | SwiGLU | Performance |
| Context | 1024 | Low-resource optimization |
| Languages | EN+ID | Bilingual support |

This architecture delivers capable bilingual language modeling within strict resource constraints.
