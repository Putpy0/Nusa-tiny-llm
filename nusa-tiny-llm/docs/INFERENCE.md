# Inference Documentation

## Overview

This document describes the inference engine for Nusa Tiny LLM, optimized for low-resource CPU deployment on smartphones (4GB RAM) and VPS (2GB RAM).

## Inference Modes

### Greedy Decoding

Select the highest probability token at each step:

```python
next_token = argmax(logits[-1])
```

**Pros:**
- Deterministic output
- Fastest decoding
- No hyperparameters

**Cons:**
- Can be repetitive
- Less diverse outputs

### Sampling

Sample from the probability distribution:

```python
probs = softmax(logits / temperature)
next_token = sample(probs)
```

**Hyperparameters:**
- `temperature`: Controls randomness (0.7 recommended)
- `top_k`: Sample from top k tokens (40 recommended)
- `top_p`: Nucleus sampling threshold (0.9 recommended)

## KV Cache

### Purpose

Avoid recomputing key-value states for previous tokens:

```python
# Without cache: O(n²) complexity
for i in range(seq_len):
    keys, values = compute_all_previous_tokens()

# With cache: O(n) complexity  
for i in range(seq_len):
    keys, values = get_cached_previous()
    new_k, new_v = compute_current_token()
    cache.update(new_k, new_v)
```

### Memory Usage

For our model (GQA, 1024 context):
```
KV_cache_size = 2 × num_kv_heads × head_dim × hidden_size × seq_len
              = 2 × 4 × 64 × 1280 × 1024 × 2 bytes (FP16)
              ≈ 128 MB
```

### Implementation

```python
class KVCache:
    def __init__(self, max_seq_len, num_kv_heads, head_dim):
        self.keys = torch.zeros(max_seq_len, num_kv_heads, head_dim)
        self.values = torch.zeros(max_seq_len, num_kv_heads, head_dim)
        self.position = 0
    
    def update(self, key, value):
        self.keys[self.position] = key
        self.values[self.position] = value
        self.position += 1
    
    def get(self):
        return self.keys[:self.position], self.values[:self.position]
```

## Generation Strategies

### Basic Generation Loop

```python
def generate(model, input_ids, max_new_tokens):
    generated = input_ids.clone()
    
    for _ in range(max_new_tokens):
        # Forward pass
        logits = model(generated)
        
        # Get next token
        next_token = argmax(logits[:, -1])
        
        # Append
        generated = torch.cat([generated, next_token], dim=-1)
        
        # Check stop condition
        if next_token == eos_token_id:
            break
    
    return generated
```

### With KV Cache

```python
def generate_with_cache(model, input_ids, max_new_tokens):
    cache = KVCache(max_seq_len=context_length)
    generated = input_ids.clone()
    
    # Prefill cache with prompt
    model.prefill(input_ids, cache)
    
    # Decode loop
    for _ in range(max_new_tokens):
        logits = model.decode_step(generated[:, -1:], cache)
        next_token = sample(logits, temperature=0.7)
        generated = torch.cat([generated, next_token], dim=-1)
        
        if next_token == eos_token_id:
            break
    
    return generated
```

## Stop Conditions

### Token-Based Stopping

Stop when specific tokens are generated:

```python
stop_tokens = [eos_token_id, sep_token_id]

if next_token in stop_tokens:
    generation_complete = True
```

### Length-Based Stopping

```python
if len(generated) >= max_length:
    generation_complete = True
```

### Pattern-Based Stopping (Future)

Stop when specific patterns appear:
- End of sentence markers
- Special conversation markers
- Repetitive patterns

## Temperature Scaling

### Effect of Temperature

```python
def apply_temperature(logits, temperature):
    if temperature == 0:
        return argmax(logits)  # Greedy
    else:
        probs = softmax(logits / temperature)
        return sample(probs)
```

**Temperature Values:**
- `T = 0.0`: Greedy (deterministic)
- `T = 0.5`: Conservative sampling
- `T = 0.7`: Balanced (recommended)
- `T = 1.0`: Standard sampling
- `T = 1.5`: Very diverse (may be incoherent)

## Top-K and Top-P Sampling

### Top-K Sampling

Only sample from the k most likely tokens:

```python
def top_k_sampling(logits, k):
    top_values, top_indices = topk(logits, k)
    probs = softmax(top_values)
    sampled_index = sample(probs)
    return top_indices[sampled_index]
```

### Top-P (Nucleus) Sampling

Sample from smallest set of tokens whose cumulative probability exceeds p:

```python
def top_p_sampling(logits, p):
    sorted_probs, sorted_indices = sort(softmax(logits), descending=True)
    cumulative_probs = cumsum(sorted_probs)
    
    # Find cutoff
    cutoff = where(cumulative_probs > p)[0][0]
    
    # Resample from truncated distribution
    truncated_probs = sorted_probs[:cutoff+1]
    truncated_probs /= truncated_probs.sum()
    
    sampled_index = sample(truncated_probs)
    return sorted_indices[sampled_index]
```

### Combined Strategy

```python
def sample_with_top_k_top_p(logits, k=40, p=0.9, temperature=0.7):
    # Apply temperature
    logits = logits / temperature
    
    # Apply top-k
    if k is not None:
        logits = apply_top_k(logits, k)
    
    # Apply top-p
    if p is not None:
        logits = apply_top_p(logits, p)
    
    # Sample
    probs = softmax(logits)
    return sample(probs)
```

## Repetition Penalty

### Prevent Repetition

Penalize tokens that have appeared before:

```python
def apply_repetition_penalty(logits, previous_tokens, penalty=1.05):
    for token_id in previous_tokens:
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits
```

**Penalty Values:**
- `penalty = 1.0`: No penalty
- `penalty = 1.05`: Mild penalty (recommended)
- `penalty = 1.1`: Moderate penalty
- `penalty = 1.2`: Strong penalty (may affect coherence)

## CPU Inference Optimization

### Threading Control

```bash
export OMP_NUM_THREADS=4
export KMP_AFFINITY=granularity=fine,compact,1,0
```

### Memory Mapping

Load model weights efficiently:

```python
model_data = torch.load(path, map_location='cpu', mmap=True)
```

### Quantization Support

See `inference/quantization_notes.md` for quantized inference.

## Usage Examples

### Basic Inference

```bash
python -m inference.cpu_inference \
  --prompt "Halo, siapa kamu?" \
  --config configs/inference_config.json \
  --checkpoint checkpoints/model.pt
```

### With Custom Parameters

```bash
python -m inference.cpu_inference \
  --prompt "Tell me a story" \
  --max-new-tokens 256 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 50
```

### Interactive Mode

```python
from inference import NusaInference

inference = NusaInference(
    model_path='checkpoints/model.pt',
    config_path='configs/inference_config.json'
)

while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    
    response = inference.generate(user_input)
    print(f"Assistant: {response}")
```

## Performance Estimation

### Tokens Per Second

| Device | Config | Performance |
|--------|--------|-------------|
| Desktop CPU | FP32 | 20-50 tok/s |
| Desktop CPU | Q4_K_M | 30-80 tok/s |
| Smartphone | Q4_K_M | 5-20 tok/s |
| VPS 2GB | Q4_K_S | 3-10 tok/s |

### Latency Breakdown

For single token generation:
- Model forward pass: 80%
- Sampling: 5%
- KV cache update: 10%
- Overhead: 5%

## Troubleshooting

### Issue: Slow generation

**Solutions:**
1. Reduce context length
2. Use quantized model
3. Enable threading optimizations
4. Close other applications

### Issue: Out of memory

**Solutions:**
1. Use more aggressive quantization
2. Reduce max context length
3. Decrease batch size to 1
4. Add swap space (VPS)

### Issue: Repetitive output

**Solutions:**
1. Increase temperature
2. Add repetition penalty
3. Use top-p sampling
4. Adjust prompt

### Issue: Incoherent output

**Solutions:**
1. Lower temperature
2. Reduce top-p
3. Check model quality
4. Improve prompt clarity
