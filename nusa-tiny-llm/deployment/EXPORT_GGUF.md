# Export to GGUF Format

## Overview

GGUF (GPT-Generated Unified Format) is the standard format for running LLMs with llama.cpp. This format enables efficient CPU inference on low-resource devices.

## Prerequisites

- Trained model weights (`.pt` file)
- Model configuration (`model_config.json`)
- Python 3.8+
- `llama.cpp` tools installed

## Step-by-Step Export Process

### 1. Prepare Model Files

Ensure you have:
```
checkpoints/
  model.pt           # Trained weights
  config.json        # Model configuration
tokenizer/
  tokenizer.json     # Tokenizer vocabulary
```

### 2. Convert to GGUF

Using llama.cpp conversion script:

```bash
cd llama.cpp
python convert.py --outfile nusa-tiny-llm-f16.gguf /path/to/nusa-tiny-llm/
```

### 3. Quantize the Model

Choose quantization based on target device:

```bash
# Q4_K_M - Recommended for most devices (best balance)
./quantize nusa-tiny-llm-f16.gguf nusa-tiny-llm-q4_k_m.gguf q4_k_m

# Q5_K_S - Higher quality
./quantize nusa-tiny-llm-f16.gguf nusa-tiny-llm-q5_k_s.gguf q5_k_s

# Q8_0 - Near lossless
./quantize nusa-tiny-llm-f16.gguf nusa-tiny-llm-q8_0.gguf q8_0
```

## Quantization Options

| Method | Size | Quality | RAM Usage |
|--------|------|---------|-----------|
| q4_k_m | ~275MB | Good | 700MB |
| q5_k_s | ~344MB | Very Good | 1GB |
| q8_0 | ~550MB | Excellent | 1.5GB |

## Running the Model

### Using llama.cpp CLI

```bash
./main -m nusa-tiny-llm-q4_k_m.gguf \
    -p "Halo, apa kabar?" \
    -n 128 \
    --temp 0.7 \
    --top_p 0.9
```

### Using Python Binding

```python
import llama_cpp

llm = llama_cpp.Llama(
    model_path="nusa-tiny-llm-q4_k_m.gguf",
    n_ctx=1024,
    n_threads=4
)

output = llm(
    "Translate to Indonesian: Hello, how are you?",
    max_tokens=64,
    temperature=0.7
)

print(output['choices'][0]['text'])
```

## Configuration for Nusa Tiny LLM

When exporting, ensure these settings match the model architecture:

```json
{
  "vocab_size": 24576,
  "hidden_size": 1280,
  "num_hidden_layers": 24,
  "num_attention_heads": 20,
  "num_key_value_heads": 4,
  "head_dim": 64,
  "intermediate_size": 4096,
  "max_position_embeddings": 1024
}
```

## Troubleshooting

### Context Length Issues
If you encounter context length errors, ensure the GGUF was built with correct `n_ctx`:
```bash
./main -m model.gguf --ctx-size 1024
```

### Memory Errors
Reduce context size or use lower quantization:
```bash
./quantize model-f16.gguf model-q4_0.gguf q4_0
```

### Slow Inference
Use more CPU threads:
```bash
./main -m model.gguf -t 8
```

## File Sizes Reference

For 0.5B parameter model:
- FP16 GGUF: ~1.0 GB
- Q8_0 GGUF: ~550 MB
- Q5_K_S GGUF: ~344 MB
- Q4_K_M GGUF: ~275 MB

## Next Steps

After successful export:
1. Test on target device (HP 4GB / VPS 2GB)
2. Benchmark inference speed
3. Validate output quality
4. Deploy to production
