# Quantization Notes for Nusa Tiny LLM

## Overview

Quantization reduces model size and improves inference speed by using lower-precision representations of weights. This document explains quantization strategies for deploying Nusa Tiny LLM on low-resource devices.

## Supported Quantization Types

### Q4 (4-bit Quantization)
- **Size reduction**: ~8x smaller than FP32
- **Memory usage**: ~275 MB for 0.5B model
- **Quality**: Good for most tasks, slight degradation possible
- **Best for**: Mobile phones, very low RAM devices

### Q5 (5-bit Quantization)
- **Size reduction**: ~6.4x smaller than FP32
- **Memory usage**: ~344 MB for 0.5B model
- **Quality**: Very good, minimal quality loss
- **Best for**: Balance between size and quality

### Q8 (8-bit Quantization)
- **Size reduction**: ~4x smaller than FP32
- **Memory usage**: ~550 MB for 0.5B model
- **Quality**: Nearly identical to FP16
- **Best for**: Devices with moderate RAM

## Memory Estimates for 0.5B Model

| Format | Size | RAM Required (inference) |
|--------|------|-------------------------|
| FP32   | 2.0 GB | 4+ GB |
| FP16   | 1.0 GB | 2+ GB |
| Q8     | 550 MB | 1.5 GB |
| Q5     | 344 MB | 1 GB |
| Q4     | 275 MB | 700 MB |

## Export to GGUF Format

For optimal CPU inference, export to GGUF format:

```bash
# Conceptual - requires llama.cpp tools
python scripts/export_gguf.py \
    --model checkpoints/model.pt \
    --config configs/model_config.json \
    --output nusa-tiny-llm-q4.gguf \
    --quantization q4_k_m
```

### Recommended GGUF Quantization Types

- `q4_k_m`: Best balance for 0.5B models
- `q5_k_s`: Higher quality, slightly larger
- `q8_0`: Near-lossless compression

## Export to ONNX Format

ONNX enables hardware-accelerated inference:

```bash
# Conceptual - requires onnx export
python scripts/export_onnx.py \
    --model checkpoints/model.pt \
    --config configs/model_config.json \
    --output nusa-tiny-llm.onnx \
    --opset 17
```

### ONNX Runtime Optimization

```python
import onnxruntime as ort

# Create optimized session
session = ort.InferenceSession(
    "nusa-tiny-llm.onnx",
    providers=["CPUExecutionProvider"]
)

# For mobile: use MobileNet/NNAPI providers
# session = ort.InferenceSession(..., providers=["DmlExecutionProvider"])
```

## Implementation Notes

### Current Repository Status

This repository provides:
- Architecture definitions compatible with quantization
- Configuration files specifying target quantization formats
- Documentation for export procedures

Actual quantization requires:
1. Trained model weights (not included)
2. External tools (llama.cpp, onnxruntime)
3. Calibration dataset for optimal results

### Quantization-Aware Training (Future)

For best results, consider quantization-aware training:

```python
# Pseudocode for QAT
model = NusaTinyLLM(config)
model = apply_quantization_aware_training(model, bits=8)
train(model, dataset)
export_to_gguf(model, quantization="q4_k_m")
```

## Low-Resource Deployment Checklist

- [ ] Export model to GGUF format
- [ ] Choose appropriate quantization (q4_k_m recommended)
- [ ] Test on target device (HP 4GB / VPS 2GB)
- [ ] Verify memory usage stays within limits
- [ ] Benchmark inference speed
- [ ] Validate output quality

## Troubleshooting

### Out of Memory Errors
- Use lower quantization (Q4 instead of Q8)
- Reduce max_context_length
- Close other applications

### Slow Inference
- Ensure KV caching is enabled
- Use single-threaded mode on low-core devices
- Consider ONNX Runtime with appropriate providers

### Quality Degradation
- Try higher quantization level (Q5 or Q8)
- Check tokenizer compatibility
- Verify prompt formatting
