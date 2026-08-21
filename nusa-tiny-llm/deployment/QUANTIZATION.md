# Quantization Guide

## Overview

Quantization reduces model precision to decrease size and improve inference speed. This guide covers quantization strategies for Nusa Tiny LLM.

## Why Quantize?

| Benefit | Description |
|---------|-------------|
| Smaller Size | 4-8x reduction in model size |
| Less RAM | Lower memory requirements |
| Faster Inference | Reduced computation time |
| Lower Power | Better battery life on mobile |

## Quantization Types

### FP32 (Full Precision)
- **Bits per weight**: 32
- **Model size**: ~2.0 GB (0.5B params)
- **RAM needed**: 4+ GB
- **Quality**: Baseline
- **Use case**: Training, research

### FP16 / BF16 (Half Precision)
- **Bits per weight**: 16
- **Model size**: ~1.0 GB
- **RAM needed**: 2+ GB
- **Quality**: Nearly identical to FP32
- **Use case**: GPU inference

### INT8 (8-bit)
- **Bits per weight**: 8
- **Model size**: ~550 MB
- **RAM needed**: 1.5 GB
- **Quality**: Minimal degradation
- **Use case**: Server deployment

### Q4_K_M (4-bit)
- **Bits per weight**: 4 (mixed precision)
- **Model size**: ~275 MB
- **RAM needed**: 700 MB
- **Quality**: Good for most tasks
- **Use case**: Mobile phones, low-RAM VPS

### Q2_K (2-bit)
- **Bits per weight**: 2 (mixed precision)
- **Model size**: ~175 MB
- **RAM needed**: 500 MB
- **Quality**: Noticeable degradation
- **Use case**: Extreme resource constraints

## Memory Requirements

For 0.5B parameter model:

| Quantization | Model Size | Min RAM | Recommended RAM |
|--------------|------------|---------|-----------------|
| FP32 | 2.0 GB | 4 GB | 8 GB |
| FP16 | 1.0 GB | 2 GB | 4 GB |
| Q8_0 | 550 MB | 1.5 GB | 2 GB |
| Q5_K_S | 344 MB | 1 GB | 1.5 GB |
| Q4_K_M | 275 MB | 700 MB | 1 GB |
| Q3_K_S | 219 MB | 600 MB | 800 MB |
| Q2_K | 175 MB | 500 MB | 700 MB |

## Quantization Methods

### Post-Training Quantization (PTQ)

Quantize after training is complete:

```bash
# Using llama.cpp
./quantize model-f16.gguf model-q4_k_m.gguf q4_k_m
```

**Pros:**
- Simple, no retraining needed
- Works with any trained model

**Cons:**
- Some quality loss
- May need calibration data

### Quantization-Aware Training (QAT)

Train with quantization simulation:

```python
# Pseudocode
model = prepare_for_quantization(model, bits=8)
train(model, dataset)
model = convert_to_quantized(model)
```

**Pros:**
- Better quality retention
- Optimized for target precision

**Cons:**
- Requires retraining
- More complex setup

## GGUF Quantization Formats

### K-quants (Recommended)

| Format | Description | Quality |
|--------|-------------|---------|
| Q2_K | Minimum viable | Low |
| Q3_K_S | Small, fast | Medium-Low |
| Q3_K_M | Balanced 3-bit | Medium |
| Q4_K_S | Small 4-bit | Medium-High |
| Q4_K_M | **Recommended** | High |
| Q5_K_S | Large 5-bit | Very High |
| Q5_K_M | Maximum 5-bit | Very High |
| Q6_K | Near-lossless | Excellent |
| Q8_0 | Virtually lossless | Best |

### Legacy Quants

| Format | Status | Notes |
|--------|--------|-------|
| Q4_0 | Legacy | Use Q4_K_S instead |
| Q4_1 | Legacy | Use Q4_K_M instead |
| Q5_0 | Legacy | Use Q5_K_S instead |
| Q5_1 | Legacy | Use Q5_K_M instead |

## Choosing the Right Quantization

### For HP 4GB RAM (Mobile)
- **Recommended**: Q4_K_M
- **Alternative**: Q3_K_M (if RAM constrained)
- **Avoid**: Q8_0, FP16

### For VPS 2GB RAM
- **Recommended**: Q4_K_M or Q5_K_S
- **Alternative**: Q3_K_S (for multiple instances)
- **Avoid**: FP16, FP32

### For Development/Testing
- **Recommended**: Q8_0 or Q6_K
- **Alternative**: FP16 if available
- **Avoid**: Q2_K, Q3_K_S

## Quality vs Size Trade-off

```
Quality
  ^
  |    ● FP32 (2.0 GB)
  |    ● FP16 (1.0 GB)
  |    ● Q8_0 (550 MB)
  |    ● Q6_K (420 MB)
  |    ● Q5_K_M (370 MB)
  |    ● Q5_K_S (344 MB)
  |    ● Q4_K_M (275 MB) ← Sweet spot
  |    ● Q4_K_S (250 MB)
  |    ● Q3_K_M (220 MB)
  |    ● Q3_K_S (190 MB)
  |    ● Q2_K (175 MB)
  +----------------------------------> Size
```

## Implementation Steps

### 1. Train Model (FP32/BF16)
```bash
python scripts/train.py --config configs/training_config.json
```

### 2. Export to GGUF
```bash
python convert.py --outfile nusa-tiny-f16.gguf /path/to/model/
```

### 3. Quantize
```bash
./quantize nusa-tiny-f16.gguf nusa-tiny-q4_k_m.gguf q4_k_m
```

### 4. Test Quality
```bash
./main -m nusa-tiny-q4_k_m.gguf -p "Test prompt" -n 64
```

### 5. Deploy
Copy `.gguf` file to target device and run with appropriate runtime.

## Troubleshooting

### Quality Degradation
- Try higher quantization level
- Use QAT instead of PTQ
- Check tokenizer compatibility

### Out of Memory
- Use lower quantization
- Reduce context length
- Close other applications

### Slow Inference
- Ensure using KV cache
- Use appropriate number of threads
- Consider ONNX Runtime

## Best Practices

1. **Always test** quantized model before deployment
2. **Keep original** FP16/FP32 weights for re-quantization
3. **Document** which quantization was used
4. **Benchmark** on target hardware
5. **Validate** bilingual performance (EN/ID)

## Resources

- [llama.cpp Quantization](https://github.com/ggerganov/llama.cpp#quantization)
- [GGUF Format Spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [ONNX Quantization](https://onnxruntime.ai/docs/performance/quantization.html)
