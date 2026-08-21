# Hardware Requirements and Deployment Guide

## Overview

This document outlines hardware requirements for different stages of the Nusa Tiny LLM project, from development to deployment on low-resource devices.

## Hardware Tiers

### Tier 1: Development & Training (Not Included)

**Purpose**: Full model training (0.5B parameters)

**Minimum Requirements:**
- GPU: NVIDIA RTX 3090/4090 (24GB VRAM) or equivalent
- RAM: 64GB system memory
- Storage: 500GB SSD
- CPU: 8+ cores

**Recommended:**
- GPU: Multi-GPU setup (A100/H100) or cloud instance
- RAM: 128GB+ system memory
- Storage: 1TB+ NVMe SSD
- CPU: 16+ cores

**Notes:**
- Training the full 0.5B model is NOT performed on low-resource devices
- This repository provides architecture and scripts only
- Actual training requires significant computational resources
- Estimated training time: days to weeks depending on hardware

### Tier 2: Inference on Desktop/Server

**Purpose**: Running full or quantized model for inference

**Full Precision (FP32):**
- RAM: 4GB+ free memory
- CPU: 4+ cores recommended
- Storage: 2.5GB for model weights
- OS: Any modern OS (Linux, Windows, macOS)

**Half Precision (FP16):**
- RAM: 2GB+ free memory
- CPU: 4+ cores with AVX2 support
- Storage: 1.2GB for model weights
- OS: Any modern OS

**Quantized (Q4_K_M):**
- RAM: 1GB+ free memory
- CPU: 2+ cores
- Storage: 300MB for model weights
- OS: Any modern OS

### Tier 3: Smartphone Deployment (4GB RAM)

**Purpose**: On-device inference for mobile applications

**Requirements:**
- Device RAM: 4GB total (2GB+ available for model)
- Storage: 500MB free space
- OS: Android 10+ or iOS 15+
- CPU: ARM Cortex-A75 or equivalent

**Supported Configurations:**

| Quantization | Model Size | RAM Usage | Performance |
|--------------|------------|-----------|-------------|
| Q4_0 | ~280MB | ~500MB | Fast |
| Q4_K_M | ~300MB | ~550MB | Balanced |
| Q5_K_S | ~350MB | ~600MB | Better quality |
| Q8_0 | ~550MB | ~800MB | Best quality |

**Recommendations:**
- Use Q4_K_M for best balance
- Limit context length to 512 tokens for faster response
- Close other apps before running inference
- Expect 5-20 tokens/second depending on device

**Example Devices:**
- Samsung Galaxy S20/S21 (8GB variant recommended)
- Google Pixel 5/6
- iPhone 12 or newer
- Mid-range Android with 6GB+ RAM preferred

### Tier 4: VPS Deployment (2GB RAM)

**Purpose**: Cloud inference on minimal VPS instances

**Requirements:**
- RAM: 2GB total (1.5GB available for model)
- CPU: 1-2 vCPU cores
- Storage: 10GB SSD
- OS: Linux (Ubuntu 20.04+, Debian 11+)

**Supported Configurations:**

| Quantization | Model Size | RAM Usage | Viability |
|--------------|------------|-----------|-----------|
| Q4_0 | ~280MB | ~450MB | ✓ Good |
| Q4_K_M | ~300MB | ~500MB | ✓ Good |
| Q4_K_S | ~290MB | ~480MB | ✓ Best for 2GB |
| Q5_K_S | ~350MB | ~600MB | ⚠ Tight |
| Q8_0 | ~550MB | ~800MB | ✗ Not recommended |

**Recommended VPS Providers:**
- DigitalOcean Droplet (Basic, 2GB)
- Linode Nanode (2GB)
- Vultr Cloud Compute (2GB)
- AWS t3.small (2GB)
- Google Cloud e2-small (2GB)

**Performance Expectations:**
- Context 512 tokens: 3-10 tokens/second
- Context 1024 tokens: 2-5 tokens/second
- Single concurrent user recommended
- Use swap space as backup (may slow down)

**Setup Tips:**
```bash
# Enable swap on 2GB VPS
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Monitor memory usage
watch -n 1 free -h

# Run with limited context
python -m inference.cpu_inference --max-context 512
```

## Memory Estimation Details

### Formula for KV Cache Memory

```
KV_Cache_MB = (2 × num_kv_heads × head_dim × hidden_size × max_seq_len × batch_size × dtype_bytes) / (1024²)
```

For our model (Q4_K_M, FP16 cache):
- num_kv_heads = 4
- head_dim = 64
- hidden_size = 1280 (for attention output)
- max_seq_len = 1024
- batch_size = 1
- dtype_bytes = 2 (FP16)

```
KV_Cache_MB = (2 × 4 × 64 × 1280 × 1024 × 1 × 2) / (1024²)
           = (2 × 4 × 64 × 1280 × 1024 × 2) / 1048576
           ≈ 128 MB
```

### Total Memory Breakdown (Q4_K_M, 1024 context)

| Component | Size | Notes |
|-----------|------|-------|
| Model Weights | 280 MB | Quantized Q4 |
| KV Cache | 128 MB | FP16, 1024 tokens |
| Activations | 64 MB | Forward pass |
| Input/Output | 32 MB | Token buffers |
| Overhead | 100 MB | Python, PyTorch runtime |
| **Total** | **~604 MB** | |

### Reducing Memory Usage

1. **Shorter Context**: Reduce from 1024 to 512 tokens saves ~64 MB
2. **Smaller Batch**: Always use batch_size=1 on low-resource
3. **Lower Precision**: Q4_0 instead of Q4_K_M saves ~20 MB
4. **Close Applications**: Free up system RAM
5. **Use Swap**: As last resort (slower)

## Performance Optimization

### CPU Inference Optimization

1. **Thread Affinity**: Pin threads to specific cores
```bash
export OMP_NUM_THREADS=2
export KMP_AFFINITY=granularity=fine,compact,1,0
```

2. **Memory Mapping**: Load model with mmap for large models
```python
model = torch.load(path, map_location='cpu', mmap=True)
```

3. **Operator Fusion**: Use optimized kernels when available

4. **Cache Warm-up**: Pre-allocate KV cache for expected length

### Mobile-Specific Optimizations

1. **NNAPI/CoreML**: Use hardware acceleration frameworks
2. **Model Splitting**: Load layers on-demand for very low RAM
3. **Background Processing**: Run inference in background thread
4. **Progressive Decoding**: Stream tokens as generated

### VPS-Specific Optimizations

1. **Process Priority**: Run with appropriate nice level
2. **Memory Limits**: Set ulimit to prevent OOM crashes
3. **Monitoring**: Set up alerts for memory pressure
4. **Auto-scaling**: Scale vertically during peak usage

## Smoke Test Configuration

For testing on any hardware without full model:

**Smoke Config Parameters:**
- hidden_size: 128
- num_layers: 2
- vocab_size: 512
- context_length: 128
- **Total parameters: ~100K**
- **Memory usage: <50 MB**

**Run smoke test:**
```bash
make smoke
```

This validates the pipeline without requiring significant resources.

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:**
- Process killed by OS
- CUDA out of memory errors
- System becomes unresponsive

**Solutions:**
1. Reduce context length
2. Use more aggressive quantization (Q4_0)
3. Close other applications
4. Add swap space (VPS)
5. Upgrade to device with more RAM

### Slow Inference

**Symptoms:**
- <1 token/second generation
- High latency per token

**Solutions:**
1. Reduce context length
2. Use fewer CPU threads (sometimes helps)
3. Check thermal throttling
4. Ensure no other heavy processes running
5. Consider quantization if not already using

### Model Loading Failures

**Symptoms:**
- Cannot load model file
- Corrupted weight errors

**Solutions:**
1. Verify file integrity (checksums)
2. Ensure sufficient free disk space
3. Check file permissions
4. Re-export model if corrupted

## Summary Table

| Device Type | RAM | Recommended Config | Expected Performance |
|-------------|-----|-------------------|---------------------|
| Desktop (Full) | 8GB+ | FP16 | 20-50 tok/s |
| Desktop (Quant) | 4GB+ | Q4_K_M | 30-80 tok/s |
| Smartphone | 4GB | Q4_K_M | 5-20 tok/s |
| VPS | 2GB | Q4_K_S | 3-10 tok/s |
| Smoke Test | 512MB | Smoke config | 100+ tok/s |

## Important Notes

⚠️ **Training Warning**: Do NOT attempt to train the 0.5B model on low-resource devices. Training requires GPU cluster or cloud infrastructure.

✅ **Inference Ready**: Quantized inference is feasible on specified low-resource devices.

📊 **Performance Varies**: Actual performance depends on specific hardware, background processes, and implementation optimizations.

🔄 **Future Improvements**: Consider ONNX Runtime, TensorRT, or CoreML for better performance on supported platforms.
