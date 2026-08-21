# Low-Resource Deployment Guide

## Target Devices

This guide covers deployment of Nusa Tiny LLM on:
- **HP 4GB RAM**: Android smartphones, budget laptops
- **VPS 2GB RAM**: Budget cloud instances, Raspberry Pi 4/5

## Hardware Requirements

### Minimum for Inference

| Component | HP 4GB | VPS 2GB |
|-----------|--------|---------|
| RAM | 4 GB | 2 GB |
| Storage | 1 GB free | 500 MB free |
| CPU | Quad-core | Single core |
| OS | Android 10+ / Linux | Linux |

### Not for Training

**Important**: Training the full 0.5B model requires:
- GPU with 8+ GB VRAM (RTX 3060 or better)
- Or cloud services (Colab Pro, AWS, etc.)
- Estimated training time: 24-72 hours

This repository provides architecture and scripts only. Actual training must be done on more powerful hardware.

## Deployment Options

### Option 1: GGUF with llama.cpp (Recommended)

Best for CPU-only devices.

#### Steps:

1. **Install llama.cpp**
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j4
```

2. **Get Quantized Model**
```bash
# Download or create quantized model
wget https://example.com/nusa-tiny-q4_k_m.gguf
```

3. **Run Inference**
```bash
./main -m nusa-tiny-q4_k_m.gguf \
    -p "Halo, siapa kamu?" \
    -n 64 \
    -t 2 \
    --ctx-size 512
```

#### Memory Usage:
- Q4_K_M: ~700 MB RAM
- Context: ~100 MB additional

### Option 2: ONNX Runtime

Best for cross-platform deployment.

#### Steps:

1. **Install ONNX Runtime**
```bash
pip install onnxruntime
```

2. **Run Inference**
```python
import onnxruntime as ort

session = ort.InferenceSession(
    "nusa-tiny-quantized.onnx",
    providers=["CPUExecutionProvider"]
)

# Run inference
outputs = session.run(None, {"input_ids": input_data})
```

#### Memory Usage:
- Quantized ONNX: ~600 MB RAM
- Overhead: ~200 MB

### Option 3: Python Native

For development and testing.

#### Steps:

1. **Install Dependencies**
```bash
pip install torch numpy
```

2. **Run with CPUInferenceEngine**
```python
from inference.cpu_inference import CPUInferenceEngine

engine = CPUInferenceEngine(
    config_path="configs/smoke_config.json",
    quantize=True
)

response = engine.generate("Hello, how are you?")
print(response)
```

#### Memory Usage:
- Smoke config: ~50 MB RAM
- Full model: 2+ GB (not recommended for low-resource)

## Configuration for Low-Resource

### Recommended Settings

```json
{
  "max_position_embeddings": 512,
  "batch_size": 1,
  "use_kv_cache": true,
  "quantization": "q4_k_m",
  "threads": 2,
  "context_length": 512
}
```

### Reduce Memory Further

1. **Shorter Context**
```bash
--ctx-size 256  # Instead of 1024
```

2. **Lower Quantization**
```bash
./quantize model.gguf model-q3_k_s.gguf q3_k_s
```

3. **Single Thread** (if RAM very limited)
```bash
-t 1
```

## Performance Expectations

### HP 4GB RAM (Snapdragon 660+)

| Metric | Value |
|--------|-------|
| Tokens/sec | 5-15 |
| First token | < 1 second |
| RAM usage | 700 MB |
| Battery impact | Moderate |

### VPS 2GB (1 vCPU)

| Metric | Value |
|--------|-------|
| Tokens/sec | 2-8 |
| First token | 1-2 seconds |
| RAM usage | 600 MB |
| Concurrent users | 1-2 |

### Smoke Config (Testing)

| Metric | Value |
|--------|-------|
| Tokens/sec | 20-50 |
| RAM usage | 50 MB |
| Use case | Development only |

## Mobile Deployment (Android)

### Using MLC LLM

1. **Convert to MLC Format**
```bash
python -m mlc_llm.convert \
    --model nusa-tiny \
    --quantization q4f16_1 \
    --output dist/
```

2. **Integrate with App**
```kotlin
// Kotlin example
val engine = MLCEngine()
engine.reload("nusa-tiny-q4f16_1")
val response = engine.generate(prompt)
```

### Using MediaPipe Tasks

1. **Export to Task format**
2. **Add to Android project**
3. **Use Task API for inference**

## VPS Deployment

### Docker Setup

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "-m", "inference.cpu_inference", "--port", "8000"]
```

### Systemd Service

```ini
[Unit]
Description=Nusa Tiny LLM Inference
After=network.target

[Service]
Type=simple
User=nusa
WorkingDirectory=/opt/nusa-tiny-llm
ExecStart=/usr/bin/python3 -m inference.cpu_inference --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name api.example.com;

    location /generate {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Troubleshooting

### Out of Memory

**Symptoms**: App crashes, OOM errors

**Solutions**:
1. Use lower quantization (Q3 instead of Q4)
2. Reduce context length
3. Close other applications
4. Use smoke config for testing

### Slow Inference

**Symptoms**: < 2 tokens/second

**Solutions**:
1. Increase thread count (-t 4)
2. Use shorter context
3. Ensure KV cache enabled
4. Check CPU frequency throttling

### Quality Issues

**Symptoms**: Nonsensical output

**Solutions**:
1. Use higher quantization (Q5 or Q8)
2. Check tokenizer compatibility
3. Verify prompt formatting
4. Adjust temperature (try 0.5-0.8)

## Monitoring

### Memory Usage
```bash
watch -n 1 'ps aux | grep nusa | awk "{print $6}"'
```

### CPU Usage
```bash
top -p $(pgrep -f nusa-tiny)
```

### Logs
```bash
tail -f /var/log/nusa-tiny.log
```

## Security Considerations

1. **Rate Limiting**: Prevent abuse
2. **Input Validation**: Sanitize prompts
3. **Output Filtering**: Block harmful content
4. **Access Control**: Authenticate API requests

## Cost Estimates

### VPS 2GB/month
- DigitalOcean Droplet: $12/month
- Linode Nanode: $10/month
- AWS t3.small: ~$15/month

### One-time Costs
- Model training (GPU rental): $50-200
- Development time: Variable

## Next Steps

1. Choose deployment option
2. Prepare quantized model
3. Test on target hardware
4. Set up monitoring
5. Deploy to production

## Resources

- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [ONNX Runtime](https://onnxruntime.ai/)
- [MLC LLM](https://mlc.ai/mlc-llm/)
