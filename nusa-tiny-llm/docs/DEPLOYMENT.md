# Deployment Documentation

## Overview

This document provides guides for deploying Nusa Tiny LLM in various formats and environments. See individual deployment guides for specific export formats.

## Available Deployment Guides

1. **[EXPORT_GGUF.md](EXPORT_GGUF.md)** - Export to GGUF format
2. **[EXPORT_ONNX.md](EXPORT_ONNX.md)** - Export to ONNX format
3. **[QUANTIZATION.md](QUANTIZATION.md)** - Quantization strategies
4. **[LOW_RESOURCE.md](LOW_RESOURCE.md)** - Low-resource deployment

## Deployment Options

### Option 1: PyTorch Native

**Format:** `.pt` files
**Best for:** Development, research
**Size:** ~2GB (FP32), ~1GB (FP16)

```python
# Save model
torch.save({
    'config': model_config,
    'state_dict': model.state_dict()
}, 'nusa_tiny_llm.pt')

# Load model
checkpoint = torch.load('nusa_tiny_llm.pt')
model = NusaTinyLLM(checkpoint['config'])
model.load_state_dict(checkpoint['state_dict'])
```

### Option 2: GGUF Format

**Format:** `.gguf`
**Best for:** CPU inference, llama.cpp compatibility
**Size:** ~300MB (Q4_K_M)

See [EXPORT_GGUF.md](EXPORT_GGUF.md) for details.

### Option 3: ONNX Format

**Format:** `.onnx`
**Best for:** Cross-platform deployment, hardware acceleration
**Size:** ~1GB (FP32)

See [EXPORT_ONNX.md](EXPORT_ONNX.md) for details.

### Option 4: Quantized Formats

**Formats:** Q4_0, Q4_K_M, Q5_K_S, Q8_0
**Best for:** Low-resource devices
**Size:** 280-550MB

See [QUANTIZATION.md](QUANTIZATION.md) for details.

## Platform-Specific Deployment

### Desktop (Windows/macOS/Linux)

**Requirements:**
- Python 3.8+
- 4GB+ RAM
- 1GB+ storage

**Steps:**
1. Install dependencies: `pip install torch numpy`
2. Download model weights
3. Run inference script

**Example:**
```bash
python -m inference.cpu_inference --prompt "Hello"
```

### Smartphone (Android/iOS)

**Requirements:**
- 4GB+ RAM
- 500MB+ storage
- Android 10+ or iOS 15+

**Approach:**
1. Convert to GGUF format
2. Use llama.cpp mobile bindings
3. Or use ONNX Runtime Mobile

**Example (Android):**
```kotlin
// Using llama.cpp Android bindings
val model = LlamaModel("nusa_tiny_q4.gguf")
val output = model.generate("Halo", maxTokens = 128)
```

### VPS/Cloud

**Requirements:**
- 2GB+ RAM
- Linux OS
- SSH access

**Steps:**
1. Set up Python environment
2. Install dependencies
3. Download quantized model
4. Run as service

**Example Service (systemd):**
```ini
[Unit]
Description=Nusa Tiny LLM Inference Service
After=network.target

[Service]
Type=simple
User=deploy
WorkingDirectory=/opt/nusa-tiny-llm
ExecStart=/usr/bin/python -m inference.cpu_inference --serve
Restart=always

[Install]
WantedBy=multi-user.target
```

### Web Browser (Future)

**Approach:**
- WebAssembly with ONNX Runtime Web
- TensorFlow.js conversion

**Status:** Experimental

## Model Serving

### Simple HTTP API

```python
from flask import Flask, request, jsonify
from inference import NusaInference

app = Flask(__name__)
inference = NusaInference('checkpoints/model.pt')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    response = inference.generate(prompt)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Production Considerations

1. **Rate Limiting**: Prevent abuse
2. **Caching**: Cache common responses
3. **Batching**: Process multiple requests together
4. **Monitoring**: Track latency and errors
5. **Scaling**: Horizontal scaling for high load

## Security

### Access Control

```python
# API key authentication
def verify_api_key(key):
    return key in VALID_API_KEYS

@app.route('/generate', methods=['POST'])
def generate():
    api_key = request.headers.get('X-API-Key')
    if not verify_api_key(api_key):
        return jsonify({'error': 'Unauthorized'}), 401
    # ... proceed
```

### Input Validation

```python
def validate_input(text):
    # Length limit
    if len(text) > 1024:
        return False
    
    # Check for injection attempts
    dangerous_patterns = ['<script>', 'DROP TABLE', '../']
    for pattern in dangerous_patterns:
        if pattern in text:
            return False
    
    return True
```

### Output Filtering

```python
def filter_output(text):
    # Remove potentially harmful content
    # Implement content safety checks
    return sanitized_text
```

## Monitoring & Logging

### Metrics to Track

1. **Request Count**: Total API calls
2. **Latency**: Response time per request
3. **Error Rate**: Failed requests
4. **Token Usage**: Tokens generated/consumed
5. **Resource Usage**: CPU, memory

### Logging Setup

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('nusa_tiny_llm')

# Log each request
logger.info(f"Request: {prompt[:50]}... Response: {response[:50]}...")
```

## Backup & Recovery

### Model Backup

```bash
# Backup model and configs
tar -czf nusa_backup.tar.gz \
    checkpoints/ \
    configs/ \
    tokenizer/
```

### Version Control

Keep track of model versions:
```
models/
├── v0.1.0/
├── v0.2.0/
└── latest -> v0.2.0/
```

## Troubleshooting

### Common Deployment Issues

**Issue: Model loading fails**
- Verify file integrity
- Check disk space
- Ensure correct format

**Issue: Slow inference**
- Check CPU utilization
- Verify quantization is applied
- Reduce context length

**Issue: High memory usage**
- Use more aggressive quantization
- Limit concurrent requests
- Add swap space

## Cost Estimation

### VPS Deployment (Monthly)

| Provider | Instance | RAM | Storage | Cost/Month |
|----------|----------|-----|---------|------------|
| DigitalOcean | Basic | 2GB | 50GB | $12 |
| Linode | Nanode | 2GB | 50GB | $10 |
| AWS | t3.small | 2GB | 30GB | $15 |
| Vultr | Cloud | 2GB | 55GB | $12 |

### Self-Hosting vs Cloud

**Self-Hosting:**
- Higher upfront cost
- Full control
- No recurring fees

**Cloud:**
- Lower upfront cost
- Managed infrastructure
- Recurring fees

## Best Practices

1. **Start Small**: Test with smoke config first
2. **Monitor Resources**: Watch memory and CPU
3. **Use Quantization**: Essential for low-resource
4. **Implement Caching**: Reduce redundant computation
5. **Plan for Scale**: Design for growth
6. **Document Everything**: Maintain clear documentation
7. **Test Thoroughly**: Validate before production
8. **Have Rollback Plan**: Easy reversion to previous version
