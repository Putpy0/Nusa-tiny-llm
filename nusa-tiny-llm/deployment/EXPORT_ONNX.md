# Export to ONNX Format

## Overview

ONNX (Open Neural Network Exchange) is an open format for representing machine learning models. Exporting Nusa Tiny LLM to ONNX enables hardware-accelerated inference on various platforms.

## Benefits of ONNX Export

- **Hardware Acceleration**: Use GPU, NPU, or specialized AI accelerators
- **Cross-Platform**: Run on Windows, Linux, macOS, Android, iOS
- **Optimized Runtimes**: ONNX Runtime provides optimized execution
- **Integration**: Easy integration with existing applications

## Prerequisites

- Trained model weights
- PyTorch 1.12+
- `onnx` and `onnxruntime` packages
- Model configuration file

## Export Process

### 1. Install Dependencies

```bash
pip install onnx onnxruntime torch
```

### 2. Export Script

Create export script:

```python
import torch
from architecture.model import NusaTinyLLM
from architecture.config import ModelConfig

# Load config and model
config = ModelConfig.from_json("configs/model_config.json")
model = NusaTinyLLM(config)
model.load_state_dict(torch.load("checkpoints/model.pt", map_location="cpu"))
model.eval()

# Create dummy input
dummy_input = torch.ones(1, 128, dtype=torch.long)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "nusa-tiny-llm.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    }
)
```

### 3. Optimize ONNX Model

```bash
python -m onnxruntime.tools.onnx_model_utils \
    --output nusa-tiny-llm-optimized.onnx \
    nusa-tiny-llm.onnx
```

## Running ONNX Model

### Python Example

```python
import onnxruntime as ort
import numpy as np

# Create inference session
session = ort.InferenceSession(
    "nusa-tiny-llm-optimized.onnx",
    providers=["CPUExecutionProvider"]  # Or "CUDAExecutionProvider"
)

# Prepare input
input_ids = np.array([[1, 45, 123, 67, 2]], dtype=np.int64)

# Run inference
outputs = session.run(None, {"input_ids": input_ids})
logits = outputs[0]

# Get next token
next_token = np.argmax(logits[0, -1])
print(f"Next token: {next_token}")
```

### C++ Example

```cpp
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

Ort::Env env;
Ort::SessionOptions session_options;
Ort::Session session(env, "nusa-tiny-llm.onnx", session_options);

// Prepare input tensor
// ... (tensor creation code)

// Run inference
auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs, num_inputs, output_names, num_outputs);
```

## Provider Options

### CPU Execution (Default)
```python
providers=["CPUExecutionProvider"]
```
- Works everywhere
- No GPU required
- Good for VPS 2GB

### CUDA Execution (NVIDIA GPU)
```python
providers=["CUDAExecutionProvider"]
```
- Requires NVIDIA GPU
- CUDA toolkit installed
- Faster inference

### DirectML (Windows)
```python
providers=["DmlExecutionProvider"]
```
- Windows only
- DirectX 12 compatible GPU
- Good for Windows laptops

### CoreML (macOS/iOS)
```python
providers=["CoreMLExecutionProvider"]
```
- Apple devices only
- Hardware acceleration via Neural Engine

### NNAPI (Android)
```python
providers=["NNAPIExecutionProvider"]
```
- Android devices
- Mobile optimization

## Configuration for Nusa Tiny LLM

Ensure ONNX export matches model config:

```json
{
  "input_shape": [1, 1024],
  "output_shape": [1, 1024, 24576],
  "dtype": "int64",
  "opset": 17
}
```

## Optimization Techniques

### 1. Graph Optimization
```python
import onnxruntime.transformers.optimizer as optimizer

optimized_model = optimizer.optimize_model(
    "nusa-tiny-llm.onnx",
    model_type="bert",
    num_heads=20,
    hidden_size=1280
)
optimized_model.save_model_to_file("optimized.onnx")
```

### 2. Quantization
```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "nusa-tiny-llm.onnx",
    "nusa-tiny-llm-quantized.onnx",
    weight_type=QuantType.QUInt8
)
```

### 3. IO Binding
```python
io_binding = session.io_binding()
io_binding.bind_cpu_input("input_ids", input_data)
io_binding.bind_output("logits")
session.run_with_iobinding(io_binding)
result = io_binding.get_outputs()
```

## File Size Comparison

| Format | Size | Notes |
|--------|------|-------|
| PyTorch (.pt) | ~2.0 GB | FP32 weights |
| ONNX (.onnx) | ~2.0 GB | Unoptimized |
| ONNX Quantized | ~550 MB | INT8 quantized |

## Troubleshooting

### Opset Version Errors
Use opset 17 or higher for transformer models:
```python
torch.onnx.export(..., opset_version=17)
```

### Dynamic Shape Issues
Specify dynamic axes during export:
```python
dynamic_axes={
    "input_ids": {0: "batch", 1: "seq"},
    "logits": {0: "batch", 1: "seq"}
}
```

### Performance Issues
Enable graph optimization:
```python
session_options.graph_optimization_level = \
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

## Next Steps

1. Export model to ONNX
2. Test with ONNX Runtime
3. Choose appropriate execution provider
4. Benchmark performance
5. Deploy to target platform
