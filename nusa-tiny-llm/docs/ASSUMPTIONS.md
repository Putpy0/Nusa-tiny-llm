# ASSUMPTIONS

## General Assumptions

1. **Training Hardware**: Training the full 0.5B model requires GPU/Colab/server with sufficient VRAM (minimum 24GB recommended for efficient training). This repository does not perform actual training on low-resource devices.

2. **Inference Hardware**: 
   - HP 4GB RAM: Can run quantized model (q4_k_m) for inference only
   - VPS 2GB RAM: Can run heavily quantized model (q4_0) with limited context length
   - Full precision inference requires significantly more memory

3. **No Internet Dependency**: All core functionality works offline. Teacher adapter is disabled by default and uses mock provider.

4. **Synthetic Data Quality**: Mock synthetic data generator produces simple bilingual patterns for testing. Production-quality data requires actual teacher model or human curation.

5. **Tokenizer Training**: Custom BPE tokenizer is trained from scratch on synthetic data. No pre-existing vocabulary is used.

6. **Parameter Budget**: The target is ~500M parameters with maximum budget of 550M. Configuration may be adjusted if estimates exceed budget.

7. **Context Length**: Maximum 1024 tokens chosen to balance memory usage and practical utility for low-resource deployment.

8. **Bilingual Support**: Equal weight given to English and Indonesian in tokenizer training and synthetic data generation.

9. **Quantization**: GGUF format recommended for deployment. q4_k_m provides good balance between size and quality.

10. **Evaluation**: Built-in evaluation uses simple bilingual tests. Comprehensive evaluation requires additional benchmarks.

## Technical Assumptions

1. **PyTorch Availability**: PyTorch CPU version is available for all operations. CUDA support is optional for training.

2. **Python Version**: Python 3.8+ required for f-strings and type hints.

3. **Memory Estimation**: Memory estimates assume overhead factors for safety margin.

4. **Grouped-Query Attention**: GQA with 4 KV heads reduces memory footprint during inference while maintaining quality.

5. **SwiGLU Activation**: Chosen for better performance than ReLU/GeLU in transformer architectures.

6. **RoPE Positional Encoding**: Provides better length extrapolation than absolute positional embeddings.

7. **RMSNorm**: Simpler computation than LayerNorm with comparable performance.

8. **Tied Embeddings**: Word embeddings tied with output projection to reduce parameter count.

## Limitations

1. **No Pretrained Weights**: Model trains from random initialization. No knowledge distillation from larger models.

2. **Synthetic Data Only**: No real-world corpus included. Quality depends on synthetic data generation strategy.

3. **Smoke Test Only**: Full training loop structure provided but not executed in repository tests.

4. **Basic Tokenizer**: Custom BPE implementation is functional but may not match quality of established tokenizers.

5. **CPU Inference**: Optimized for CPU but may be slow without hardware acceleration.

## Future Considerations

1. **Teacher Model Integration**: Optional adapter interface prepared for future teacher model integration (not included).

2. **Export Formats**: Documentation provided for GGUF/ONNX export but implementation requires additional tools.

3. **Advanced Quantization**: Notes provided but actual quantization requires external tools like llama.cpp.

4. **Curriculum Learning**: Basic framework provided; advanced strategies can be added.

5. **Evaluation Metrics**: Basic perplexity and bilingual tests; comprehensive benchmarks can be added.
