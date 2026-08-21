# Nusa Tiny LLM

## Project Name
**Nusa Tiny LLM** - A small bilingual (English-Indonesian) language model with approximately 0.5B parameters.

## Project Goal
Build a complete LLM repository from scratch without using any pretrained models, datasets, or weights from HuggingFace or other public sources. The model is designed for low-resource inference on devices like smartphones (4GB RAM) and small VPS (2GB RAM).

## Project Status
**Alpha** - Repository contains complete architecture, training pipeline, synthetic data generation, and documentation. Actual training of the 0.5B model requires GPU/Colab/server hardware.

## Architecture Overview

- **Model Type**: Decoder-only Transformer
- **Target Parameters**: ~500M (max budget: 550M)
- **Hidden Size**: 1280
- **Layers**: 24
- **Attention Heads**: 20 (Q), 4 (KV) - Grouped Query Attention
- **Vocabulary Size**: 24,576
- **Context Length**: 1024 tokens
- **Normalization**: RMSNorm
- **Positional Encoding**: RoPE (Rotary Position Embedding)
- **Activation**: SwiGLU
- **Languages**: English & Indonesian (bilingual)

## Target Hardware

### Inference
- **Smartphone**: 4GB RAM with quantized model (q4_k_m)
- **VPS**: 2GB RAM, 1 core with heavily quantized model (q4_0)
- **Desktop**: CPU inference with moderate quantization

### Training (Not included in this repo execution)
- **Minimum**: GPU with 24GB VRAM
- **Recommended**: Multi-GPU setup or cloud training (Colab Pro, AWS, etc.)

## Important Restrictions

⚠️ **This repository strictly prohibits:**
1. Using HuggingFace datasets, models, or transformers library
2. Downloading pretrained weights from any source
3. Using public corpora or datasets
4. Internet dependency for core functionality
5. Teacher model downloads (adapter interface is optional and disabled by default)

All components are built from scratch including tokenizer, data generation, and model architecture.

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Smoke Test
Test the entire pipeline with a small configuration:
```bash
make smoke
```

Or run directly:
```bash
python scripts/run_smoke_test.py
```

### Generate Synthetic Data (Mock Mode)
Generate dummy bilingual data for testing:
```bash
make generate-data-mock
```

Or run directly:
```bash
python scripts/generate_synthetic_data.py --mode mock --output data/generated/sample.jsonl --num-samples 100
```

### Train Tokenizer (Mock Data)
Train the custom BPE tokenizer on synthetic data:
```bash
make train-tokenizer-mock
```

Or run directly:
```bash
python scripts/train_tokenizer.py --input data/generated/sample.jsonl --output tokenizer/vocab.json --vocab-size 512
```

### Estimate Parameters
Calculate parameter count for the model configuration:
```bash
make estimate-params
```

Or run directly:
```bash
python scripts/estimate_parameters.py --config configs/model_config.json
```

### Run Tests
```bash
make test
```

## Repository Structure

```
nusa-tiny-llm/
├── architecture/      # Model architecture components
├── tokenizer/         # Custom BPE tokenizer implementation
├── data/             # Synthetic data generation pipeline
├── training/         # Training loop and utilities
├── evaluation/       # Evaluation scripts and tests
├── inference/        # Inference engine and optimization
├── deployment/       # Deployment guides for low-resource
├── configs/          # Configuration files
├── docs/             # Documentation
├── tests/            # Test suite
└── scripts/          # Utility scripts
```

## Development Roadmap

### Phase 1: Foundation (Current)
- [x] Model architecture implementation
- [x] Custom tokenizer from scratch
- [x] Synthetic data pipeline
- [x] Training infrastructure
- [x] Basic inference engine
- [x] Documentation

### Phase 2: Training Preparation
- [ ] Large-scale synthetic data generation
- [ ] Data quality filtering and deduplication
- [ ] Curriculum learning strategy
- [ ] Distributed training setup

### Phase 3: Model Training
- [ ] Full 0.5B model training on GPU cluster
- [ ] Checkpoint management
- [ ] Training monitoring and logging
- [ ] Iterative hyperparameter tuning

### Phase 4: Optimization
- [ ] Quantization to GGUF format
- [ ] ONNX export for compatibility
- [ ] CPU inference optimization
- [ ] Mobile deployment testing

### Phase 5: Evaluation & Release
- [ ] Comprehensive bilingual evaluation
- [ ] Benchmark comparisons
- [ ] Model release with weights
- [ ] Community documentation

## License

MIT License - Copyright (c) 2024 Sultan

## Contributing

This is an open project for educational and research purposes. Contributions welcome for:
- Architecture improvements
- Synthetic data generation strategies
- Low-resource optimization
- Bilingual evaluation benchmarks

## Citation

If you use this model in your research:
```
@misc{nusa-tiny-llm,
  title={Nusa Tiny LLM: A Bilingual English-Indonesian Small Language Model},
  author={Sultan},
  year={2024},
  howpublished={\url{https://github.com/sultan/nusa-tiny-llm}}
}
```
