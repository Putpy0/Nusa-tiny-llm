# Training Documentation

## Overview

This document describes the training pipeline for Nusa Tiny LLM. **Important**: Actual training of the 0.5B model requires GPU/Colab/server hardware and is NOT performed on low-resource devices. This repository provides the complete training infrastructure.

## Training Objective

**Causal Language Modeling (CLM)**

The model is trained to predict the next token in a sequence:

```
P(x₁, x₂, ..., xₙ) = ∏ P(xᵢ | x₁, ..., xᵢ₋₁)
```

**Loss Function:**
```
Loss = -∑ log P(xᵢ | x₁, ..., xᵢ₋₁)
```

## Training Configuration

### Key Hyperparameters

See `configs/training_config.json`:

```json
{
  "batch_size": 64,
  "gradient_accumulation_steps": 8,
  "learning_rate": 3e-4,
  "weight_decay": 0.1,
  "warmup_steps": 1000,
  "max_steps": 100000,
  "precision": "bf16",
  "gradient_clipping": 1.0
}
```

### Effective Batch Size

```
effective_batch = batch_size × gradient_accumulation_steps × seq_len
                = 64 × 8 × 1024
                = 524,288 tokens
```

## Learning Rate Schedule

### Cosine Decay with Warmup

```python
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)
else:
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(π * progress))
```

**Schedule Parameters:**
- Warmup: 1,000 steps (linear increase)
- Decay: Cosine from step 1,000 to 100,000
- Min LR: 10% of base LR

## Dataset Preparation

### Data Format

Training data in JSONL format:
```jsonl
{"id": "...", "lang": "en", "prompt": "...", "response": "..."}
{"id": "...", "lang": "id", "prompt": "...", "response": "..."}
```

### Tokenization Pipeline

1. **Load JSONL**: Read synthetic data
2. **Format**: Combine prompt + response
3. **Tokenize**: Convert to token IDs
4. **Pack**: Create sequences of fixed length
5. **Batch**: Group into batches

### Sequence Packing

```python
# Concatenate all tokens
all_tokens = concat(sample_tokens)

# Chunk into fixed-length sequences
sequences = []
for i in range(0, len(all_tokens), seq_length):
    seq = all_tokens[i:i+seq_length]
    if len(seq) == seq_length:
        sequences.append(seq)
```

## Curriculum Learning

### Progressive Sequence Length

Start with shorter sequences, gradually increase:

```python
curriculum = [
    (steps_0_to_10k, seq_len=128),
    (steps_10k_to_30k, seq_len=256),
    (steps_30k_to_60k, seq_len=512),
    (steps_60k_to_100k, seq_len=1024),
]
```

### Benefits

1. **Faster Early Training**: Shorter sequences = more updates
2. **Stable Initialization**: Learn basic patterns first
3. **Gradual Complexity**: Build up to long contexts

## Optimizer

### AdamW Configuration

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1
)
```

### Parameter Groups

Different weight decay for different parameters:

```python
# No decay for biases and norms
no_decay = ['bias', 'rmsnorm.weight']
optimizer_grouped = [
    {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.1},
    {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
```

## Gradient Clipping

### Clip by Norm

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Purpose:**
- Prevent exploding gradients
- Stabilize training
- Allow higher learning rates

## Mixed Precision Training

### BFloat16 Training

```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    loss = model(inputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits:**
- 2× memory reduction
- Faster matrix operations (Tensor Cores)
- Stable with BFloat16 (vs Float16)

### Fallback to Float32

If BFloat16 not available:
```python
dtype = torch.float32  # Safe fallback
```

## Checkpointing

### Save Checkpoint

```python
checkpoint = {
    'step': current_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': current_loss,
    'config': model_config
}
torch.save(checkpoint, f'checkpoints/step_{current_step}.pt')
```

### Load Checkpoint

```python
checkpoint = torch.load('checkpoints/step_50000.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_step = checkpoint['step']
```

### Checkpoint Strategy

- Save every 1,000 steps
- Keep last 3 checkpoints
- Save best model separately

## Logging & Monitoring

### Training Metrics

Log every 10 steps:
- Loss (training)
- Learning rate
- Gradient norm
- Tokens per second

### Evaluation Metrics

Evaluate every 500 steps:
- Validation loss
- Perplexity
- Sample generations

### Example Log Output

```
Step 1000/100000 | Loss: 4.523 | LR: 3.0e-4 | Grad: 0.82 | Tok/s: 12500
Step 2000/100000 | Loss: 4.201 | LR: 3.0e-4 | Grad: 0.75 | Tok/s: 12800
...
Eval @ 5000: Val Loss: 3.892 | Perplexity: 49.02
```

## Training Loop Structure

```python
def train(model, dataloader, optimizer, scheduler, config):
    model.train()
    
    for epoch in range(config.num_epochs):
        for batch in dataloader:
            # Forward pass
            with autocast():
                outputs = model(batch['input_ids'])
                loss = criterion(outputs, batch['labels'])
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), config.gradient_clipping)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            # Logging
            if step % config.logging_every == 0:
                log_metrics(step, loss, lr)
            
            # Checkpointing
            if step % config.checkpoint_every == 0:
                save_checkpoint(step)
            
            # Evaluation
            if step % config.eval_every == 0:
                evaluate(model, val_loader)
            
            step += 1
```

## Hardware Requirements

### Minimum for Training

- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **RAM**: 64GB system memory
- **Storage**: 500GB SSD
- **Time**: ~1-2 weeks for 100K steps

### Recommended

- **GPU**: 4× A100 (40GB) or equivalent
- **RAM**: 128GB+ system memory
- **Storage**: 1TB NVMe SSD
- **Time**: ~2-3 days for 100K steps

### Memory Estimation

For 0.5B model with batch size 64:

| Component | Memory |
|-----------|--------|
| Model weights (BF16) | 1 GB |
| Gradients (BF16) | 1 GB |
| Optimizer states (Adam) | 4 GB |
| Activations | 8 GB |
| Batches | 4 GB |
| **Total** | **~18 GB** |

With gradient accumulation (8 steps): **~3-4 GB** per micro-batch

## Distributed Training (Future)

### Data Parallel

```python
model = torch.nn.DataParallel(model)
# or
model = torch.nn.DistributedDataParallel(model)
```

### ZeRO Optimization (DeepSpeed)

For larger batch sizes:
- Partition optimizer states
- Partition gradients
- Partition parameters

## Resuming Training

### From Checkpoint

```bash
python training/trainer.py \
  --config configs/training_config.json \
  --resume checkpoints/step_50000.pt
```

### Fine-tuning

```bash
python training/trainer.py \
  --config configs/training_config.json \
  --pretrained checkpoints/final_model.pt \
  --learning-rate 1e-5
```

## Expected Training Progress

### Loss Curve (Estimated)

| Step | Train Loss | Val Loss | Perplexity |
|------|------------|----------|------------|
| 1K | 6.5 | 6.6 | 735 |
| 10K | 4.5 | 4.6 | 100 |
| 30K | 3.8 | 3.9 | 49 |
| 60K | 3.2 | 3.4 | 30 |
| 100K | 2.8 | 3.0 | 20 |

**Note**: Actual values depend on data quality and hyperparameters.

## Post-Training

### Model Export

```python
# Save final model
torch.save({
    'config': model_config,
    'state_dict': model.state_dict(),
    'tokenizer': tokenizer_config
}, 'nusa_tiny_llm_0.5b.pt')
```

### Conversion to GGUF

See `deployment/EXPORT_GGUF.md` for conversion to GGUF format for efficient inference.

## Troubleshooting

### Common Issues

**Issue: Loss not decreasing**
- Check learning rate (try lower)
- Verify data quality
- Check for NaN/Inf in inputs

**Issue: OOM during training**
- Reduce batch size
- Increase gradient accumulation
- Use activation checkpointing

**Issue: Unstable training**
- Lower learning rate
- Increase warmup steps
- Check gradient clipping

**Issue: Slow training**
- Enable mixed precision
- Optimize data loading
- Use multiple GPUs

## Safety Considerations

1. **Content Filtering**: Ensure training data is safe
2. **Bias Monitoring**: Check for harmful biases
3. **Access Control**: Restrict model access if needed
4. **Documentation**: Document known limitations
