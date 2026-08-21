# Evaluation Documentation

## Overview

This document describes the evaluation framework for Nusa Tiny LLM, including bilingual tests and perplexity measurement.

## Evaluation Types

### 1. Perplexity

Measures how well the model predicts the test data:

```
Perplexity = exp(-1/N × ∑ log P(xᵢ | x₁...xᵢ₋₁))
```

**Lower is better.**

### 2. Bilingual Tests

Task-specific evaluation in English and Indonesian.

### 3. Generation Quality

Qualitative assessment of generated text.

## Bilingual Test Suite

### Test Categories

The test suite (`evaluation/bilingual_tests.json`) includes:

1. **Translation** (EN→ID, ID→EN)
2. **Question Answering**
3. **Instruction Following**
4. **Conversation**
5. **Reasoning**

### Test Format

```json
{
  "id": "test_001",
  "category": "translation",
  "language": "en",
  "prompt": "Translate to Indonesian: Good morning",
  "expected_keywords": ["selamat", "pagi"],
  "max_length": 50
}
```

### Running Evaluation

```bash
python -m evaluation.eval \
  --model checkpoints/model.pt \
  --tests evaluation/bilingual_tests.json \
  --output results/eval_results.json
```

## Perplexity Calculation

### Method

```python
def calculate_perplexity(model, dataloader):
    total_loss = 0
    total_tokens = 0
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch['input_ids'])
            loss = criterion(outputs, batch['labels'])
            
            total_loss += loss.item() * batch['num_tokens']
            total_tokens += batch['num_tokens']
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity
```

### Interpretation

| Perplexity | Quality Level |
|------------|---------------|
| < 10 | Excellent |
| 10-20 | Good |
| 20-50 | Moderate |
| 50-100 | Basic |
| > 100 | Poor |

**Note**: Expected perplexity for 0.5B model after training: 20-50

## Evaluation Metrics

### Automatic Metrics

1. **Exact Match**: Response matches expected exactly
2. **Keyword Match**: Response contains expected keywords
3. **BLEU** (Future): Translation quality
4. **ROUGE** (Future): Summary quality

### Manual Evaluation

For qualitative assessment:
- Fluency
- Coherence
- Relevance
- Language correctness

## Test Examples

### Translation Tests

```json
{
  "id": "trans_en_id_001",
  "category": "translation",
  "language": "en",
  "prompt": "Translate to Indonesian: How are you?",
  "expected_keywords": ["apa", "kabar"],
  "acceptable_answers": [
    "Apa kabar?",
    "Bagaimana kabarmu?"
  ]
}
```

### QA Tests

```json
{
  "id": "qa_id_001",
  "category": "qa",
  "language": "id",
  "prompt": "Apa ibukota Indonesia?",
  "expected_keywords": ["jakarta"],
  "acceptable_answers": [
    "Jakarta",
    "Ibukota Indonesia adalah Jakarta"
  ]
}
```

### Instruction Tests

```json
{
  "id": "instr_en_001",
  "category": "instruction",
  "language": "en",
  "prompt": "Write a greeting message",
  "expected_keywords": ["hello", "hi", "welcome"],
  "min_length": 5,
  "max_length": 50
}
```

## Evaluation Pipeline

### Step 1: Prepare Test Data

```python
from evaluation import load_tests

tests = load_tests('evaluation/bilingual_tests.json')
print(f"Loaded {len(tests)} tests")
```

### Step 2: Run Inference

```python
from evaluation import run_evaluation

results = run_evaluation(
    model=model,
    tokenizer=tokenizer,
    tests=tests,
    config=inference_config
)
```

### Step 3: Calculate Metrics

```python
from evaluation import calculate_metrics

metrics = calculate_metrics(results)
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"Keyword Match: {metrics['keyword_match']:.2%}")
```

### Step 4: Generate Report

```python
from evaluation import generate_report

report = generate_report(results, metrics)
report.save('results/evaluation_report.md')
```

## Language-Specific Evaluation

### English Tests

Focus on:
- Grammar correctness
- Natural phrasing
- Cultural appropriateness

### Indonesian Tests

Focus on:
- Proper word order (SPOK)
- Affix usage (me-, ber-, di-)
- Formal vs informal register

### Code-Switching Tests

Mixed language scenarios:
```
Prompt: "Saya ingin translate this sentence to Indonesian"
Expected: Handles mixed EN-ID naturally
```

## Benchmark Comparison (Future)

### Standard Benchmarks

When model is trained, consider:
- **MMLU**: General knowledge
- **IndoMMLU**: Indonesian knowledge
- **XNLI**: Cross-lingual inference

### Custom Benchmarks

Create domain-specific tests:
- Customer service dialogues
- Educational content
- Technical Q&A

## Regression Testing

### Baseline Preservation

After model updates:
```bash
# Run full test suite
python -m evaluation.eval --full-suite

# Compare with baseline
python scripts/compare_results.py \
  --old results/baseline.json \
  --new results/current.json
```

### Alert Thresholds

Flag if metrics drop by more than:
- Accuracy: 2%
- Perplexity: 5%
- Any critical test fails

## Reporting

### Results Format

```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "model_version": "0.1.0",
  "overall": {
    "accuracy": 0.75,
    "perplexity": 35.2
  },
  "by_category": {
    "translation": {"accuracy": 0.80},
    "qa": {"accuracy": 0.70},
    "instruction": {"accuracy": 0.75}
  },
  "by_language": {
    "en": {"accuracy": 0.76},
    "id": {"accuracy": 0.74}
  }
}
```

### Visualization

Generate charts for:
- Accuracy by category
- Progress over time
- Language comparison
