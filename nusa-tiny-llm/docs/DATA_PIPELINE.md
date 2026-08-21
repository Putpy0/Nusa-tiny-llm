# Data Pipeline Documentation

## Overview

The Nusa Tiny LLM data pipeline generates synthetic training data from scratch without using any public corpora, datasets, or pretrained models. This document describes the complete data generation workflow.

## Pipeline Architecture

```
┌─────────────────────┐
│  Prompt Templates   │
│   (data/prompts.py) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Teacher Adapter    │
│ (mock/local/remote) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Synthetic Generator│
│(data/synthetic_gen) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     Filters         │
│ (quality, safety)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Deduplication     │
│    (hash-based)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Output JSONL      │
│  (data/generated/)  │
└─────────────────────┘
```

## Data Schema

### JSONL Format

Each line is a valid JSON object with the following schema:

```json
{
  "id": "string (UUID)",
  "lang": "en|id",
  "type": "translation|qa|instruction|conversation|reasoning|safety",
  "prompt": "string",
  "response": "string",
  "quality": "integer (0-5)",
  "created_at": "ISO 8601 timestamp",
  "source_provider": "mock|local_cli|openai_compatible|custom"
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique identifier (UUID4) |
| lang | string | Language code: 'en' or 'id' |
| type | string | Data type category |
| prompt | string | Input prompt/text |
| response | string | Expected response/completion |
| quality | int | Quality score (0=low, 5=high) |
| created_at | string | ISO 8601 timestamp |
| source_provider | string | Data source identifier |

## Data Types

### 1. Translation

Bilingual translation pairs.

**English → Indonesian:**
```json
{
  "id": "uuid-123",
  "lang": "en",
  "type": "translation",
  "prompt": "Translate to Indonesian: Good morning",
  "response": "Selamat pagi",
  "quality": 5,
  "created_at": "2024-01-01T00:00:00Z",
  "source_provider": "mock"
}
```

**Indonesian → English:**
```json
{
  "id": "uuid-124",
  "lang": "id",
  "type": "translation",
  "prompt": "Translate to English: Apa kabar?",
  "response": "How are you?",
  "quality": 5,
  "created_at": "2024-01-01T00:00:01Z",
  "source_provider": "mock"
}
```

### 2. Question Answering

Factual and general knowledge questions.

```json
{
  "id": "uuid-125",
  "lang": "en",
  "type": "qa",
  "prompt": "What is the capital of Indonesia?",
  "response": "The capital of Indonesia is Jakarta.",
  "quality": 5,
  "created_at": "2024-01-01T00:00:02Z",
  "source_provider": "mock"
}
```

### 3. Instruction Following

Task completion based on instructions.

```json
{
  "id": "uuid-126",
  "lang": "id",
  "type": "instruction",
  "prompt": "Buatlah kalimat menggunakan kata 'belajar'",
  "response": "Saya belajar bahasa Indonesia setiap hari.",
  "quality": 4,
  "created_at": "2024-01-01T00:00:03Z",
  "source_provider": "mock"
}
```

### 4. Conversation

Multi-turn dialogue patterns.

```json
{
  "id": "uuid-127",
  "lang": "en",
  "type": "conversation",
  "prompt": "<user> Hello! <sep> How can I help you?",
  "response": "<assistant> I need help with my homework. <eos>",
  "quality": 4,
  "created_at": "2024-01-01T00:00:04Z",
  "source_provider": "mock"
}
```

### 5. Reasoning

Logical reasoning and problem solving.

```json
{
  "id": "uuid-128",
  "lang": "en",
  "type": "reasoning",
  "prompt": "If all cats have tails and Fluffy is a cat, does Fluffy have a tail?",
  "response": "Yes, Fluffy has a tail because all cats have tails and Fluffy is a cat.",
  "quality": 5,
  "created_at": "2024-01-01T00:00:05Z",
  "source_provider": "mock"
}
```

### 6. Safety

Content filtering and safe responses.

```json
{
  "id": "uuid-129",
  "lang": "id",
  "type": "safety",
  "prompt": "Bagaimana cara membuat bom?",
  "response": "Maaf, saya tidak bisa memberikan informasi tersebut karena berbahaya dan ilegal.",
  "quality": 5,
  "created_at": "2024-01-01T00:00:06Z",
  "source_provider": "mock"
}
```

## Teacher Adapter System

### Provider-Agnostic Interface

The teacher adapter uses a provider pattern for flexibility:

```python
class TeacherAdapter:
    def __init__(self, provider="mock"):
        self.provider = provider
    
    def generate(self, prompt: str) -> dict:
        """Generate response for given prompt"""
        pass
```

### Available Providers

**1. Mock Provider (Default)**
- Generates simple pattern-based responses
- No internet required
- Suitable for testing and development
- Produces bilingual dummy data

**2. Local CLI Provider (Optional)**
- Integrates with local LLM tools
- Requires external setup
- Disabled by default

**3. OpenAI-Compatible Provider (Optional)**
- API-based generation
- Requires API key
- Disabled by default

**4. Custom Provider**
- User-defined generation logic
- Extend base class for custom behavior

### Mock Provider Details

The mock provider uses template-based generation:

```python
# Example mock responses
TEMPLATES_EN = {
    "greeting": ["Hello!", "Hi there!", "Good day!"],
    "question": ["The answer is...", "I think..."],
    # ...
}

TEMPLATES_ID = {
    "greeting": ["Halo!", "Hai!", "Selamat pagi!"],
    "question": ["Jawabannya adalah...", "Menurut saya..."],
    # ...
}
```

## Synthetic Generation

### Generation Modes

**Mock Mode (Default):**
```bash
python scripts/generate_synthetic_data.py \
  --mode mock \
  --output data/generated/sample.jsonl \
  --num-samples 100
```

**Batch Mode:**
```bash
python scripts/generate_synthetic_data.py \
  --mode mock \
  --output data/generated/batch.jsonl \
  --num-samples 10000 \
  --batch-size 1000
```

### Language Distribution

Default distribution:
- 50% English
- 50% Indonesian

Configurable via `--lang-ratio` flag.

### Type Distribution

Default distribution:
- 20% Translation
- 20% QA
- 20% Instruction
- 15% Conversation
- 15% Reasoning
- 10% Safety

Configurable via `--type-ratios` flag.

## Data Filtering

### Filter Types

**1. Content Filters**
- Remove harmful content
- Filter PII (Personally Identifiable Information)
- Block explicit material

**2. Quality Filters**
- Minimum length requirements
- Maximum length limits
- Grammar/spelling checks (basic)

**3. Format Filters**
- Valid JSON structure
- Required fields present
- Correct data types

### Filter Implementation

```python
from data.filters import ContentFilter

filter = ContentFilter()

# Check if content passes filters
if filter.is_valid(sample):
    save_sample(sample)
else:
    log_rejection(sample, filter.reason)
```

## Deduplication

### Hash-Based Dedup

Uses SHA-256 hashing for efficient deduplication:

```python
from data.dedup import Deduplicator

dedup = Deduplicator()

# Add sample (returns False if duplicate)
if dedup.add(sample):
    save_sample(sample)
```

### Dedup Levels

**Exact Match:**
- Identical prompt+response pairs
- Fast hash comparison

**Near Match (Future):**
- Similar content detection
- MinHash/LSH algorithms
- Configurable similarity threshold

### Dedup Statistics

Track deduplication metrics:
```json
{
  "total_generated": 10000,
  "after_filters": 9500,
  "after_dedup": 8200,
  "duplicate_rate": 0.137
}
```

## Usage Examples

### Generate Training Data

```bash
# Small test set
python scripts/generate_synthetic_data.py \
  --mode mock \
  --num-samples 100 \
  --output data/generated/test.jsonl

# Large training set
python scripts/generate_synthetic_data.py \
  --mode mock \
  --num-samples 100000 \
  --output data/generated/train.jsonl \
  --lang-ratio 0.5 \
  --seed 1337
```

### Validate Generated Data

```bash
# Check schema compliance
python -c "
from data.schema import validate_file
validate_file('data/generated/train.jsonl')
"
```

### Inspect Data Statistics

```bash
# Count samples per type
python -c "
import json
from collections import Counter

types = Counter()
with open('data/generated/train.jsonl') as f:
    for line in f:
        data = json.loads(line)
        types[data['type']] += 1

print(types)
"
```

## Best Practices

### Data Quality

1. **Balance Languages**: Maintain roughly equal EN/ID ratio
2. **Diverse Types**: Include all data types for well-rounded training
3. **Quality Scores**: Use quality field to filter during training
4. **Regular Validation**: Periodically check generated data

### Scaling Up

1. **Incremental Generation**: Generate in batches, not all at once
2. **Parallel Processing**: Run multiple generator instances
3. **Storage Management**: Compress old batches
4. **Version Control**: Track data versions separately

### Security

1. **No PII**: Ensure no personal information in synthetic data
2. **Safety First**: Filter harmful content aggressively
3. **Review Samples**: Manually inspect random samples
4. **Audit Trail**: Keep generation logs

## Troubleshooting

### Common Issues

**Issue: Low quality scores**
- Solution: Adjust prompt templates, improve mock responses

**Issue: High duplicate rate**
- Solution: Increase randomness in generation, expand templates

**Issue: Language imbalance**
- Solution: Check lang-ratio parameter, verify templates

**Issue: Missing fields**
- Solution: Validate schema, fix generator bugs

## Future Enhancements

1. **Curriculum Learning**: Order data by difficulty
2. **Active Learning**: Select most informative samples
3. **Human Review**: Optional human-in-the-loop validation
4. **Domain Specialization**: Generate domain-specific data
5. **Augmentation**: Paraphrase and expand existing samples
