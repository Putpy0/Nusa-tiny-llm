# Tokenizer Documentation

## Overview

Nusa Tiny LLM uses a custom byte-level Byte Pair Encoding (BPE) tokenizer trained from scratch on bilingual English-Indonesian synthetic data. This document explains the tokenizer design, training process, and usage.

## Why Custom Tokenizer?

### Restrictions Compliance

1. **No HuggingFace**: Cannot use transformers.tokenization or sentencepiece from HF
2. **No Pretrained Vocab**: Must build vocabulary from scratch
3. **No External Dependencies**: Minimal library dependencies
4. **Full Control**: Complete ownership of tokenization pipeline

### Bilingual Requirements

1. **English Support**: Standard Latin alphabet, common words
2. **Indonesian Support**: Latin alphabet with specific patterns, affixes
3. **Code-Switching**: Handle mixed EN-ID text naturally
4. **Special Characters**: Support both languages' punctuation and symbols

## Tokenizer Type: Byte-Level BPE

### Design Choice

Byte-level BPE offers several advantages:

1. **No OOV Tokens**: Any byte sequence can be tokenized
2. **Compact Vocabulary**: Efficient representation of rare characters
3. **Unicode Safe**: Handles all Unicode characters through byte fallback
4. **Reversible**: Lossless encoding-decoding roundtrip

### How It Works

**Step 1: Byte Conversion**
- Input text → UTF-8 bytes
- Each byte represented as special token
- Example: "café" → [99, 97, 102, 195, 169]

**Step 2: Base Vocabulary**
- All 256 byte values as initial tokens
- Plus special tokens (<pad>, <bos>, <eos>, etc.)

**Step 3: BPE Merges**
- Iteratively merge most frequent adjacent pairs
- Continue until target vocabulary size reached
- Record merge rules for application

**Step 4: Word Pieces**
- Common substrings become single tokens
- Example: "ing", "tion", "ber", "me-" 

## Vocabulary Composition

### Target Size: 24,576 tokens

**Breakdown:**
- Special tokens: 7 tokens
- Byte tokens: 256 tokens (fallback)
- Character tokens: ~100 tokens
- Subword tokens: ~10,000 tokens
- Word tokens: ~14,000 tokens

### Language Distribution

Designed for balanced EN-ID support:

| Category | English | Indonesian | Shared |
|----------|---------|------------|--------|
| Common words | 3,000 | 3,000 | 2,000 |
| Affixes/prefixes | 500 | 800 | 200 |
| Technical terms | 2,000 | 1,500 | 1,000 |
| Names/entities | 1,500 | 1,500 | 500 |
| Function words | 500 | 500 | 200 |

## Special Tokens

### Defined Tokens

```json
{
  "<pad>": 0,      // Padding token
  "<bos>": 1,      // Beginning of sequence
  "<eos>": 2,      // End of sequence
  "<unk>": 3,      // Unknown (should not occur with byte fallback)
  "<sep>": 4,      // Separator for conversations
  "<user>": 5,     // User role indicator
  "<assistant>": 6 // Assistant role indicator
}
```

### Usage Patterns

**Single Turn:**
```
<bos> Hello, how are you? <eos>
```

**Multi-turn Conversation:**
```
<bos><user> What is AI? <sep><assistant> AI stands for... <eos>
```

**Bilingual:**
```
<bos> Translate: Hello = <sep> Halo <eos>
```

## Training Process

### Data Requirements

**Minimum:**
- 100,000 sentences (balanced EN-ID)
- Diverse domains (news, conversation, technical)
- Natural language patterns

**Recommended:**
- 1,000,000+ sentences
- Synthetic + curated data
- Multiple registers (formal, informal)

### Training Steps

1. **Preprocessing**
   - Normalize whitespace
   - Handle punctuation consistently
   - Preserve case (important for both languages)

2. **Initial Vocabulary**
   - Add all special tokens
   - Add all byte values (0-255)
   - Count character frequencies

3. **BPE Iterations**
   - Count adjacent token pairs
   - Merge most frequent pair
   - Update counts
   - Repeat until vocab_size reached

4. **Validation**
   - Test encode-decode roundtrip
   - Check vocabulary coverage
   - Verify special token IDs

### Training Command

```bash
# Generate synthetic data first
python scripts/generate_synthetic_data.py --mode mock --output data/generated/train.jsonl --num-samples 10000

# Train tokenizer
python scripts/train_tokenizer.py \
  --input data/generated/train.jsonl \
  --output tokenizer/vocab.json \
  --vocab-size 24576 \
  --show-progress
```

## Usage Examples

### Basic Encoding/Decoding

```python
from tokenizer import NusaTokenizer

# Load tokenizer
tokenizer = NusaTokenizer.from_pretrained("tokenizer/vocab.json")

# Encode text
text_en = "Hello, world!"
tokens_en = tokenizer.encode(text_en)
# Output: [1, 1234, 567, 890, 2]  # includes <bos>, <eos>

# Decode tokens
decoded_en = tokenizer.decode(tokens_en)
# Output: "Hello, world!"

# Indonesian text
text_id = "Halo, apa kabar?"
tokens_id = tokenizer.encode(text_id)
decoded_id = tokenizer.decode(tokens_id)
```

### Batch Processing

```python
texts = [
    "Good morning!",
    "Selamat pagi!",
    "How are you today?",
    "Apa kabar hari ini?"
]

batch_tokens = tokenizer.encode_batch(texts)
# Returns list of token lists
```

### Special Token Handling

```python
# With special tokens
tokens = tokenizer.encode(
    "Hello",
    add_special_tokens=True  # Adds <bos> and <eos>
)

# Without special tokens
tokens = tokenizer.encode(
    "Hello",
    add_special_tokens=False
)
```

## Implementation Details

### File Structure

```
tokenizer/
├── __init__.py          # Package initialization
├── bpe.py              # BPE algorithm implementation
├── tokenizer.py        # Main tokenizer class
├── train_tokenizer.py  # Training script
└── special_tokens.json # Special token definitions
```

### Key Classes

**BPETrainer:**
- Implements BPE training algorithm
- Manages vocabulary construction
- Handles merge rule generation

**NusaTokenizer:**
- Main tokenizer interface
- encode() / decode() methods
- Special token handling
- Byte fallback support

### Performance Considerations

1. **Caching**: Cache encoded results for repeated texts
2. **Batching**: Process multiple texts together when possible
3. **Pre-tokenization**: Split on spaces before BPE for efficiency
4. **Lookup Tables**: Use hash maps for O(1) token lookup

## Testing

### Roundtrip Test

```python
def test_roundtrip():
    texts = [
        "Hello world",
        "Halo dunia",
        "Test 123",
        "Special chars: @#$%",
        "Mixed: Hello dunia"
    ]
    
    for text in texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text, f"Failed for: {text}"
```

### Edge Cases

Test these scenarios:
1. Empty string
2. Only whitespace
3. Only special characters
4. Very long text (>1024 chars)
5. Mixed languages
6. Numbers and symbols
7. Emojis (handled via bytes)

## Troubleshooting

### Common Issues

**Issue: Unknown tokens in output**
- Solution: Should not occur with byte fallback; check special token config

**Issue: Vocabulary too small**
- Solution: Increase vocab_size or training data

**Issue: Slow encoding**
- Solution: Enable caching, check pre-tokenization

**Issue: Poor Indonesian coverage**
- Solution: Add more Indonesian training data

### Debugging Tips

```python
# Inspect vocabulary
print(f"Vocab size: {len(tokenizer.vocab)}")
print(f"Special tokens: {tokenizer.special_tokens}")

# Check token for specific text
token = tokenizer.get_token("hello")
print(f"Token ID for 'hello': {token}")

# List most common tokens
common = tokenizer.get_most_common(20)
print(common)
```

## Future Improvements

1. **Unigram LM**: Consider unigram language model for better subword segmentation
2. **Language Tags**: Optional language identification tokens
3. **Domain Adaptation**: Fine-tune vocabulary for specific domains
4. **Compression**: Optimize vocabulary storage format

## References

- Original BPE paper: Sennrich et al. (2016)
- Byte-level BPE: Radford et al. (2019) - GPT-2
- SentencePiece: Kudo & Richardson (2018)

Note: Implementation is from scratch without using sentencepiece or transformers libraries.
