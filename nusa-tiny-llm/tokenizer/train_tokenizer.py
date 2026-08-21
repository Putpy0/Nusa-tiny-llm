"""
Train Tokenizer Script

Script to train the custom BPE tokenizer on synthetic data.
"""

import argparse
import json
from pathlib import Path
from .tokenizer import NusaTokenizer


def load_texts_from_jsonl(jsonl_path: str) -> list:
    """Load texts from JSONL file."""
    texts = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Extract prompt and response
            if 'prompt' in data:
                texts.append(data['prompt'])
            if 'response' in data:
                texts.append(data['response'])
    
    return texts


def train_tokenizer_from_file(
    input_path: str,
    output_path: str,
    vocab_size: int = 24576,
    show_progress: bool = True
):
    """
    Train tokenizer from JSONL file.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to save vocabulary
        vocab_size: Target vocabulary size
        show_progress: Show training progress
    """
    print(f"Loading texts from {input_path}...")
    texts = load_texts_from_jsonl(input_path)
    print(f"Loaded {len(texts)} text samples")
    
    # Create and train tokenizer
    print(f"Training tokenizer with vocab_size={vocab_size}...")
    tokenizer = NusaTokenizer(vocab_size=vocab_size)
    tokenizer.train(texts, show_progress=show_progress)
    
    # Save
    print(f"Saving tokenizer to {output_path}...")
    tokenizer.save(output_path)
    
    print(f"Tokenizer trained successfully!")
    print(f"Vocabulary size: {tokenizer.vocab_size()}")
    
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description='Train BPE tokenizer')
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSONL file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output vocabulary file')
    parser.add_argument('--vocab-size', type=int, default=24576,
                        help='Vocabulary size (default: 24576)')
    parser.add_argument('--no-progress', action='store_true',
                        help='Hide progress output')
    
    args = parser.parse_args()
    
    train_tokenizer_from_file(
        input_path=args.input,
        output_path=args.output,
        vocab_size=args.vocab_size,
        show_progress=not args.no_progress
    )


if __name__ == '__main__':
    main()
