"""
Text Generation for Nusa Tiny LLM

Implements text generation with various decoding strategies.
"""

import torch
from typing import Optional, List, Dict, Any, Tuple
from .kv_cache import KVCache


class TextGenerator:
    """
    Text generator with multiple decoding strategies.
    
    Supports:
    - Greedy decoding
    - Temperature sampling
    - Top-k sampling
    - Top-p (nucleus) sampling
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        config: Dict[str, Any] = None,
        device: str = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Get config values
        self.vocab_size = config.get("vocab_size", 24576) if config else 24576
        
        # Special token IDs (assumed)
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        use_kv_cache: bool = True,
        stop_strings: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Generate text from input tokens.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repeating tokens
            use_kv_cache: Whether to use KV caching
            stop_strings: Strings that trigger early stopping
            
        Returns:
            Generated token IDs
        """
        self.model.eval()
        
        # Initialize
        generated = input_ids.clone()
        seq_length = generated.shape[-1]
        
        # Create KV cache if enabled
        kv_cache = None
        if use_kv_cache:
            num_layers = getattr(self.model.config, 'num_hidden_layers', 24) if hasattr(self.model, 'config') else 24
            num_kv_heads = getattr(self.model.config, 'num_key_value_heads', 4) if hasattr(self.model, 'config') else 4
            head_dim = getattr(self.model.config, 'head_dim', 64) if hasattr(self.model, 'config') else 64
            
            kv_cache = KVCache(
                num_layers=num_layers,
                batch_size=1,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_seq_len=seq_length + max_new_tokens,
                device=self.device.type
            )
        
        # Track position for KV cache
        past_length = seq_length
        
        for i in range(max_new_tokens):
            # Prepare inputs
            if use_kv_cache and i > 0:
                # Only process the last token
                current_input = generated[:, -1:]
            else:
                current_input = generated
            
            # Forward pass
            outputs = self.model(input_ids=current_input)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    if next_token_logits[0, token_id] < 0:
                        next_token_logits[0, token_id] *= repetition_penalty
                    else:
                        next_token_logits[0, token_id] /= repetition_penalty
            
            # Apply top-k
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if next_token.item() == self.eos_token_id:
                break
            
            # Check stop strings
            if stop_strings:
                current_text = self._decode_partial(generated[0])
                if any(stop_str in current_text for stop_str in stop_strings):
                    break
        
        return generated
    
    def _decode_partial(self, token_ids: torch.Tensor) -> str:
        """Decode tokens to string."""
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(token_ids.tolist())
        else:
            return "".join(chr(int(t) % 256) for t in token_ids.tolist())
    
    @torch.no_grad()
    def generate_greedy(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128
    ) -> torch.Tensor:
        """Greedy decoding (always pick most likely token)."""
        return self.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=1,
            top_p=1.0
        )
    
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        **kwargs
    ) -> str:
        """
        Generate text from a string prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum new tokens
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Tokenize prompt
        if hasattr(self.tokenizer, 'encode'):
            input_ids = self.tokenizer.encode(prompt)
        else:
            input_ids = [ord(c) % 256 for c in prompt]
        
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long)
        
        input_ids = input_ids.to(self.device)
        
        # Generate
        output_ids = self.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        # Decode
        if hasattr(self.tokenizer, 'decode'):
            full_text = self.tokenizer.decode(output_ids[0].tolist())
        else:
            full_text = "".join(chr(int(t) % 256) for t in output_ids[0].tolist())
        
        # Return only generated part
        return full_text[len(prompt):].strip()


def generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    config: Dict[str, Any] = None,
    **kwargs
) -> str:
    """
    Convenience function for text generation.
    
    Args:
        model: Model to use
        tokenizer: Tokenizer to use
        prompt: Input prompt
        config: Model configuration
        **kwargs: Generation parameters
        
    Returns:
        Generated text
    """
    generator = TextGenerator(model, tokenizer, config)
    return generator.generate_text(prompt, **kwargs)
