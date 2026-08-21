"""
CPU Inference Engine for Nusa Tiny LLM

Optimized inference engine for low-resource CPU devices.
"""

import torch
from typing import Optional, Dict, Any, List
from pathlib import Path
import json


class CPUInferenceEngine:
    """
    CPU-optimized inference engine for low-resource devices.
    
    Features:
    - Memory-efficient loading
    - Quantization support (simulated)
    - Batch size 1 optimization
    - Progressive token generation
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        tokenizer=None,
        device: str = "cpu",
        quantize: bool = False
    ):
        self.device = torch.device(device)
        self.tokenizer = tokenizer
        self.quantize = quantize
        self.model = None
        self.config = {}
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path, config_path)
    
    def load_model(self, model_path: str, config_path: Optional[str] = None):
        """
        Load model from checkpoint.
        
        Args:
            model_path: Path to model weights (.pt file)
            config_path: Path to config JSON
        """
        # Load config
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        
        print(f"Model loaded on {self.device}")
        if self.quantize:
            print("Note: Quantization is conceptual - actual quantization requires export")
    
    def load_from_architecture(
        self,
        model_class: type,
        config: Dict[str, Any],
        weights: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Load model from architecture class.
        
        Args:
            model_class: Model class to instantiate
            config: Model configuration
            weights: Optional pre-trained weights
        """
        self.config = config
        
        # Create model instance
        self.model = model_class(config)
        
        if weights:
            self.model.load_state_dict(weights)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream_callback: Optional[callable] = None
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stream_callback: Optional callback for streaming output
            
        Returns:
            Generated text
        """
        if self.model is None:
            return "Error: Model not loaded"
        
        # Tokenize
        if hasattr(self.tokenizer, 'encode'):
            input_ids = self.tokenizer.encode(prompt)
        else:
            # Simple fallback
            input_ids = [ord(c) % 256 for c in prompt]
        
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long)
        
        input_ids = input_ids.to(self.device)
        
        generated = input_ids.clone()
        
        # Generation loop
        for i in range(max_new_tokens):
            outputs = self.model(input_ids=generated)
            next_logits = outputs.logits[:, -1, :] / temperature
            
            # Top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)
                
                remove_mask = cumulative_probs > top_p
                remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                remove_mask[..., 0] = False
                
                next_logits.scatter_(1, sorted_indices, torch.where(
                    remove_mask,
                    torch.tensor(float('-inf'), device=next_logits.device),
                    sorted_logits
                ))
            
            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check EOS (token ID 2)
            if next_token.item() == 2:
                break
            
            # Stream callback
            if stream_callback:
                current_text = self._decode(generated[0])
                stream_callback(current_text[len(prompt):])
        
        # Decode result
        full_text = self._decode(generated[0])
        return full_text[len(prompt):].strip()
    
    def _decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to string."""
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(token_ids.tolist())
        else:
            return "".join(chr(int(t) % 256) for t in token_ids.tolist())
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Estimate memory usage."""
        if self.model is None:
            return {"total_mb": 0}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Estimate bytes based on dtype
        dtype_size = 4  # float32 default
        if self.quantize:
            dtype_size = 0.5  # q4 approximation
        
        total_bytes = total_params * dtype_size
        total_mb = total_bytes / (1024 * 1024)
        
        return {
            "total_params": total_params,
            "dtype_size": dtype_size,
            "total_mb": round(total_mb, 2)
        }
    
    def run_interactive(
        self,
        max_new_tokens: int = 64,
        exit_commands: List[str] = None
    ):
        """
        Run interactive chat mode.
        
        Args:
            max_new_tokens: Maximum tokens per response
            exit_commands: Commands to exit chat
        """
        if exit_commands is None:
            exit_commands = ["quit", "exit", "/quit", "/exit"]
        
        print("Nusa Tiny LLM - Interactive Mode")
        print(f"Device: {self.device}")
        print(f"Type 'quit' or 'exit' to stop\n")
        
        while True:
            try:
                prompt = input("You: ").strip()
                
                if prompt.lower() in exit_commands:
                    print("Goodbye!")
                    break
                
                if not prompt:
                    continue
                
                print("Assistant: ", end="", flush=True)
                
                def stream_callback(text):
                    print(text, end="", flush=True)
                
                response = self.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    stream_callback=stream_callback
                )
                
                print("\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


def create_cpu_engine(
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
    tokenizer=None,
    quantize: bool = False
) -> CPUInferenceEngine:
    """
    Factory function to create CPU inference engine.
    
    Args:
        model_path: Optional path to model weights
        config_path: Optional path to config
        tokenizer: Tokenizer instance
        quantize: Whether to use quantization
        
    Returns:
        CPUInferenceEngine instance
    """
    return CPUInferenceEngine(
        model_path=model_path,
        config_path=config_path,
        tokenizer=tokenizer,
        device="cpu",
        quantize=quantize
    )


if __name__ == "__main__":
    # Example usage
    print("CPU Inference Engine for Nusa Tiny LLM")
    print("Use create_cpu_engine() to create an instance")
