"""
Nusa Tiny LLM Model

Complete decoder-only transformer model implementation.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass

from .config import ModelConfig
from .embedding import TokenEmbedding
from .transformer_block import TransformerBlock, create_attention_mask
from .rmsnorm import RMSNorm


@dataclass
class ModelOutput:
    """Model output container."""
    logits: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    loss: Optional[torch.Tensor] = None


class NusaTinyLLM(nn.Module):
    """
    Nusa Tiny LLM - Decoder-only transformer for bilingual language modeling.
    
    Architecture:
        Embedding → [TransformerBlock × N] → RMSNorm → Output Projection
    
    Args:
        config (ModelConfig): Model configuration
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        
        # Token embedding
        self.embed_tokens = TokenEmbedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            tie_weights=config.tie_word_embeddings
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                intermediate_size=config.intermediate_size,
                max_position_embeddings=config.max_position_embeddings,
                rope_theta=config.rope_theta,
                dropout=config.dropout,
                use_bias=config.use_bias,
                layer_norm_epsilon=config.layer_norm_epsilon
            )
            for _ in range(config.num_hidden_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon
        )
        
        # Output projection (tied with embedding if configured)
        if config.tie_word_embeddings:
            # Weight tying handled by embedding layer
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False
            )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize module weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
    
    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embedding layer."""
        return self.embed_tokens.embedding
    
    def set_input_embeddings(self, value: nn.Embedding):
        """Set input embedding layer."""
        self.embed_tokens.embedding = value
    
    def get_output_embeddings(self):
        """Get output projection layer."""
        if self.lm_head is not None:
            return self.lm_head
        else:
            return self.embed_tokens.get_output_projection()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Union[ModelOutput, Tuple]:
        """
        Forward pass for Nusa Tiny LLM.
        
        Args:
            input_ids (torch.Tensor): Token IDs (batch, seq_len)
            attention_mask (torch.Tensor, optional): Attention mask
            position_ids (torch.Tensor, optional): Position IDs
            past_key_values (list, optional): Cached KV states for each layer
            labels (torch.Tensor, optional): Labels for loss computation
            use_cache (bool): Whether to return KV cache
            output_hidden_states (bool): Whether to return all hidden states
            return_dict (bool): Return ModelOutput vs tuple
            
        Returns:
            ModelOutput or tuple of (logits, hidden_states, past_key_values)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = next(self.parameters()).dtype
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = create_attention_mask(
                batch_size, seq_len, device, dtype
            )
        else:
            # Convert to causal mask format
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * torch.finfo(dtype).min
        
        # Prepare position IDs
        if position_ids is None:
            if past_key_values is not None:
                # Use last position for decoding
                start_pos = past_key_values[0][0].shape[2]
                position_ids = torch.arange(
                    start_pos, start_pos + seq_len,
                    device=device
                ).unsqueeze(0)
            else:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # Process through layers
        all_hidden_states = () if output_hidden_states else None
        present_key_values = () if use_cache else None
        
        for i, layer in enumerate(self.layers):
            # Get past KV for this layer
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            # Layer forward
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            hidden_states, present_kv = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache
            )
            
            if use_cache:
                present_key_values = present_key_values + (present_kv,)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Output projection
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            # Tied weights
            logits = nn.functional.linear(
                hidden_states,
                self.embed_tokens.embedding.weight
            )
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output = output + (all_hidden_states,)
            if use_cache:
                output = output + (present_key_values,)
            return ((loss,) + output) if loss is not None else output
        
        return ModelOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            past_key_values=present_key_values,
            loss=loss
        )
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids (torch.Tensor): Starting token IDs
            max_new_tokens (int): Maximum new tokens to generate
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling
            top_p (float): Top-p (nucleus) sampling
            repetition_penalty (float): Penalty for repetition
            eos_token_id (int, optional): End of sequence token
            pad_token_id (int, optional): Padding token
            use_cache (bool): Use KV cache
            
        Returns:
            Generated token IDs
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        
        generated = input_ids.clone()
        past_kv = None
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(
                input_ids=generated,
                past_key_values=past_kv,
                use_cache=use_cache
            )
            
            # Get last token logits
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0 and generated is not None:
                for token_id in set(generated[0].tolist()):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(
                    next_token_logits, top_k
                )[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Update cache
            past_kv = outputs.past_key_values
            
            # Check for EOS
            if next_token.item() == eos_token_id:
                break
        
        return generated
    
    def __str__(self) -> str:
        """String representation."""
        return f"NusaTinyLLM(config={self.config})"


def create_model(config: ModelConfig) -> NusaTinyLLM:
    """Create model from config."""
    return NusaTinyLLM(config)


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Optional[ModelConfig] = None,
    config_path: Optional[str] = None,
    device: str = 'cpu'
) -> NusaTinyLLM:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        config (ModelConfig, optional): Config object
        config_path (str, optional): Path to config JSON
        device (str): Device to load model on
        
    Returns:
        Loaded model
    """
    # Load config
    if config is None:
        if config_path is not None:
            config = ModelConfig.from_json(config_path)
        else:
            raise ValueError("Must provide config or config_path")
    
    # Create model
    model = NusaTinyLLM(config)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model
