"""
Token Embedding Layer

Implementation of token embeddings with optional tied output projection.
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Token embedding layer with optional weight tying.
    
    Maps token IDs to dense vectors and optionally shares weights
    with the output projection layer.
    
    Args:
        vocab_size (int): Size of vocabulary
        embedding_dim (int): Dimension of embeddings
        padding_idx (int, optional): Padding token index
        tie_weights (bool): Whether to tie input/output embeddings
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int = 0,
        tie_weights: bool = True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.tie_weights = tie_weights
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        # Use normal initialization with small std
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        # Ensure padding is zero
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.padding_idx].fill_(0)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for input token IDs.
        
        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch, seq_len)
            
        Returns:
            Embeddings of shape (batch, seq_len, embedding_dim)
        """
        return self.embedding(input_ids)
    
    def get_output_projection(self) -> nn.Linear:
        """
        Get output projection layer (tied or separate).
        
        When tie_weights is True, returns a linear layer that shares
        weights with the embedding. Otherwise returns a separate layer.
        
        Returns:
            Output projection layer
        """
        if self.tie_weights:
            # Create linear layer with shared weights
            output_proj = nn.Linear(
                self.embedding_dim,
                self.vocab_size,
                bias=False
            )
            # Share weights with embedding (transpose for linear layer)
            output_proj.weight = self.embedding.weight
            return output_proj
        else:
            # Separate output projection
            return nn.Linear(self.embedding_dim, self.vocab_size, bias=False)
    
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convenience method for getting embeddings."""
        return self.forward(input_ids)
    
    def project_to_vocab(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to vocabulary logits.
        
        Args:
            hidden_states (torch.Tensor): Hidden states (batch, seq_len, dim)
            
        Returns:
            Logits over vocabulary (batch, seq_len, vocab_size)
        """
        if self.tie_weights:
            # Use transposed embedding weights
            return F.linear(hidden_states, self.embedding.weight)
        else:
            # Use separate output projection
            return self.get_output_projection()(hidden_states)
    
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return f'vocab_size={self.vocab_size}, embedding_dim={self.embedding_dim}, ' \
               f'padding_idx={self.padding_idx}, tie_weights={self.tie_weights}'


# Import F at module level for project_to_vocab
import torch.nn.functional as F
