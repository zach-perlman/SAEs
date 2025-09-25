from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import BaseSAE


class MatryoshkaSAE(BaseSAE):
    """
    Matryoshka Sparse Autoencoder.

    This implementation follows the structure of the original paper, featuring
    nested dictionaries of features of increasing size. It uses a standard TopK
    activation function for sparsity.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        hidden_size = config["hidden_size"]
        latent_size = config["latent_size"]
        group_sizes: List[int] = config["group_sizes"]
        self.k = config["k"]

        assert sum(group_sizes) == latent_size, "group_sizes must sum to latent_size"
        assert all(s > 0 for s in group_sizes), "all group sizes must be positive"
        assert isinstance(self.k, int) and self.k > 0, f"k={self.k} must be a positive integer"

        self.W_enc = nn.Parameter(torch.empty(hidden_size, latent_size))
        self.b_enc = nn.Parameter(torch.zeros(latent_size))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(latent_size, hidden_size)))
        self.b_dec = nn.Parameter(torch.zeros(hidden_size))

        # Initialize decoder weights to unit norm, and encoder as its transpose
        self.W_dec.data = self._set_decoder_norm_to_unit_norm(self.W_dec.data.T).T
        self.W_enc.data = self.W_dec.data.clone().T

        # Track group boundaries for masking
        self.active_groups = len(group_sizes)
        group_indices = [0] + list(torch.cumsum(torch.tensor(group_sizes), dim=0).tolist())
        self.group_indices = group_indices
        self.register_buffer("group_sizes", torch.tensor(group_sizes))


    def _set_decoder_norm_to_unit_norm(self, W_dec: torch.Tensor) -> torch.Tensor:
        """Set decoder weights to unit norm."""
        return W_dec / torch.norm(W_dec, dim=0, keepdim=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # (x - b_dec) @ W_enc + b_enc, then ReLU
        post_relu_feat_acts = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

        # Use standard TopK per sample
        k_to_use = min(self.k, post_relu_feat_acts.shape[-1])
        
        if k_to_use <= 0:
            encoded_acts = torch.zeros_like(post_relu_feat_acts)
        else:
            topk_vals, topk_idxs = torch.topk(post_relu_feat_acts, k=k_to_use, dim=-1)
            encoded_acts = torch.zeros_like(post_relu_feat_acts)
            encoded_acts.scatter_(-1, topk_idxs, topk_vals)

        # Apply group masking
        max_act_index = self.group_indices[self.active_groups]
        encoded_acts[:, max_act_index:] = 0

        return encoded_acts

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latents back to activations."""
        return x @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        original_shape = x.shape
        
        # Handle multi-dimensional input by flattening
        x_reshaped = x.reshape(-1, original_shape[-1])

        feature_acts = self.encode(x_reshaped)
        sae_out = self.decode(feature_acts)

        # Reshape back to original dimensions
        sae_out = sae_out.reshape(original_shape)
        
        # Reshape latents to match input structure
        new_shape = original_shape[:-1] + (self.config["latent_size"],)
        feature_acts = feature_acts.reshape(new_shape)
            
        return {"sae_out": sae_out, "feature_acts": feature_acts}
