from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import BaseSAE


class MatryoshkaSAE(BaseSAE):
    """
    Matryoshka Sparse Autoencoder.

    This implementation follows the structure of the original paper, featuring
    nested dictionaries of features of increasing size. It uses batch-level TopK
    activation function for sparsity, where the top k*batch_size activations are
    selected across the entire batch.
    
    Key architectural features:
    - Batch-level TopK sparsity (not per-sample)
    - Learned adaptive threshold for inference
    - Group-wise incremental loss computation
    - Decoder gradient removal to maintain orthogonality
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
        
        # Learned threshold for inference (starts at -1 to indicate uninitialized)
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))


    def _set_decoder_norm_to_unit_norm(self, W_dec: torch.Tensor) -> torch.Tensor:
        """Set decoder weights to unit norm."""
        return W_dec / torch.norm(W_dec, dim=0, keepdim=True)

    def encode(self, x: torch.Tensor, use_threshold: bool = False, return_active: bool = False) -> torch.Tensor:
        """
        Encode input activations to sparse latent representation.
        
        Args:
            x: Input tensor of shape [batch, hidden_size]
            use_threshold: If True, use learned threshold instead of batch TopK (for inference)
            return_active: If True, return (encoded_acts, active_mask, post_relu_acts)
        
        Returns:
            Sparse encoded activations, optionally with active mask and pre-threshold acts
        """
        # Compute pre-activation features: (x - b_dec) @ W_enc + b_enc, then ReLU
        post_relu_feat_acts = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

        if use_threshold:
            # Inference mode: use learned threshold
            encoded_acts = post_relu_feat_acts * (post_relu_feat_acts > self.threshold)
        else:
            # Training mode: batch-level TopK
            # Flatten activations across batch and select top k*batch_size activations
            flattened_acts = post_relu_feat_acts.flatten()
            total_k = self.k * x.size(0)
            
            if total_k <= 0 or total_k > flattened_acts.numel():
                encoded_acts = torch.zeros_like(post_relu_feat_acts)
            else:
                # Get top-k values and indices across the entire batch
                post_topk = flattened_acts.topk(total_k, sorted=False, dim=-1)
                
                # Scatter top-k values back to original positions
                encoded_acts = (
                    torch.zeros_like(flattened_acts)
                    .scatter_(-1, post_topk.indices, post_topk.values)
                    .reshape(post_relu_feat_acts.shape)
                )

        # Apply group masking to enforce nested structure
        max_act_index = self.group_indices[self.active_groups]
        encoded_acts[:, max_act_index:] = 0

        if return_active:
            # Return which features fired (for dead feature tracking)
            active_mask = encoded_acts.sum(0) > 0
            return encoded_acts, active_mask, post_relu_feat_acts
        else:
            return encoded_acts

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latents back to activations."""
        return x @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor, use_threshold: bool = False, return_active: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the SAE.
        
        Args:
            x: Input tensor
            use_threshold: Whether to use threshold-based sparsity (inference)
            return_active: Whether to return active features and pre-threshold acts
        """
        original_shape = x.shape
        
        # Handle multi-dimensional input by flattening
        x_reshaped = x.reshape(-1, original_shape[-1])

        if return_active:
            feature_acts, active_mask, post_relu_acts = self.encode(
                x_reshaped, use_threshold=use_threshold, return_active=True
            )
            sae_out = self.decode(feature_acts)
            
            # Reshape back to original dimensions
            sae_out = sae_out.reshape(original_shape)
            new_shape = original_shape[:-1] + (self.config["latent_size"],)
            feature_acts = feature_acts.reshape(new_shape)
            post_relu_acts = post_relu_acts.reshape(new_shape)
            
            return {
                "sae_out": sae_out,
                "feature_acts": feature_acts,
                "active_mask": active_mask,
                "post_relu_acts": post_relu_acts
            }
        else:
            feature_acts = self.encode(x_reshaped, use_threshold=use_threshold)
            sae_out = self.decode(feature_acts)

            # Reshape back to original dimensions
            sae_out = sae_out.reshape(original_shape)
            new_shape = original_shape[:-1] + (self.config["latent_size"],)
            feature_acts = feature_acts.reshape(new_shape)
                
            return {"sae_out": sae_out, "feature_acts": feature_acts}
