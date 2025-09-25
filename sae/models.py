from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from .core import BaseSAE


class VanillaSAE(BaseSAE):
    """
    Vanilla Sparse Autoencoder.

    Implements:
        latents = ReLU(encoder(x - pre_bias) + latent_bias)
        reconstruction = decoder(latents) + pre_bias
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        hidden_size = config["hidden_size"]
        latent_size = config["latent_size"]

        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))
        self.latent_bias = nn.Parameter(torch.zeros(latent_size))
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)

        # "tied" init
        self.decoder.weight.data = self.encoder.weight.data.T.clone()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.pre_bias
        pre_acts = self.encoder(x) + self.latent_bias
        return F.relu(pre_acts)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents) + self.pre_bias


class GatedSAE(BaseSAE):
    """
    Gated Sparse Autoencoder.

    Implements:
        latents = gate(pre_acts + gate_bias) * relu(r_mag.exp() * pre_acts + mag_bias)
        reconstruction = decoder(latents) + pre_bias
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        hidden_size = config["hidden_size"]
        latent_size = config["latent_size"]

        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))
        self.gate_bias = nn.Parameter(torch.zeros(latent_size))
        self.mag_bias = nn.Parameter(torch.zeros(latent_size))
        self.r_mag = nn.Parameter(torch.zeros(latent_size))
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)

        # "tied" init
        self.decoder.weight.data = self.encoder.weight.data.T.clone()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = self.encoder(x - self.pre_bias)

        pi_gate = pre_acts + self.gate_bias
        f_gate = (pi_gate > 0).float()

        pi_mag = torch.exp(self.r_mag) * pre_acts + self.mag_bias
        f_mag = F.relu(pi_mag)

        return f_gate * f_mag

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents) + self.pre_bias


class TopKSAE(BaseSAE):
    """
    Top-K Sparse Autoencoder.

    Implements:
        latents = TopK(ReLU(encoder(x - pre_bias) + latent_bias))
        reconstruction = decoder(latents) + pre_bias
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        hidden_size = config["hidden_size"]
        latent_size = config["latent_size"]
        self.k = config["k"]
        
        assert self.k <= latent_size, f"k ({self.k}) should be less than or equal to latent_size ({latent_size})"

        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))
        self.latent_bias = nn.Parameter(torch.zeros(latent_size))
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)

        # "tied" init
        self.decoder.weight.data = self.encoder.weight.data.T.clone()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.pre_bias
        pre_acts = self.encoder(x) + self.latent_bias
        
        pre_acts = F.relu(pre_acts)

        # Get the values and indices of the top k activations
        topk_values, topk_indices = torch.topk(pre_acts, self.k, dim=-1)

        # Create a sparse tensor with only the top k activations
        latents = torch.zeros_like(pre_acts)
        latents.scatter_(-1, topk_indices, topk_values)
        
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents) + self.pre_bias

class Jump_func(autograd.Function):
    """
    Jump ReLU activation function.
    """
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return x * (x > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            - (threshold / bandwidth) 
            * ((x - threshold).abs() < bandwidth / 2).float()
            * grad_output
        )
        return x_grad, threshold_grad, None


class JumpReLUSAE(BaseSAE):
    """
    JumpReLU Sparse Autoencoder.

    Implements:
        latents = JumpReLU(encoder(x - pre_bias) + latent_bias, threshold, bandwidth)
        reconstruction = decoder(latents) + pre_bias
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        hidden_size = config["hidden_size"]
        latent_size = config["latent_size"]
        threshold = config.get("threshold", 1e-3)
        self.bandwidth = config.get("bandwidth", 1e-3)

        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))
        self.latent_bias = nn.Parameter(torch.zeros(latent_size))
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)
        self.threshold = nn.Parameter(torch.full((latent_size,), threshold))

        # "tied" init
        self.decoder.weight.data = self.encoder.weight.data.T.clone()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.pre_bias
        pre_acts = self.encoder(x) + self.latent_bias
        return Jump_func.apply(pre_acts, self.threshold, self.bandwidth)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents) + self.pre_bias
