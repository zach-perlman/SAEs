from typing import Dict, Any, Tuple, Union, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
from torch.optim.lr_scheduler import LambdaLR

from sae import models as sae_models
from sae.router import RouteSAE, RouteMatryoshkaSAE
from sae.matryoshka import MatryoshkaSAE
import torch.nn.functional as F


def compute_variance_explained(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Computes the fraction of variance explained by the reconstruction.
    """
    total_variance = torch.var(original.float(), dim=0).sum()
    residual_variance = torch.var((original - reconstructed).float(), dim=0).sum()
    return 1 - residual_variance / (total_variance + 1e-9)


def Normalized_MSE_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    Computes the normalized mean squared error loss.
    """
    # Ensure inputs are float32 for stable calculation
    x = x.float()
    x_hat = x_hat.float()
    
    # Calculate squared errors and variance of the original signal
    error = ((x_hat - x) ** 2).mean(dim=-1)
    variance = (x.var(dim=-1))
    
    # Normalize by variance and take the mean
    return (error / (variance + 1e-9)).mean()


class CosineWarmupLR(LambdaLR):
    """
    Learning rate scheduler with linear warmup and cosine decay.
    """
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, decay_start_step=None, min_lr_ratio=0.02):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.decay_start_step = decay_start_step if decay_start_step is not None else num_warmup_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step: int):
        if step < self.num_warmup_steps:
            return float(step) / float(max(1, self.num_warmup_steps))
        elif step >= self.decay_start_step:
            progress = (step - self.decay_start_step) / (self.num_training_steps - self.decay_start_step)
            progress = min(progress, 1.0)
            cosine_term = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_term
        else:
            return 1.0


@torch.no_grad()
def unit_norm_decoder(model: torch.nn.Module) -> None:
    """
    Normalize the decoder weights to unit norm.
    """
    if isinstance(model, (sae_models.VanillaSAE, sae_models.GatedSAE, sae_models.TopKSAE, sae_models.JumpReLUSAE)):
        model.decoder.weight.data.div_(torch.norm(model.decoder.weight.data, dim=0, keepdim=True))
    elif isinstance(model, (RouteSAE, RouteMatryoshkaSAE)):
        # For routed models, normalize the base SAE's decoder
        if hasattr(model.sae, 'decoder'):
            model.sae.decoder.weight.data.div_(torch.norm(model.sae.decoder.weight.data, dim=0, keepdim=True))
        elif hasattr(model.sae, 'W_dec'): # Matryoshka style
             model.sae.W_dec.data.div_(torch.norm(model.sae.W_dec.data, dim=1, keepdim=True))
    elif isinstance(model, MatryoshkaSAE):
        model.W_dec.data.div_(torch.norm(model.W_dec.data, dim=1, keepdim=True))



def load_language_model(config: Dict[str, Any]) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Loads the language model and tokenizer from the specified path.

    Args:
        config (Dict[str, Any]): The configuration dictionary.

    Returns:
        Tuple[AutoTokenizer, AutoModelForCausalLM]: The tokenizer and the language model.
    """
    model_path = config["model"]["language_model_path"]
    device = config["training"]["device"]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    language_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        output_hidden_states=True,
        torch_dtype=torch.bfloat16,
        device_map="auto" if "cuda" in device else None
    )

    return tokenizer, language_model


def get_hidden_states(
    model: AutoModelForCausalLM,
    batch: Dict[str, torch.Tensor],
    layer: Union[int, List[int]],
    device: str
) -> torch.Tensor:
    """
    Performs a forward pass on the language model and extracts the hidden states.

    Args:
        model (AutoModelForCausalLM): The language model.
        batch (Dict[str, torch.Tensor]): The tokenized batch.
        layer (Union[int, List[int]]): The layer(s) from which to extract hidden states.
        device (str): The device to move the batch to.

    Returns:
        torch.Tensor: The hidden states from the specified layer(s).
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    hidden_states = outputs.hidden_states
    
    if isinstance(layer, int):
        # Single layer
        return hidden_states[layer]
    elif isinstance(layer, list):
        # Multiple layers for RouteSAE
        # Stack the hidden states from the specified layers
        return torch.stack([hidden_states[l] for l in layer], dim=2)

    return hidden_states
