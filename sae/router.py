from typing import Dict, Any

import torch
import torch.nn as nn

from .core import BaseSAE
from . import models
from .matryoshka import MatryoshkaSAE


class Router(nn.Module):
    """
    A router module that learns to route hidden states from different layers
    of a language model to an SAE.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        hidden_size = config["hidden_size"]
        n_layers = config["n_layers"]
        
        # Default routing layers if not specified
        self.start_layer = config.get("start_layer", n_layers // 5)
        self.end_layer = config.get("end_layer", n_layers * 4 // 5 + 1)

        print(f"Router layers: {self.start_layer} to {self.end_layer}")
        
        self.router = nn.Linear(hidden_size, self.end_layer - self.start_layer, bias=False)

    def forward(self, x: torch.Tensor, aggre: str = 'sum') -> torch.Tensor:
        """
        Forward pass of the router.

        Args:
            x (torch.Tensor): Hidden states from the language model, expected
                              shape (batch_size, seq_len, n_layers, hidden_size).
            aggre (str): Aggregation method ('sum' or 'mean').

        Returns:
            torch.Tensor: Softmaxed router weights.
        """
        if aggre == 'sum':
            router_input = x.sum(dim=2)
        elif aggre == 'mean':
            router_input = x.mean(dim=2)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggre}")
        
        router_output = self.router(router_input)
        return torch.softmax(router_output, dim=-1)


class RouteSAE(BaseSAE):
    """
    A composite SAE that uses a Router to select and process hidden states
    from multiple layers of a language model.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.router = Router(config)

        sae_type = config.get("base_sae_type", "TopKSAE")
        SAEClass = getattr(models, sae_type, None)
        if SAEClass is None:
            raise ValueError(f"Unknown base_sae_type: {sae_type}")
        
        self.sae = SAEClass(config)

    def forward(self, x: torch.Tensor, aggre: str = 'sum', routing: str = 'hard') -> Dict[str, torch.Tensor]:
        # x shape: (batch_size, max_length, n_layers, hidden_size)
        
        router_weights = self.router(x, aggre=aggre)
        
        if routing == 'hard':
            max_weights, target_layer_indices = router_weights.max(dim=-1)
            
            # Gather the hidden states from the target layers
            # The shape of target_layer_indices is (batch_size, max_length)
            # We need to expand it to match the dimensions of x for gathering
            expanded_indices = target_layer_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, x.size(-1))
            sae_input = torch.gather(x, 2, expanded_indices).squeeze(2)
            
            # Weight the selected hidden states by the router weights
            sae_input = sae_input * max_weights.unsqueeze(-1)
            
        elif routing == 'soft':
            weighted_hidden_states = x * router_weights.unsqueeze(-1)
            sae_input = weighted_hidden_states.sum(dim=2)
        else:
            raise ValueError(f"Unsupported routing method: {routing}")

        output = self.sae(sae_input)
        output["router_weights"] = router_weights
        if routing == 'hard':
            output["target_layer_indices"] = target_layer_indices
        output["sae_input"] = sae_input
        
        return output

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("RouteSAE does not support direct encoding/decoding.")

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("RouteSAE does not support direct encoding/decoding.")


class RouteMatryoshkaSAE(RouteSAE):
    """
    A specialized version of RouteSAE that uses a MatryoshkaSAE as its base.
    """
    def __init__(self, config: Dict[str, Any]):
        # We call BaseSAE's init directly to avoid RouteSAE's init logic
        BaseSAE.__init__(self, config)
        self.router = Router(config)
        self.sae = MatryoshkaSAE(config)
