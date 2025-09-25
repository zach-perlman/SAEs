from abc import ABC, abstractmethod
from typing import Dict, Any

import torch
import torch.nn as nn


class BaseSAE(nn.Module, ABC):
    """
    Abstract base class for Sparse Autoencoders (SAEs).

    This class provides a common interface for all SAE implementations. Subclasses
    must implement the `encode` and `decode` methods. The `forward` method is
    provided, which chains `encode` and `decode`.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the BaseSAE.

        Args:
            config (Dict[str, Any]): A dictionary containing the configuration
                                     for the SAE model.
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor into a sparse representation.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The sparse representation of the input.
        """
        pass

    @abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes the sparse representation back into the original space.

        Args:
            x (torch.Tensor): The sparse representation.

        Returns:
            torch.Tensor: The reconstructed tensor.
        """
        pass



    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Performs the forward pass of the SAE.

        The forward pass consists of encoding the input to get a sparse
        representation, and then decoding it to reconstruct the original input.
        This method returns a dictionary containing the sparse representation,
        the reconstructed input, and any other relevant tensors for loss
        calculations or analysis.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the output tensors,
                                     including 'sae_out' (reconstructed input)
                                     and 'feature_acts' (sparse representation).
        """
        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)
        
        return {"sae_out": sae_out, "feature_acts": feature_acts}
