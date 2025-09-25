from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from tqdm import tqdm
import wandb

from sae.core import BaseSAE
from src.utils import get_hidden_states, compute_variance_explained
from src.train import SAELoss


class Evaluator:
    """
    Handles the evaluation process for the SAE.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        sae_model: BaseSAE,
        language_model: AutoModelForCausalLM,
        data_loader: DataLoader,
    ):
        self.config = config
        self.sae_model = sae_model.to(config["training"]["device"])
        self.language_model = language_model
        self.data_loader = data_loader
        self.loss_fn = SAELoss(
            l1_coefficient=config["training"].get("l1_coefficient", 0.001),
            auxk_alpha=config["training"].get("auxk_alpha", 0.0)
        )

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Runs the evaluation loop.
        """
        self.sae_model.eval()
        
        total_loss = 0.0
        total_mse_loss = 0.0
        total_l1_loss = 0.0
        total_variance_explained = 0.0
        num_batches = 0

        progress_bar = tqdm(self.data_loader, desc="Evaluating")
        for batch in progress_bar:
            layer = self.config["model"]["layer"]
            if self.config["model"]["type"] in ["RouteSAE", "RouteMatryoshkaSAE"]:
                start = self.sae_model.router.start_layer
                end = self.sae_model.router.end_layer
                layer = list(range(start, end))

            hidden_states = get_hidden_states(
                self.language_model,
                batch,
                layer,
                self.config["training"]["device"]
            )
            
            # Handle routing parameters for RouteSAE
            if self.config["model"]["type"] in ["RouteSAE", "RouteMatryoshkaSAE"]:
                sae_output = self.sae_model(
                    hidden_states,
                    aggre=self.config["model"]["aggre"],
                    routing=self.config["model"]["routing"]
                )
            else:
                sae_output = self.sae_model(hidden_states)
            
            reconstruction_target = hidden_states
            if self.config["model"]["type"] in ["RouteSAE", "RouteMatryoshkaSAE"]:
                reconstruction_target = sae_output["sae_input"]

            loss_dict = self.loss_fn(
                original_hidden_states=reconstruction_target,
                sae_model=self.sae_model, # Pass the model for dead feature revival
                sae_out=sae_output["sae_out"],
                feature_acts=sae_output["feature_acts"],
                num_tokens_since_fired=torch.zeros(self.config["model"]["latent_size"]), # Dummy tensor
                dead_feature_threshold=self.config["training"].get("dead_feature_threshold", 0)
            )
            
            variance_explained = compute_variance_explained(
                reconstruction_target, sae_output["sae_out"]
            )

            total_loss += loss_dict["loss"].item()
            total_mse_loss += loss_dict["mse_loss"].item()
            total_l1_loss += loss_dict["l1_loss"].item()
            total_variance_explained += variance_explained.item()
            num_batches += 1

            metrics = {
                "eval_loss": total_loss / num_batches,
                "eval_mse_loss": total_mse_loss / num_batches,
                "eval_l1_loss": total_l1_loss / num_batches,
                "eval_variance_explained": total_variance_explained / num_batches
            }
            progress_bar.set_postfix(metrics)
        
        avg_metrics = {k: v for k, v in metrics.items()}

        if self.config["wandb"]["use_wandb"]:
            wandb.log(avg_metrics)
            
        return avg_metrics
