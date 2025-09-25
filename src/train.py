from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
from tqdm import tqdm
import os

from sae.core import BaseSAE
from sae import models as sae_models
from sae.matryoshka import MatryoshkaSAE
from sae.router import RouteMatryoshkaSAE, RouteSAE
from src.utils import get_hidden_states, CosineWarmupLR, unit_norm_decoder, \
    compute_variance_explained, Normalized_MSE_loss


class SAELoss(nn.Module):
    """
    Computes the loss for a Sparse Autoencoder.
    """

    def __init__(self, l1_coefficient: float, auxk_alpha: float):
        super().__init__()
        self.l1_coefficient = l1_coefficient
        self.auxk_alpha = auxk_alpha

    def forward(
        self,
        original_hidden_states: torch.Tensor,
        sae_model: BaseSAE,
        sae_out: torch.Tensor,
        feature_acts: torch.Tensor,
        num_tokens_since_fired: torch.Tensor,
        dead_feature_threshold: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates the loss, including an auxiliary loss for dead features.
        """
        mse_loss = Normalized_MSE_loss(original_hidden_states, sae_out)
        
        l1_loss = torch.tensor(0.0, device=mse_loss.device)
        if isinstance(sae_model, (sae_models.VanillaSAE, sae_models.GatedSAE, sae_models.JumpReLUSAE)):
            l1_loss = torch.norm(feature_acts, p=1, dim=-1).mean()
        
        # Dead feature revival loss
        dead_features = num_tokens_since_fired >= dead_feature_threshold
        
        auxk_loss = torch.tensor(0.0, device=mse_loss.device)
        if dead_features.any():
            residual = original_hidden_states - sae_out
            dead_feature_indices = torch.where(dead_features)[0]
            
            # Get the correct decoder weights
            if isinstance(sae_model, (MatryoshkaSAE, RouteMatryoshkaSAE)):
                sae_instance = sae_model.sae if isinstance(sae_model, RouteMatryoshkaSAE) else sae_model
                decoder_weights = sae_instance.W_dec
            else:
                decoder_weights = sae_model.decoder.weight.T # Transpose to match Matryoshka's shape

            dead_decoder_weights = decoder_weights[dead_feature_indices, :].T # Now [hidden_size, num_dead]
            
            residual_projection = residual @ dead_decoder_weights
            
            aux_reconstruction = residual_projection @ dead_decoder_weights.T
            
            auxk_loss = F.mse_loss(aux_reconstruction, residual)

        total_loss = mse_loss + self.l1_coefficient * l1_loss + self.auxk_alpha * auxk_loss

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "auxk_loss": auxk_loss,
            "dead_features": dead_features.sum().float()
        }


class Trainer:
    """
    Handles the training process for the SAE.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        sae_model: BaseSAE,
        language_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        train_data_loader: DataLoader,
    ):
        self.config = config
        self.sae_model = sae_model.to(config["training"]["device"])
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.train_data_loader = train_data_loader
        self.optimizer = Adam(
            self.sae_model.parameters(),
            lr=config["training"]["lr"],
            betas=tuple(config["training"]["betas"])
        )
        self.scheduler = CosineWarmupLR(
            self.optimizer,
            num_warmup_steps=config["training"].get("num_warmup_steps", 0),
            num_training_steps=config["training"].get("steps", 10000) # Assuming steps-based training
        )
        self.loss_fn = SAELoss(
            l1_coefficient=config["training"].get("l1_coefficient", 0.001),
            auxk_alpha=config["training"].get("auxk_alpha", 0.0)
        )
        self.dead_feature_threshold = config["training"].get("dead_feature_threshold", 0)
        
        if self.dead_feature_threshold > 0:
            self.num_tokens_since_fired = torch.zeros(
                config["model"]["latent_size"], 
                dtype=torch.long, 
                device=config["training"]["device"]
            )
        else:
            self.num_tokens_since_fired = None

        # Early stopping
        self.early_stopping_patience = config["training"].get("early_stopping_patience", 0)
        self.early_stopping_min_delta = config["training"].get("early_stopping_min_delta", 0.0)
        self.best_loss = float('inf')
        self.patience_counter = 0

        # Ensure the models directory exists for saving checkpoints
        os.makedirs("models", exist_ok=True)

        if config["wandb"]["use_wandb"]:
            wandb.init(
                project=config["wandb"]["project"],
                config=config,
                entity=config["wandb"].get("entity")
            )

    def train(self):
        """
        Runs the training loop.
        """
        self.sae_model.train()
        
        total_tokens_processed = 0
        
        if "steps" not in self.config["training"]:
            raise ValueError("The 'steps' parameter must be defined in the training config for streaming datasets.")
        num_training_steps = self.config["training"]["steps"]
        
        progress_bar = tqdm(self.train_data_loader, total=num_training_steps, desc="Training")
        
        step = 0
        for batch in progress_bar:
            if step >= num_training_steps:
                break
            
            self.optimizer.zero_grad()
            
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
            
            if self.config["model"]["type"] in ["RouteSAE", "RouteMatryoshkaSAE"]:
                sae_output = self.sae_model(
                    hidden_states,
                    aggre=self.config["model"]["aggre"],
                    routing=self.config["model"]["routing"]
                )
            else:
                sae_output = self.sae_model(hidden_states)
            
            if isinstance(self.sae_model, RouteSAE):
                reconstruction_target = sae_output["sae_input"]
            else:
                reconstruction_target = hidden_states

            loss_dict = self.loss_fn(
                original_hidden_states=reconstruction_target,
                sae_model=self.sae_model,
                sae_out=sae_output["sae_out"],
                feature_acts=sae_output["feature_acts"],
                num_tokens_since_fired=self.num_tokens_since_fired,
                dead_feature_threshold=self.dead_feature_threshold
            )
            
            loss = loss_dict["loss"]
            loss.backward()
            
            # Gradient Clipping
            if self.config["training"].get("gradient_clip_norm"):
                torch.nn.utils.clip_grad_norm_(
                    self.sae_model.parameters(), 
                    self.config["training"]["gradient_clip_norm"]
                )

            self.optimizer.step()
            self.scheduler.step()
            
            # Unit norm decoder weights periodically
            unit_norm_freq = self.config["training"].get("unit_norm_frequency")
            if unit_norm_freq and (step + 1) % unit_norm_freq == 0:
                unit_norm_decoder(self.sae_model)

            # Update dead feature tracker
            if self.num_tokens_since_fired is not None:
                # A more memory-efficient way to calculate fired features
                fired_features = (sae_output["feature_acts"].abs() > 0).sum(dim=(0, 1)) > 0
                non_padding_tokens = batch["attention_mask"].sum().item()
                
                self.num_tokens_since_fired += non_padding_tokens
                self.num_tokens_since_fired[fired_features] = 0

            # Logging
            total_tokens_processed += non_padding_tokens
                
            l0_norm = (sae_output["feature_acts"] > 0).float().sum(dim=-1).mean().item()
            variance_explained = compute_variance_explained(
                reconstruction_target, sae_output["sae_out"]
            ).item()

            log_data = {
                **{k: v.item() for k, v in loss_dict.items()},
                "l0_norm": l0_norm,
                "variance_explained": variance_explained,
                "tokens_processed": total_tokens_processed,
            }

            if isinstance(self.sae_model, RouteSAE):
                start_layer = self.sae_model.router.start_layer
                routing_type = self.config["model"].get("routing", "hard")
                if routing_type == 'hard':
                    unique_layers, counts = torch.unique(sae_output["target_layer_indices"], return_counts=True)
                    layer_dist = {f"layer_{start_layer + l.item()}": c.item() / sae_output["target_layer_indices"].numel() for l, c in zip(unique_layers, counts)}
                else: # soft
                    avg_weights = sae_output["router_weights"].mean(dim=[0, 1])
                    layer_dist = {f"layer_{start_layer + i}": w.item() for i, w in enumerate(avg_weights)}
                log_data["layer_distribution"] = layer_dist
            
            if self.config["wandb"]["use_wandb"]:
                wandb.log(log_data)
            
            if step % self.config["training"].get("log_frequency", 5) == 0:
                print(f"\n--- Step {step} ---")
                
                # Print metrics in a structured format
                metrics_to_print = {
                    "loss": log_data.get("loss"),
                    "mse_loss": log_data.get("mse_loss"),
                    "l1_loss": log_data.get("l1_loss"),
                    "l0_norm": log_data.get("l0_norm"),
                    "variance_explained": log_data.get("variance_explained"),
                    "dead_features": log_data.get("dead_features"),
                    "tokens_processed": log_data.get("tokens_processed")
                }
                
                for key, value in metrics_to_print.items():
                    if value is not None:
                        if isinstance(value, float):
                            print(f"{key:<20}: {value:.4f}")
                        else:
                            print(f"{key:<20}: {value}")
                
                if "layer_distribution" in log_data:
                    filtered_dist = {k: v for k, v in log_data["layer_distribution"].items() if v > 0.05}
                    sorted_layers = sorted(filtered_dist.items(), key=lambda item: int(item[0].split('_')[1]))
                    dist_str = ", ".join([f"L{k.split('_')[1]}:{v:.3f}" for k, v in sorted_layers])
                    print(f"{'Layer Distribution':<20}: {dist_str}")
                
                print("-" * 32)
            
            postfix_data = {
                "Loss": f"{loss.item():.4f}",
                "MSE": f"{loss_dict['mse_loss'].item():.4f}",
                "L1": f"{loss_dict['l1_loss'].item():.4f}",
                "AuxK": f"{loss_dict['auxk_loss'].item():.4f}",
                "L0": f"{l0_norm:.2f}",
                "VarExp": f"{variance_explained:.3f}",
                "Dead": f"{loss_dict['dead_features'].item():.0f}",
                "LR": f"{self.scheduler.get_last_lr()[0]:.2e}",
                "Tokens": f"{total_tokens_processed / 1e6:.2f}M"
            }

            if isinstance(self.sae_model, RouteSAE):
                # Add layer distribution to the progress bar in a compact form
                filtered_dist = {k: v for k, v in log_data["layer_distribution"].items() if v > 0.05}
                dist_str = ", ".join([f"{k.split('_')[1]}:{v:.3f}" for k, v in filtered_dist.items()])
                postfix_data["Layers"] = dist_str
            
            progress_bar.set_postfix(postfix_data)
            
            # Early stopping check based on MSE loss
            if self.early_stopping_patience > 0:
                current_mse_loss = loss_dict['mse_loss'].item()
                if current_mse_loss < self.best_loss - self.early_stopping_min_delta:
                    self.best_loss = current_mse_loss
                    self.patience_counter = 0
                    torch.save(self.sae_model.state_dict(), "models/best_model.pt")
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {step+1} steps.")
                    break
            
            step += 1

        if self.config["wandb"]["use_wandb"]:
            wandb.finish()
            
        # Load the best model if early stopping was used
        if self.early_stopping_patience > 0 and os.path.exists("models/best_model.pt"):
            self.sae_model.load_state_dict(torch.load("models/best_model.pt"))
            print("Loaded best model from early stopping checkpoint.")

        model_name = f"{self.config['model']['type']}_{self.config['model']['layer']}.pt"
        save_path = os.path.join("models", model_name)
        torch.save(self.sae_model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
