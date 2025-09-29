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
    compute_variance_explained, Normalized_MSE_loss, remove_gradient_parallel_to_decoder_directions, \
    geometric_median


class SAELoss(nn.Module):
    """
    Computes the loss for a Sparse Autoencoder.
    
    For Matryoshka SAEs, this implements group-wise incremental loss computation,
    where each nested dictionary level is trained to improve reconstruction
    progressively. This is essential for proper Matryoshka training.
    """

    def __init__(self, l1_coefficient: float, auxk_alpha: float, group_weights: list = None):
        super().__init__()
        self.l1_coefficient = l1_coefficient
        self.auxk_alpha = auxk_alpha
        self.group_weights = group_weights

    def forward(
        self,
        original_hidden_states: torch.Tensor,
        sae_model: BaseSAE,
        sae_out: torch.Tensor,
        feature_acts: torch.Tensor,
        num_tokens_since_fired: torch.Tensor,
        dead_feature_threshold: int,
        post_relu_acts: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates the loss, including an auxiliary loss for dead features.
        
        For Matryoshka SAEs, computes group-wise incremental reconstruction loss
        where each group's contribution is weighted and accumulated.
        """
        # For Matryoshka SAEs, compute group-wise incremental loss
        if isinstance(sae_model, (MatryoshkaSAE, RouteMatryoshkaSAE)):
            sae_instance = sae_model.sae if isinstance(sae_model, RouteMatryoshkaSAE) else sae_model
            
            # Reshape inputs for loss computation
            original_flat = original_hidden_states.reshape(-1, original_hidden_states.shape[-1])
            feature_acts_flat = feature_acts.reshape(-1, feature_acts.shape[-1])
            
            # Get group information
            group_sizes = sae_instance.group_sizes.tolist()
            active_groups = sae_instance.active_groups
            
            # Use provided group weights or uniform weights
            if self.group_weights is not None:
                weights = self.group_weights[:active_groups]
            else:
                weights = [1.0 / active_groups] * active_groups
            
            # Split features into groups
            W_dec_chunks = torch.split(sae_instance.W_dec, group_sizes, dim=0)
            f_chunks = torch.split(feature_acts_flat, group_sizes, dim=1)
            
            # Compute incremental reconstruction loss for each group
            x_reconstruct = torch.zeros_like(original_flat) + sae_instance.b_dec
            total_mse_loss = 0.0
            
            # Track per-group statistics
            group_stats = []
            
            for i in range(active_groups):
                W_dec_slice = W_dec_chunks[i]
                acts_slice = f_chunks[i]
                
                # Add this group's contribution to reconstruction
                x_reconstruct = x_reconstruct + acts_slice @ W_dec_slice
                
                # Compute MSE for this level of reconstruction
                group_mse = ((original_flat - x_reconstruct) ** 2).sum(dim=-1).mean()
                total_mse_loss += group_mse * weights[i]
                
                # Compute per-group variance explained
                total_variance = torch.var(original_flat.float(), dim=0).sum()
                residual_variance = torch.var((original_flat - x_reconstruct).float(), dim=0).sum()
                group_var_exp = 1 - residual_variance / (total_variance + 1e-9)
                
                # Compute per-group L0 (sparsity)
                group_l0 = (acts_slice.abs() > 0).float().sum(dim=-1).mean()
                
                group_stats.append({
                    'mse': group_mse.item(),
                    'var_exp': group_var_exp.item(),
                    'l0': group_l0.item()
                })
            
            mse_loss = total_mse_loss
            
        else:
            # Standard loss for non-Matryoshka models
            mse_loss = Normalized_MSE_loss(original_hidden_states, sae_out)
            group_stats = None
        
        # L1 sparsity loss (only for certain architectures)
        l1_loss = torch.tensor(0.0, device=mse_loss.device)
        if isinstance(sae_model, (sae_models.VanillaSAE, sae_models.GatedSAE, sae_models.JumpReLUSAE)):
            l1_loss = torch.norm(feature_acts, p=1, dim=-1).mean()
        
        # Dead feature revival loss with proper normalization
        dead_features = num_tokens_since_fired >= dead_feature_threshold
        
        auxk_loss = torch.tensor(0.0, device=mse_loss.device)
        if dead_features.any() and post_relu_acts is not None:
            # Get residual for auxiliary reconstruction
            residual = original_hidden_states - sae_out
            residual_flat = residual.reshape(-1, residual.shape[-1])
            
            # Get the correct decoder weights
            if isinstance(sae_model, (MatryoshkaSAE, RouteMatryoshkaSAE)):
                sae_instance = sae_model.sae if isinstance(sae_model, RouteMatryoshkaSAE) else sae_model
                # For Matryoshka: Use top-k dead features for auxiliary loss
                dead_feature_indices = torch.where(dead_features)[0]
                
                if len(dead_feature_indices) > 0:
                    # Get pre-activation values for dead features
                    post_relu_flat = post_relu_acts.reshape(-1, post_relu_acts.shape[-1])
                    
                    # Compute top-k auxiliary loss (similar to reference implementation)
                    activation_dim = sae_instance.W_dec.shape[1]
                    k_aux = min(activation_dim // 2, len(dead_feature_indices))
                    
                    if k_aux > 0:
                        # Select only dead features and get top-k
                        auxk_latents = torch.where(
                            dead_features[None], 
                            post_relu_flat, 
                            -torch.inf
                        )
                        auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
                        
                        # Create sparse activation buffer
                        auxk_buffer = torch.zeros_like(post_relu_flat)
                        auxk_acts_sparse = auxk_buffer.scatter_(
                            dim=-1, index=auxk_indices, src=auxk_acts
                        )
                        
                        # Reconstruct using only dead features (no bias)
                        x_reconstruct_aux = auxk_acts_sparse @ sae_instance.W_dec
                        
                        # Compute MSE on residual
                        l2_loss_aux = ((residual_flat.float() - x_reconstruct_aux.float()) ** 2).sum(dim=-1).mean()
                        
                        # Normalize by residual variance (following reference implementation)
                        residual_mu = residual_flat.mean(dim=0, keepdim=True)
                        loss_denom = ((residual_flat.float() - residual_mu.float()) ** 2).sum(dim=-1).mean()
                        
                        auxk_loss = (l2_loss_aux / (loss_denom + 1e-9)).nan_to_num(0.0)
            else:
                # Standard auxiliary loss for other models
                dead_feature_indices = torch.where(dead_features)[0]
                decoder_weights = sae_model.decoder.weight.T
                dead_decoder_weights = decoder_weights[dead_feature_indices, :].T
                residual_flat = residual.reshape(-1, residual.shape[-1])
                residual_projection = residual_flat @ dead_decoder_weights
                aux_reconstruction = residual_projection @ dead_decoder_weights.T
                auxk_loss = F.mse_loss(aux_reconstruction, residual_flat)

        total_loss = mse_loss + self.l1_coefficient * l1_loss + self.auxk_alpha * auxk_loss

        return {
            "loss": total_loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "auxk_loss": auxk_loss,
            "dead_features": dead_features.sum().float(),
            "group_stats": group_stats  # Per-group statistics for Matryoshka SAEs
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
        
        # Get group weights for Matryoshka SAEs
        group_weights = None
        if config["model"]["type"] in ["MatryoshkaSAE", "RouteMatryoshkaSAE"]:
            group_weights = config["model"].get("group_weights", None)
        
        self.loss_fn = SAELoss(
            l1_coefficient=config["training"].get("l1_coefficient", 0.001),
            auxk_alpha=config["training"].get("auxk_alpha", 0.0),
            group_weights=group_weights
        )
        self.dead_feature_threshold = config["training"].get("dead_feature_threshold", 0)
        
        # Threshold update parameters for Matryoshka SAEs
        self.threshold_beta = config["training"].get("threshold_beta", 0.999)
        self.threshold_start_step = config["training"].get("threshold_start_step", 1000)
        
        # Gradient accumulation for larger effective batch sizes
        self.gradient_accumulation_steps = config["training"].get("gradient_accumulation_steps", 1)
        
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
    
    def update_threshold(self, feature_acts: torch.Tensor, sae_model: BaseSAE, step: int):
        """
        Update the learned threshold using exponential moving average of minimum activations.
        This helps with stability and dead feature handling.
        """
        if not isinstance(sae_model, (MatryoshkaSAE, RouteMatryoshkaSAE)):
            return
        
        if step < self.threshold_start_step:
            return
        
        sae_instance = sae_model.sae if isinstance(sae_model, RouteMatryoshkaSAE) else sae_model
        
        with torch.no_grad():
            # Find minimum non-zero activation
            active = feature_acts[feature_acts > 0]
            
            if active.numel() == 0:
                min_activation = 0.0
            else:
                min_activation = active.min().detach().float()
            
            # Initialize or update threshold with exponential moving average
            if sae_instance.threshold < 0:
                sae_instance.threshold.fill_(min_activation)
            else:
                sae_instance.threshold.mul_(self.threshold_beta).add_(
                    min_activation * (1 - self.threshold_beta)
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
            
            # Gradient accumulation: only zero gradients at the start of accumulation
            if step % self.gradient_accumulation_steps == 0:
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
            
            # Geometric median initialization on first batch
            if step == 0 and isinstance(self.sae_model, (MatryoshkaSAE, RouteMatryoshkaSAE)):
                sae_instance = self.sae_model.sae if isinstance(self.sae_model, RouteMatryoshkaSAE) else self.sae_model
                with torch.no_grad():
                    hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
                    median = geometric_median(hidden_flat)
                    sae_instance.b_dec.data = median
                    print(f"Initialized b_dec with geometric median")
            
            # Forward pass - get post_relu_acts for Matryoshka models
            if self.config["model"]["type"] in ["RouteSAE", "RouteMatryoshkaSAE"]:
                sae_output = self.sae_model(
                    hidden_states,
                    aggre=self.config["model"]["aggre"],
                    routing=self.config["model"]["routing"]
                )
            elif isinstance(self.sae_model, MatryoshkaSAE):
                # For Matryoshka, get additional outputs needed for loss computation
                sae_output = self.sae_model(hidden_states, return_active=True)
            else:
                sae_output = self.sae_model(hidden_states)
            
            if isinstance(self.sae_model, RouteSAE):
                reconstruction_target = sae_output["sae_input"]
            else:
                reconstruction_target = hidden_states

            # Get post_relu_acts for auxiliary loss if available
            post_relu_acts = sae_output.get("post_relu_acts", None)
            
            loss_dict = self.loss_fn(
                original_hidden_states=reconstruction_target,
                sae_model=self.sae_model,
                sae_out=sae_output["sae_out"],
                feature_acts=sae_output["feature_acts"],
                num_tokens_since_fired=self.num_tokens_since_fired,
                dead_feature_threshold=self.dead_feature_threshold,
                post_relu_acts=post_relu_acts
            )
            
            loss = loss_dict["loss"]
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / self.gradient_accumulation_steps
            scaled_loss.backward()
            
            # Update threshold for Matryoshka SAEs after threshold_start_step
            if isinstance(self.sae_model, (MatryoshkaSAE, RouteMatryoshkaSAE)):
                self.update_threshold(sae_output["feature_acts"], self.sae_model, step)
            
            # Only update weights after accumulating gradients
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Remove gradient component parallel to decoder directions (prevents decoder collapse)
                remove_gradient_parallel_to_decoder_directions(self.sae_model)
                
                # Gradient Clipping
                if self.config["training"].get("gradient_clip_norm"):
                    torch.nn.utils.clip_grad_norm_(
                        self.sae_model.parameters(), 
                        self.config["training"]["gradient_clip_norm"]
                    )

                self.optimizer.step()
                self.scheduler.step()
                
                # Enforce unit norm on decoder weights every step (critical for stability)
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
                **{k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items() if k != "group_stats"},
                "l0_norm": l0_norm,
                "variance_explained": variance_explained,
                "tokens_processed": total_tokens_processed,
            }
            
            # Add per-group stats if available
            if loss_dict.get("group_stats") is not None:
                log_data["group_stats"] = loss_dict["group_stats"]

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
                # Add per-group metrics to wandb
                wandb_data = log_data.copy()
                if "group_stats" in wandb_data and wandb_data["group_stats"] is not None:
                    for i, stats in enumerate(wandb_data["group_stats"]):
                        wandb_data[f"group_{i+1}/mse"] = stats['mse']
                        wandb_data[f"group_{i+1}/var_exp"] = stats['var_exp']
                        wandb_data[f"group_{i+1}/l0"] = stats['l0']
                    del wandb_data["group_stats"]  # Remove the dict itself
                wandb.log(wandb_data)
            
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
                
                # Print per-group (doll) statistics for Matryoshka SAEs
                if "group_stats" in log_data and log_data["group_stats"] is not None:
                    print(f"\n{'Matryoshka Dolls':<20}:")
                    for i, stats in enumerate(log_data["group_stats"]):
                        # Compact format: "Doll 1 (2048):  MSE=0.0123  VarExp=0.456  L0=12.3"
                        group_sizes = self.config["model"]["group_sizes"]
                        cumulative_size = sum(group_sizes[:i+1])
                        print(f"  Doll {i+1} ({cumulative_size:>5}): MSE={stats['mse']:.4f}  VarExp={stats['var_exp']:.3f}  L0={stats['l0']:.1f}")
                
                if "layer_distribution" in log_data:
                    filtered_dist = {k: v for k, v in log_data["layer_distribution"].items() if v > 0.05}
                    sorted_layers = sorted(filtered_dist.items(), key=lambda item: int(item[0].split('_')[1]))
                    dist_str = ", ".join([f"L{k.split('_')[1]}:{v:.3f}" for k, v in sorted_layers])
                    print(f"{'Layer Distribution':<20}: {dist_str}")
                
                print("-" * 64)
            
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
