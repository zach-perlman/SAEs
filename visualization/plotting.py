from typing import List, Dict, Any, Tuple
import os

import matplotlib.pyplot as plt
import torch
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from sae.core import BaseSAE
from src.utils import get_hidden_states
from sae.router import RouteSAE
from sae.matryoshka import MatryoshkaSAE


def plot_losses(run_id: str, project: str, output_dir: str = "visualizations"):
    """
    Fetches loss data from a wandb run and plots it.

    Args:
        run_id (str): The ID of the wandb run.
        project (str): The wandb project name.
        output_dir (str): The directory to save the plot in.
    """
    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")

    history = run.history()
    
    plt.figure(figsize=(10, 5))
    plt.plot(history["_step"], history["mse_loss"], label="MSE Loss")
    plt.plot(history["_step"], history["l1_loss"], label="L1 Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves for Run: {run.name}")
    plt.legend()
    plt.grid(True)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(os.path.join(output_dir, f"{run.name}_loss_curves.png"))
    plt.close()


def plot_feature_activation_histogram(
    feature_acts: torch.Tensor,
    title: str = "Feature Activation Histogram",
    output_dir: str = "visualizations",
    filename: str = "feature_activation_histogram.png"
):
    """
    Plots a histogram of the feature activations.

    Args:
        feature_acts (torch.Tensor): The feature activations.
        title (str): The title of the plot.
        output_dir (str): The directory to save the plot in.
        filename (str): The filename for the saved plot.
    """
    activations = feature_acts.detach().cpu().numpy().flatten()
    
    # Filter out zeros for a more informative plot of active features
    activations = activations[activations > 0]
    
    plt.figure(figsize=(10, 5))
    plt.hist(activations, bins=100, log=True)
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency (log scale)")
    plt.title(title)
    plt.grid(True)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


class FeatureDashboard:
    """
    Generates an HTML dashboard to visualize SAE feature activations.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        sae_model: BaseSAE,
        language_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        data_loader: torch.utils.data.DataLoader,
    ):
        self.config = config
        self.sae_model = sae_model
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.data_loader = data_loader

    def _get_feature_activations(self, feature_indices: List[int]) -> Tuple[Dict[int, List[Tuple[float, torch.Tensor, int, int, Any]]], torch.Tensor, torch.Tensor]:
        """
        Gets all non-zero activations for a list of features.
        Returns a dict mapping feature_idx to a list of (activation_value, input_ids, token_index, sample_index, routing_info),
        all_input_ids tensor, and all_feature_acts tensor for the specified features.
        """
        activations = {idx: [] for idx in feature_indices}
        all_input_ids = []
        all_feature_acts_for_indices = []
        
        self.sae_model.eval()
        with torch.no_grad():
            sample_idx = 0
            for i, batch in enumerate(tqdm(self.data_loader, desc="Getting activations")):
                
                # Determine the correct layer(s) for hidden states
                layer = self.config["model"]["layer"]
                model_type = self.config["model"]["type"]
                if isinstance(self.sae_model, (RouteSAE)):
                    start = self.sae_model.router.start_layer
                    end = self.sae_model.router.end_layer
                    layer = list(range(start, end))

                hidden_states = get_hidden_states(
                    self.language_model,
                    batch,
                    layer,
                    self.config["training"]["device"]
                )
                
                sae_output = self.sae_model(hidden_states)
                feature_acts = sae_output["feature_acts"]

                # Store all activations for the requested features
                all_feature_acts_for_indices.append(feature_acts[:, :, feature_indices].cpu())
                all_input_ids.append(batch["input_ids"].cpu())

                for local_idx, feature_idx in enumerate(feature_indices):
                    # Use the subset of activations we've already gathered
                    feature_acts_for_idx = feature_acts[:, :, feature_idx]
                    non_zero_indices = torch.nonzero(feature_acts_for_idx)
                    
                    for idx_tuple in non_zero_indices:
                        batch_idx, token_idx = idx_tuple[0].item(), idx_tuple[1].item()
                        activation_value = feature_acts_for_idx[batch_idx, token_idx].item()
                        
                        input_ids = batch["input_ids"][batch_idx]
                        global_sample_idx = sample_idx + batch_idx
                        
                        routing_info = None
                        if isinstance(self.sae_model, RouteSAE):
                            routing_type = self.config["model"].get("routing", "hard")
                            if routing_type == 'hard':
                                routing_info = sae_output["target_layer_indices"][batch_idx, token_idx].item()
                            else: # soft
                                routing_info = sae_output["router_weights"][batch_idx, token_idx, :].argmax().item()

                        activations[feature_idx].append((activation_value, input_ids, token_idx, global_sample_idx, routing_info))
                
                sample_idx += batch["input_ids"].shape[0]

        # Sort all feature activation lists
        for idx in feature_indices:
            activations[idx].sort(key=lambda x: x[0], reverse=True)
        
        return activations, torch.cat(all_input_ids, dim=0), torch.cat(all_feature_acts_for_indices, dim=0)

    def _get_quintile_samples(self, activations: List[Tuple[float, torch.Tensor, int, int, Any]], max_examples: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """
        Samples activating examples from quintiles.
        """
        if not activations:
            return {}

        quintile_size = len(activations) // 5
        quintile_weights = [0.4, 0.3, 0.2, 0.1, 0.0]  # Heavier sampling on top quintiles
        
        sampled_examples = {"Quintile " + str(i+1): [] for i in range(5)}
        
        for i in range(5):
            if quintile_weights[i] == 0.0:
                continue

            start_idx = i * quintile_size
            end_idx = (i + 1) * quintile_size if i < 4 else len(activations)
            quintile_group = activations[start_idx:end_idx]

            if quintile_group:
                num_samples = int(max_examples * quintile_weights[i])
                step = max(1, len(quintile_group) // num_samples) if num_samples > 0 else len(quintile_group)
                
                for j in range(0, len(quintile_group), step):
                    activation_value, input_ids, token_idx, sample_idx, routing_info = quintile_group[j]
                    sampled_examples["Quintile " + str(i+1)].append({
                        "activation": activation_value,
                        "input_ids": input_ids,
                        "token_idx": token_idx,
                        "sample_idx": sample_idx,
                        "routing_info": routing_info
                    })
        
        return sampled_examples

    def _generate_html_report(
        self,
        feature_idx: int,
        quintile_samples: Dict[str, List[Dict[str, Any]]],
        all_input_ids: torch.Tensor,
        all_feature_acts: torch.Tensor,
        feature_indices_map: Dict[int, int],
        doll_group: int = None,
        context_window: int = 10
    ) -> str:
        """
        Generates an HTML report for a feature, showing activations for all tokens in context.
        """
        html = f"""
        <html>
            <head>
                <title>Feature {feature_idx} Dashboard</title>
                <style>
                    body {{ font-family: sans-serif; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; }}
                    th {{ background-color: #f2f2f2; }}
                    .context {{ line-height: 1.6; }}
                    .token {{ display: inline-block; padding: 1px 2px; margin: 1px; border-radius: 3px; }}
                    .token-mark {{ background-color: #FFD700 !important; font-weight: bold; }}
                </style>
            </head>
            <body>
                <h1>Feature {feature_idx} Activation Dashboard</h1>
        """
        
        if doll_group is not None:
            html += f"<h2>Matryoshka Doll Group: {doll_group}</h2>"

        routing_info_html = ""
        # Check if routing info exists in the first example
        if quintile_samples and "Quintile 1" in quintile_samples and quintile_samples["Quintile 1"]:
            first_example = quintile_samples["Quintile 1"][0]
            if first_example.get("routing_info") is not None:
                routing_type = self.config["model"].get("routing", "hard")
                layer_idx = first_example["routing_info"]
                start_layer = self.sae_model.router.start_layer
                actual_layer = start_layer + layer_idx
                routing_info_html = f"<h2>Routing Info: Layer {actual_layer} ({routing_type})</h2>"
        html += routing_info_html

        for quintile_name, examples in quintile_samples.items():
            if not examples: continue
            
            html += f"<h2>{quintile_name}</h2>"
            html += "<table><tr><th>Peak Activation</th><th>Context (Activation Heatmap)</th></tr>"
            
            for example in examples:
                input_ids = example["input_ids"]
                token_idx = example["token_idx"]
                activation = example["activation"]
                sample_idx = example["sample_idx"]
                
                start = max(0, token_idx - context_window)
                end = min(len(input_ids), token_idx + context_window + 1)
                
                context_ids = input_ids[start:end]
                
                # Get activations for the context window
                local_feature_idx = feature_indices_map[feature_idx]
                context_activations = all_feature_acts[sample_idx, start:end, local_feature_idx]
                max_context_act = context_activations.max().item()

                context_html = '<div class="context">'
                for i, token_id in enumerate(context_ids):
                    token_str = self.tokenizer.decode(token_id)
                    act_val = context_activations[i].item()
                    
                    # Normalize activation for color intensity (0 to 1)
                    opacity = max(0, min(1, act_val / (max_context_act + 1e-9)))
                    
                    # Yellow with varying opacity based on activation strength
                    color = f"rgba(255, 215, 0, {opacity})"
                    
                    token_class = "token-mark" if (start + i) == token_idx else ""
                    
                    context_html += f'<span class="token {token_class}" style="background-color: {color}" title="Activation: {act_val:.4f}">{token_str}</span>'

                context_html += '</div>'

                html += f"<tr><td>{activation:.4f}</td><td>{context_html}</td></tr>"
                
            html += "</table>"
        
        html += "</body></html>"
        return html

    def _generate_index_page(self, feature_indices: List[int], output_dir: str) -> str:
        """
        Generates the main index page with links to each feature dashboard.
        """
        html = "<h1>SAE Feature Dashboards</h1>"
        html += "<ul>"
        for idx in feature_indices:
            html += f'<li><a href="feature_{idx}_dashboard.html">Feature {idx}</a></li>'
        html += "</ul>"
        
        index_path = os.path.join(output_dir, "index.html")
        with open(index_path, "w") as f:
            f.write(html)
        print(f"Index page saved to: {index_path}")

    def generate_dashboards(self, feature_indices: List[int], output_dir: str = "visualizations", doll_groups: Dict[int, int] = None):
        """
        Generates and saves the feature dashboards for a list of features.
        """
        print(f"Generating dashboards for features: {feature_indices}...")
        
        all_activations, all_input_ids, all_feature_acts = self._get_feature_activations(feature_indices)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        feature_indices_map = {idx: i for i, idx in enumerate(feature_indices)}

        for feature_idx in feature_indices:
            activations = all_activations[feature_idx]
            if not activations:
                print(f"No activations found for feature {feature_idx}.")
                continue
                
            quintile_samples = self._get_quintile_samples(activations)
            
            doll_group = doll_groups.get(feature_idx) if doll_groups else None

            html_content = self._generate_html_report(
                feature_idx, 
                quintile_samples,
                all_input_ids,
                all_feature_acts,
                feature_indices_map,
                doll_group=doll_group
            )
            
            output_path = os.path.join(output_dir, f"feature_{feature_idx}_dashboard.html")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
                
            print(f"Dashboard for feature {feature_idx} saved to: {output_path}")
            
        self._generate_index_page(feature_indices, output_dir)
