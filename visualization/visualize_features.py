import argparse
import yaml
from typing import Dict, Any

import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import create_sae_model, load_config
from src.utils import load_language_model
from src.data_loader import get_data_loader
from visualization.plotting import FeatureDashboard
from sae.matryoshka import MatryoshkaSAE
from sae.router import RouteMatryoshkaSAE
import random

def main():
    """
    Main function to generate feature dashboards.
    """
    parser = argparse.ArgumentParser(description="Generate feature dashboards for a trained SAE")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the trained SAE model (.pt file)"
    )
    parser.add_argument(
        "--feature_indices", 
        type=int, 
        nargs="+",
        default=None,
        help="The indices of the features to visualize"
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=100,
        help="Number of random features to visualize if feature_indices is not provided"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5_000,
        help="The number of examples to process to find activations"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Load language model and tokenizer
    print("Loading language model and tokenizer...")
    tokenizer, language_model = load_language_model(config)
    
    # Create SAE model and load state dict
    print(f"Loading SAE model from: {args.model_path}")
    sae_model = create_sae_model(config)
    sae_model.load_state_dict(torch.load(args.model_path))
    sae_model = sae_model.to(dtype=language_model.dtype) # Match the language model's dtype
    sae_model = sae_model.to(config["training"]["device"])
    
    feature_indices = args.feature_indices
    doll_groups = {}

    if feature_indices is None:
        if isinstance(sae_model, (MatryoshkaSAE, RouteMatryoshkaSAE)):
            base_sae = sae_model.sae if isinstance(sae_model, RouteMatryoshkaSAE) else sae_model
            num_dolls = len(base_sae.group_sizes)
            features_per_doll = args.num_features // num_dolls
            feature_indices = []
            for i in range(num_dolls):
                start_idx = base_sae.group_indices[i]
                end_idx = base_sae.group_indices[i+1]
                selected = random.sample(range(start_idx, end_idx), features_per_doll)
                feature_indices.extend(selected)
                for f_idx in selected:
                    doll_groups[f_idx] = i
        else:
            # Randomly select features
            num_latents = sae_model.config["latent_size"]
            feature_indices = random.sample(range(num_latents), args.num_features)

    if isinstance(sae_model, (MatryoshkaSAE, RouteMatryoshkaSAE)):
        base_sae = sae_model.sae if isinstance(sae_model, RouteMatryoshkaSAE) else sae_model
        for f_idx in feature_indices:
            for i in range(len(base_sae.group_sizes)):
                if base_sae.group_indices[i] <= f_idx < base_sae.group_indices[i+1]:
                    doll_groups[f_idx] = i
                    break

    # Create data loader for visualization
    # We'll override the max_samples to a smaller number for efficiency
    vis_config = config.copy()
    vis_config["data"]["max_samples"] = args.num_examples
    vis_config["training"]["batch_size"] = 16 
    print(f"Creating data loader for visualization (processing {args.num_examples} examples)...")
    data_loader = get_data_loader(vis_config, tokenizer)
    
    # Initialize dashboard generator
    dashboard_generator = FeatureDashboard(
        config=config,
        sae_model=sae_model,
        language_model=language_model,
        tokenizer=tokenizer,
        data_loader=data_loader,
    )
    
    # Generate the dashboard
    dashboard_generator.generate_dashboards(feature_indices, doll_groups=doll_groups if doll_groups else None)


if __name__ == "__main__":
    main()
