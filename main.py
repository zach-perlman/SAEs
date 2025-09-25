import argparse
import yaml
from typing import Dict, Any

import torch

from sae import models
from sae.matryoshka import MatryoshkaSAE
from sae.router import RouteSAE, RouteMatryoshkaSAE
from src.utils import load_language_model
from src.data_loader import get_data_loader
from src.train import Trainer
from src.evaluate import Evaluator


def create_sae_model(config: Dict[str, Any]):
    """
    Factory function to create the appropriate SAE model based on the configuration.
    
    Args:
        config (Dict[str, Any]): The configuration dictionary.
        
    Returns:
        BaseSAE: The instantiated SAE model.
    """
    model_type = config["model"]["type"]
    
    model_classes = {
        "VanillaSAE": models.VanillaSAE,
        "GatedSAE": models.GatedSAE,
        "TopKSAE": models.TopKSAE,
        "JumpReLUSAE": models.JumpReLUSAE,
        "MatryoshkaSAE": MatryoshkaSAE,
        "RouteSAE": RouteSAE,
        "RouteMatryoshkaSAE": RouteMatryoshkaSAE,
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(model_classes.keys())}")
    
    return model_classes[model_type](config["model"])


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads the configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        Dict[str, Any]: The configuration dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main():
    """
    Main function that orchestrates the SAE training and evaluation pipeline.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate Sparse Autoencoders")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to the YAML configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Set random seed for reproducibility
    if "seed" in config["training"]:
        torch.manual_seed(config["training"]["seed"])
        
    # Load language model and tokenizer
    print("Loading language model and tokenizer...")
    tokenizer, language_model = load_language_model(config)
    
    # Create SAE model
    print(f"Creating SAE model: {config['model']['type']}")
    sae_model = create_sae_model(config)
    sae_model = sae_model.to(dtype=language_model.dtype)
    
    # Create data loaders
    print("Creating data loaders...")
    train_data_loader = get_data_loader(config, tokenizer, split="train")
    
    # For evaluation, we create a separate dataloader that skips the training samples
    eval_data_loader = get_data_loader(config, tokenizer, split="eval")
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        config=config,
        sae_model=sae_model,
        language_model=language_model,
        tokenizer=tokenizer,
        train_data_loader=train_data_loader
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = Evaluator(
        config=config,
        sae_model=sae_model,
        language_model=language_model,
        data_loader=eval_data_loader
    )
    
    # Evaluate the model
    print("Starting evaluation...")
    eval_metrics = evaluator.evaluate()
    
    print("\nEvaluation Results:")
    for metric, value in eval_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("Training and evaluation completed!")


if __name__ == "__main__":
    main()
