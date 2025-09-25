# Sparse Autoencoders (SAEs)

A clean, modular, and extensible codebase for training and evaluating Sparse Autoencoders on language models. This repository supports multiple SAE architectures including Vanilla, Gated, TopK, JumpReLU, RouteSAE, MatryoshkaSAE, and RouteMatryoshkaSAE.

## Features

- **Multiple SAE Architectures**: Support for various SAE types
- **Unified Interface**: All models inherit from a common `BaseSAE` class
- **Flexible Configuration**: YAML-based configuration system
- **Streaming Data**: Efficient data loading with HuggingFace datasets
- **Training & Evaluation**: Complete pipeline with metrics and visualization
- **Experiment Tracking**: Integration with Weights & Biases (wandb)
- **Modular Design**: Easy to extend with new SAE types

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SAEs
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Training

Train a TopK SAE with default settings:

```bash
python main.py --config configs/default.yaml
```

### Custom Configuration

Create a custom configuration file or override the default:

```bash
python main.py --config configs/my_experiment.yaml
```

## Supported SAE Models

### Standard SAEs
- **VanillaSAE**: Basic sparse autoencoder with ReLU activation
- **GatedSAE**: Uses gating mechanism for improved feature learning
- **TopKSAE**: Enforces exact k-sparse activations
- **JumpReLUSAE**: Uses learnable threshold activation function

### Advanced SAEs
- **MatryoshkaSAE**: Nested feature dictionaries of increasing size
- **RouteSAE**: Routes between multiple layers using a learned router
- **RouteMatryoshkaSAE**: Combines routing with Matryoshka architecture

## Configuration

The configuration system uses YAML files with the following structure:

```yaml
model:
  type: TopKSAE  # Model type
  hidden_size: 2048
  latent_size: 65536
  k: 50  # For TopK models
  language_model_path: "Qwen/Qwen3-1.7B"

training:
  batch_size: 128
  lr: 0.0005
  num_epochs: 1
  device: "cuda:0"
  l1_coefficient: 0.001

data:
  train_path: "openbmb/ultra-fineweb"
  dataset_split: "train"
  streaming: true
  max_samples: 10000000

wandb:
  use_wandb: false
  project: "sae-experiments"
```

### Model-Specific Parameters

#### TopKSAE
```yaml
model:
  type: TopKSAE
  k: 50  # Number of active features
```

#### JumpReLUSAE
```yaml
model:
  type: JumpReLUSAE
  threshold: 0.001  # Activation threshold
  bandwidth: 0.001  # Gradient smoothing
```

#### MatryoshkaSAE
```yaml
model:
  type: MatryoshkaSAE
  group_sizes: [8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192]
  k: 50
```

#### RouteSAE
```yaml
model:
  type: RouteSAE
  base_sae_type: TopKSAE  # Underlying SAE type
  n_layers: 28
  aggre: 'sum'  # 'sum' or 'mean'
  routing: 'hard'  # 'hard' or 'soft'
```

## Project Structure

```
SAEs/
├── configs/                 # Configuration files
│   └── default.yaml        # Default configuration
├── sae/                    # SAE model implementations
│   ├── __init__.py
│   ├── core.py            # Base SAE class
│   ├── models.py          # Standard SAE models
│   ├── router.py          # Routing SAE models
│   └── matryoshka.py      # Matryoshka SAE
├── src/                   # Core functionality
│   ├── __init__.py
│   ├── data_loader.py     # Data loading utilities
│   ├── train.py           # Training logic
│   ├── evaluate.py        # Evaluation logic
│   └── utils.py           # Helper functions
├── tests/                 # Unit tests
│   ├── __init__.py
│   └── test_models.py     # Model tests
├── visualization/         # Plotting and visualization
│   ├── __init__.py
│   └── plotting.py        # Plotting functions
├── main.py               # Main entry point
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Architecture Details

### BaseSAE

All SAE models inherit from the `BaseSAE` abstract class, which defines the common interface:

```python
class BaseSAE(nn.Module, ABC):
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse representation"""
        pass
    
    @abstractmethod  
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode sparse representation back to input space"""
        pass
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return dictionary with sae_out and feature_acts"""
        pass
```

### Loss Function

The training uses a combination of reconstruction loss and L1 sparsity penalty:

```
Loss = MSE(original, reconstructed) + λ * L1(activations)
```

Where λ is controlled by the `l1_coefficient` parameter.

## Data Loading

The data loader supports:
- Streaming datasets from HuggingFace
- Multiple dataset formats (ultra-fineweb, mathinstruct, etc.)
- Automatic text field detection
- Configurable tokenization and sequence length

## Evaluation Metrics

The evaluation pipeline computes:
- Reconstruction loss (MSE)
- L1 sparsity loss  
- Total loss
- Variance explained
- Feature activation statistics

## Visualization

Basic visualization functions are provided:
- Loss curves over training
- Feature activation histograms
- Integration with wandb for experiment tracking

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Or run individual test files:

```bash
python -m unittest tests.test_models -v
```

## Extending the Framework

### Adding a New SAE Model

1. Create a new class inheriting from `BaseSAE`
2. Implement the `encode` and `decode` methods
3. Add the model to the factory function in `main.py`
4. Add configuration parameters to the default config
5. Write unit tests for the new model

Example:

```python
class CustomSAE(BaseSAE):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Initialize your model components
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Your encoding logic
        pass
        
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        # Your decoding logic  
        pass
```

### Adding New Metrics

Extend the `Evaluator` class in `src/evaluate.py` to compute additional metrics during evaluation.

### Custom Data Sources

Modify the `_StreamingDataset` class in `src/data_loader.py` to support new data formats or sources.

## Performance Considerations

- Use `torch.bfloat16` for memory efficiency with large models
- Enable streaming for large datasets to avoid memory issues  
- Adjust `batch_size` and `num_workers` based on your hardware
- Use appropriate `device_map` settings for multi-GPU setups