import unittest
import torch
from typing import Dict, Any

from sae.models import VanillaSAE, GatedSAE, TopKSAE, JumpReLUSAE
from sae.matryoshka import MatryoshkaSAE
from sae.router import RouteSAE, RouteMatryoshkaSAE


class TestSAEModels(unittest.TestCase):
    """
    Unit tests for SAE model implementations.
    """

    def setUp(self):
        """Set up common test configurations."""
        self.batch_size = 4
        self.seq_len = 10
        self.hidden_size = 128
        self.latent_size = 512
        self.k = 32
        self.n_layers = 12

        self.base_config = {
            "hidden_size": self.hidden_size,
            "latent_size": self.latent_size,
            "k": self.k,
            "n_layers": self.n_layers,
            "threshold": 0.001,
            "bandwidth": 0.001,
            "group_sizes": [64, 64, 64, 64, 64, 64, 64, 64],  # Must sum to latent_size
            "base_sae_type": "TopKSAE"
        }

        # Test input for single-layer SAEs
        self.test_input = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        
        # Test input for multi-layer SAEs (RouteSAE)
        self.test_input_multilayer = torch.randn(
            self.batch_size, self.seq_len, self.n_layers, self.hidden_size
        )

    def _check_sae_output(self, output: Dict[str, torch.Tensor], expected_shape: torch.Size):
        """Helper method to check SAE output format and shapes."""
        self.assertIsInstance(output, dict)
        self.assertIn("sae_out", output)
        self.assertIn("feature_acts", output)
        
        # Check output shapes
        self.assertEqual(output["sae_out"].shape, expected_shape)
        self.assertEqual(
            output["feature_acts"].shape, 
            expected_shape[:-1] + (self.latent_size,)
        )

    def test_vanilla_sae(self):
        """Test VanillaSAE implementation."""
        model = VanillaSAE(self.base_config)
        
        # Test forward pass
        output = model(self.test_input)
        self._check_sae_output(output, self.test_input.shape)
        
        # Test individual encode/decode
        latents = model.encode(self.test_input)
        reconstruction = model.decode(latents)
        
        self.assertEqual(latents.shape, (self.batch_size, self.seq_len, self.latent_size))
        self.assertEqual(reconstruction.shape, self.test_input.shape)

    def test_gated_sae(self):
        """Test GatedSAE implementation."""
        model = GatedSAE(self.base_config)
        
        # Test forward pass
        output = model(self.test_input)
        self._check_sae_output(output, self.test_input.shape)
        
        # Test individual encode/decode
        latents = model.encode(self.test_input)
        reconstruction = model.decode(latents)
        
        self.assertEqual(latents.shape, (self.batch_size, self.seq_len, self.latent_size))
        self.assertEqual(reconstruction.shape, self.test_input.shape)

    def test_topk_sae(self):
        """Test TopKSAE implementation."""
        model = TopKSAE(self.base_config)
        
        # Test forward pass
        output = model(self.test_input)
        self._check_sae_output(output, self.test_input.shape)
        
        # Test sparsity - should have exactly k non-zero elements per sequence position
        latents = output["feature_acts"]
        non_zero_counts = (latents != 0).sum(dim=-1)
        
        # All positions should have exactly k non-zero elements (or less if k > latent_size)
        expected_k = min(self.k, self.latent_size)
        self.assertTrue(torch.all(non_zero_counts <= expected_k))

    def test_jumprelu_sae(self):
        """Test JumpReLUSAE implementation."""
        model = JumpReLUSAE(self.base_config)
        
        # Test forward pass
        output = model(self.test_input)
        self._check_sae_output(output, self.test_input.shape)
        
        # Test individual encode/decode
        latents = model.encode(self.test_input)
        reconstruction = model.decode(latents)
        
        self.assertEqual(latents.shape, (self.batch_size, self.seq_len, self.latent_size))
        self.assertEqual(reconstruction.shape, self.test_input.shape)

    def test_matryoshka_sae(self):
        """Test MatryoshkaSAE implementation."""
        model = MatryoshkaSAE(self.base_config)
        
        # Test forward pass
        output = model(self.test_input)
        self._check_sae_output(output, self.test_input.shape)
        
        # Test sparsity - should have exactly k non-zero elements per sequence position
        latents = output["feature_acts"]
        non_zero_counts = (latents != 0).sum(dim=-1)
        
        # Should have k or fewer non-zero elements
        expected_k = min(self.k, self.latent_size)
        self.assertTrue(torch.all(non_zero_counts <= expected_k))

    def test_route_sae(self):
        """Test RouteSAE implementation."""
        model = RouteSAE(self.base_config)
        
        # Test forward pass with routing parameters
        output = model(self.test_input_multilayer, aggre='sum', routing='hard')
        
        # Check output format
        self.assertIsInstance(output, dict)
        self.assertIn("sae_out", output)
        self.assertIn("feature_acts", output)
        self.assertIn("router_weights", output)
        self.assertIn("sae_input", output)
        
        # Check shapes
        expected_output_shape = (self.batch_size, self.seq_len, self.hidden_size)
        self.assertEqual(output["sae_out"].shape, expected_output_shape)
        self.assertEqual(output["sae_input"].shape, expected_output_shape)
        self.assertEqual(
            output["feature_acts"].shape, 
            (self.batch_size, self.seq_len, self.latent_size)
        )
        
        # Router weights should sum to 1 across the layer dimension
        router_weights = output["router_weights"]
        weight_sums = router_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6))

    def test_route_matryoshka_sae(self):
        """Test RouteMatryoshkaSAE implementation."""
        model = RouteMatryoshkaSAE(self.base_config)
        
        # Test forward pass with routing parameters
        output = model(self.test_input_multilayer, aggre='mean', routing='soft')
        
        # Check output format
        self.assertIsInstance(output, dict)
        self.assertIn("sae_out", output)
        self.assertIn("feature_acts", output)
        self.assertIn("router_weights", output)
        self.assertIn("sae_input", output)
        
        # Check shapes
        expected_output_shape = (self.batch_size, self.seq_len, self.hidden_size)
        self.assertEqual(output["sae_out"].shape, expected_output_shape)
        self.assertEqual(output["sae_input"].shape, expected_output_shape)
        self.assertEqual(
            output["feature_acts"].shape, 
            (self.batch_size, self.seq_len, self.latent_size)
        )

    def test_model_parameters_exist(self):
        """Test that all models have trainable parameters."""
        models_to_test = [
            VanillaSAE(self.base_config),
            GatedSAE(self.base_config),
            TopKSAE(self.base_config),
            JumpReLUSAE(self.base_config),
            MatryoshkaSAE(self.base_config),
            RouteSAE(self.base_config),
            RouteMatryoshkaSAE(self.base_config),
        ]
        
        for model in models_to_test:
            params = list(model.parameters())
            self.assertGreater(len(params), 0, f"{model.__class__.__name__} has no parameters")
            
            # Check that parameters require gradients
            for param in params:
                self.assertTrue(param.requires_grad, 
                              f"Parameter in {model.__class__.__name__} doesn't require gradients")

    def test_device_placement(self):
        """Test that models can be moved to different devices."""
        model = VanillaSAE(self.base_config)
        
        # Test CPU
        model = model.to('cpu')
        output = model(self.test_input.to('cpu'))
        self.assertEqual(output["sae_out"].device.type, 'cpu')
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model = model.to('cuda')
            cuda_input = self.test_input.to('cuda')
            output = model(cuda_input)
            self.assertEqual(output["sae_out"].device.type, 'cuda')


if __name__ == '__main__':
    unittest.main()
