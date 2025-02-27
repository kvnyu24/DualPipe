import unittest
import os
import tempfile
import shutil

import torch
import torch.nn as nn

from dualpipe import DualPipe, set_p2p_tensor_shapes, set_p2p_tensor_dtype
from dualpipe.utils import enable_profiling, disable_profiling, get_profiler
from dualpipe.adapters import TransformerLayerAdapter


class SimpleModule(nn.Module):
    """A simple module for testing DualPipe integration."""
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.linear(x)


class SimpleTransformer(nn.Module):
    """A simple transformer model for integration testing."""
    
    def __init__(self, hidden_size=64, num_layers=4, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(100, hidden_size)
        
        # Create transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)


class TestDualPipeIntegration(unittest.TestCase):
    """Test the integration of DualPipe components."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temp directory for profiles
        self.test_dir = tempfile.mkdtemp()
        
        # Ensure CUDA is available
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
            
        # Move to CUDA device
        torch.cuda.set_device(0)
        
        # Create a simple model
        self.model = SimpleModule(10, 10).cuda()
        
        # Set tensor shapes and dtype
        set_p2p_tensor_shapes([(8, 10)])  # Batch size 8, input dimension 10
        set_p2p_tensor_dtype(torch.float32)
    
    def tearDown(self):
        """Clean up test environment."""
        disable_profiling()
        shutil.rmtree(self.test_dir)
    
    def test_dualpipe_with_profiling(self):
        """Test DualPipe with profiling enabled."""
        # Enable profiling
        profiler = enable_profiling(log_dir=self.test_dir)
        
        # Create DualPipe
        dualpipe = DualPipe([self.model, self.model])
        
        # Create inputs and labels
        inputs = torch.randn(8, 10, device='cuda')
        labels = torch.randn(8, 10, device='cuda')
        
        # Define criterion
        criterion = nn.MSELoss()
        
        # Run model
        loss, _ = dualpipe.step(
            inputs,
            num_chunks=8,
            criterion=criterion,
            labels=(labels,),
            return_outputs=False
        )
        
        # Check that loss was computed
        self.assertIsNotNone(loss)
        
        # Check that profiling recorded data
        summary = profiler.get_summary()
        self.assertIn("total_time_s", summary)
        self.assertIn("avg_time_per_phase_ms", summary)
        
        # Save profile to file
        filepath = profiler.save_to_file("test_profile.json")
        self.assertTrue(os.path.exists(filepath))
    
    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs for this test")
    def test_model_adapter_integration(self):
        """Test the integration of model adapters with DualPipe."""
        # Create a simple transformer model
        model = SimpleTransformer(hidden_size=64, num_layers=4).cuda()
        
        # Create adapter
        adapter = TransformerLayerAdapter(model)
        
        # Partition model - only creating 2 partitions to test on single GPU
        partitions = adapter.partition(num_partitions=2)
        
        # Check that partitions were created correctly
        self.assertEqual(len(partitions), 2)
        
        # Set tensor shapes and dtype
        set_p2p_tensor_shapes([(8, 10, 64)])  # Batch size 8, seq len 10, hidden dim 64
        set_p2p_tensor_dtype(torch.float32)
        
        # Create DualPipe with the same partition on both ranks
        # (this won't train properly but will test the integration)
        local_module = partitions[0].cuda()
        dualpipe = DualPipe([local_module, local_module])
        
        # Create inputs and labels
        inputs = torch.randint(0, 100, (8, 10), device='cuda')
        labels = torch.randn(8, 10, 64, device='cuda')
        
        # Define criterion
        criterion = nn.MSELoss()
        
        # Run model - with profiling
        enable_profiling(log_dir=self.test_dir)
        loss, _ = dualpipe.step(
            inputs,
            num_chunks=8,
            criterion=criterion,
            labels=(labels,),
            return_outputs=False
        )
        
        # Check for profile data
        profiler = get_profiler()
        summary = profiler.get_summary()
        self.assertIn("total_time_s", summary)
        
        # Reset config
        set_p2p_tensor_shapes([])
        set_p2p_tensor_dtype(None)


if __name__ == '__main__':
    unittest.main() 