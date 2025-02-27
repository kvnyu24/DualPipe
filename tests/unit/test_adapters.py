import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from dualpipe.adapters import (
    ModelAdapterBase,
    TransformerLayerAdapter,
    VisionModelAdapter,
    get_model_adapter
)


class SimpleTransformerModel(nn.Module):
    """A simple transformer model for testing adapters."""
    
    def __init__(self, num_layers=4):
        super().__init__()
        
        # Create a sequence of transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, batch_first=True)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SimpleVisionModel(nn.Module):
    """A simple vision model for testing adapters."""
    
    def __init__(self):
        super().__init__()
        
        # Create a simple CNN
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Define layers in the style of ResNet
        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.fc = nn.Linear(8*8*32, 10)  # Assuming 32x32 input images
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.adaptive_avg_pool2d(x, (8, 8))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class TestModelAdapterBase(unittest.TestCase):
    
    def test_adapter_base_initialization(self):
        """Test that the base adapter initializes correctly."""
        model = nn.Linear(10, 10)
        adapter = ModelAdapterBase(model)
        self.assertEqual(adapter.model, model)
        
        # partition should raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            adapter.partition(num_partitions=2)
    
    def test_forward_backward_overlap(self):
        """Test the forward_backward_overlap method."""
        model = nn.Linear(10, 10)
        adapter = ModelAdapterBase(model)
        
        # Create test inputs
        module0 = nn.Linear(10, 10)
        inputs0 = [torch.randn(2, 10)]
        criterion0 = nn.MSELoss()
        labels0 = [torch.randn(2, 10)]
        
        module1 = nn.Linear(10, 10)
        loss1 = torch.tensor(0.5, requires_grad=True)
        outputs1 = None
        output_grads1 = None
        
        # Test with loss1
        outputs0, loss0 = adapter.forward_backward_overlap(
            module0, inputs0, criterion0, labels0,
            module1, loss1, outputs1, output_grads1
        )
        
        self.assertIsNotNone(outputs0)
        self.assertIsNotNone(loss0)
        
        # Test with outputs1 and output_grads1
        outputs1 = [torch.randn(2, 10, requires_grad=True)]
        output_grads1 = [torch.randn(2, 10)]
        
        outputs0, loss0 = adapter.forward_backward_overlap(
            module0, inputs0, None, [],
            module1, None, outputs1, output_grads1
        )
        
        self.assertIsNotNone(outputs0)
        self.assertIsNone(loss0)


class TestTransformerLayerAdapter(unittest.TestCase):
    
    def setUp(self):
        """Set up a transformer model for testing."""
        self.model = SimpleTransformerModel(num_layers=4)
        self.adapter = TransformerLayerAdapter(self.model)
    
    def test_adapter_initialization(self):
        """Test that the transformer adapter initializes correctly."""
        self.assertEqual(self.adapter.model, self.model)
        self.assertEqual(self.adapter.layer_attr, "layers")
        self.assertEqual(self.adapter.layers, self.model.layers)
    
    def test_invalid_model(self):
        """Test that the adapter raises an error for invalid models."""
        # Model without layers
        model = nn.Linear(10, 10)
        with self.assertRaises(ValueError):
            TransformerLayerAdapter(model)
        
        # Model with invalid layers
        class InvalidModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Linear(10, 10)  # Not a ModuleList or Sequential
                
        model = InvalidModel()
        with self.assertRaises(ValueError):
            TransformerLayerAdapter(model)
    
    def test_partition(self):
        """Test that the adapter can partition the model correctly."""
        # 2 partitions, 4 layers -> 2 layers per partition
        partitions = self.adapter.partition(num_partitions=2)
        self.assertEqual(len(partitions), 2)
        
        # Test forward pass through each partition
        x = torch.randn(2, 10, 128)  # Batch size, sequence length, embedding dimension
        output = partitions[0](x)
        self.assertEqual(output.shape, (2, 10, 128))
        output = partitions[1](output)
        self.assertEqual(output.shape, (2, 10, 128))
        
        # Check that partition has overlaped_forward_backward method
        self.assertTrue(hasattr(type(partitions[0]), 'overlaped_forward_backward'))
        
        # Test with uneven partitioning
        with self.assertRaises(ValueError):
            # Can't partition 4 layers into 5 partitions
            self.adapter.partition(num_partitions=5)


class TestVisionModelAdapter(unittest.TestCase):
    
    def setUp(self):
        """Set up a vision model for testing."""
        self.model = SimpleVisionModel()
        self.adapter = VisionModelAdapter(self.model)
    
    def test_adapter_initialization(self):
        """Test that the vision adapter initializes correctly."""
        self.assertEqual(self.adapter.model, self.model)
        
    def test_detect_model_structure(self):
        """Test that the adapter can detect model structure."""
        structure = self.adapter._detect_model_structure()
        self.assertTrue(len(structure) > 0)
        
        # Check for expected components
        self.assertTrue(any('conv1' in name for name, _ in structure))
        self.assertTrue(any('layer1' in name for name, _ in structure))
        self.assertTrue(any('layer2' in name for name, _ in structure))
        self.assertTrue(any('fc' in name for name, _ in structure))
    
    def test_partition(self):
        """Test that the adapter can partition the model correctly."""
        # Create partitions
        partitions = self.adapter.partition(num_partitions=2)
        self.assertEqual(len(partitions), 2)
        
        # Test forward pass through each partition
        x = torch.randn(2, 3, 32, 32)  # Batch, channels, height, width
        output = partitions[0](x)
        self.assertIsNotNone(output)
        output = partitions[1](output)
        self.assertIsNotNone(output)
        
        # Check that partition has overlapped_forward_backward method
        self.assertTrue(hasattr(type(partitions[0]), 'overlaped_forward_backward'))


class TestModelAdapterFactory(unittest.TestCase):
    
    def test_get_model_adapter_transformer(self):
        """Test that get_model_adapter identifies transformer models."""
        model = SimpleTransformerModel()
        adapter = get_model_adapter(model)
        self.assertIsInstance(adapter, TransformerLayerAdapter)
    
    def test_get_model_adapter_vision(self):
        """Test that get_model_adapter identifies vision models."""
        model = SimpleVisionModel()
        adapter = get_model_adapter(model)
        self.assertIsInstance(adapter, VisionModelAdapter)
    
    def test_get_model_adapter_default(self):
        """Test that get_model_adapter returns base adapter for unknown models."""
        model = nn.Linear(10, 10)
        adapter = get_model_adapter(model)
        self.assertIsInstance(adapter, ModelAdapterBase)


if __name__ == '__main__':
    unittest.main() 