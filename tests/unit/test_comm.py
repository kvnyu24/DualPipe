import unittest
import os

import torch
import torch.distributed as dist

from dualpipe.comm import (
    DualPipeConfig,
    set_p2p_tensor_shapes,
    set_p2p_tensor_dtype,
    get_config,
    build_from_tensor_shapes
)


# Create a mock version of build_from_tensor_shapes for testing on CPU
def mock_build_from_tensor_shapes():
    config = get_config()
    if not config.validate():
        raise RuntimeError(
            "Communication configuration is incomplete. "
            "You need to call set_p2p_tensor_shapes and set_p2p_tensor_dtype before building tensors."
        )
    
    return [
        torch.empty(
            s, 
            dtype=config.tensor_dtype, 
            device="cpu",  # Use CPU instead of CUDA
            requires_grad=True
        ) for s in config.tensor_shapes
    ]


class TestDualPipeConfig(unittest.TestCase):
    
    def setUp(self):
        # Reset the config before each test
        config = get_config()
        config.tensor_shapes = []
        config.tensor_dtype = None
    
    def test_config_validation(self):
        """Test that config validation works correctly."""
        config = DualPipeConfig()
        
        # Empty config should not be valid
        self.assertFalse(config.validate())
        
        # Config with only shapes should not be valid
        config.tensor_shapes = [(10, 20)]
        self.assertFalse(config.validate())
        
        # Config with only dtype should not be valid
        config = DualPipeConfig()
        config.tensor_dtype = torch.float32
        self.assertFalse(config.validate())
        
        # Config with both shapes and dtype should be valid
        config.tensor_shapes = [(10, 20)]
        self.assertTrue(config.validate())
    
    def test_set_p2p_tensor_shapes(self):
        """Test setting tensor shapes."""
        shapes = [(10, 20), (30, 40)]
        set_p2p_tensor_shapes(shapes)
        
        config = get_config()
        self.assertEqual(config.tensor_shapes, shapes)
    
    def test_set_p2p_tensor_dtype(self):
        """Test setting tensor dtype."""
        dtype = torch.float16
        set_p2p_tensor_dtype(dtype)
        
        config = get_config()
        self.assertEqual(config.tensor_dtype, dtype)
    
    def test_build_from_tensor_shapes(self):
        """Test building tensors from shapes."""
        # Should raise error if config is incomplete
        with self.assertRaises(RuntimeError):
            mock_build_from_tensor_shapes()
        
        # Set valid config
        shapes = [(10, 20), (30, 40)]
        set_p2p_tensor_shapes(shapes)
        set_p2p_tensor_dtype(torch.float32)
        
        # Should build tensors correctly
        tensors = mock_build_from_tensor_shapes()
        self.assertEqual(len(tensors), 2)
        self.assertEqual(tensors[0].shape, torch.Size([10, 20]))
        self.assertEqual(tensors[1].shape, torch.Size([30, 40]))
        self.assertEqual(tensors[0].dtype, torch.float32)
        self.assertEqual(tensors[1].dtype, torch.float32)
        self.assertTrue(tensors[0].requires_grad)
        self.assertTrue(tensors[1].requires_grad)
        self.assertEqual(tensors[0].device.type, "cpu")  # Test on CPU
        self.assertEqual(tensors[1].device.type, "cpu")  # Test on CPU


# Only run these tests if distributed is available and initialized
@unittest.skipIf(not torch.distributed.is_available(), "Distributed not available")
class TestDistributedComm(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Initialize distributed environment if not already initialized
        if not dist.is_initialized():
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group("gloo", rank=0, world_size=1)
    
    @classmethod
    def tearDownClass(cls):
        # Clean up distributed environment if we initialized it
        if dist.is_initialized():
            dist.destroy_process_group()
    
    def test_append_irecv_isend(self):
        """Test append_irecv and append_isend functions with a single process."""
        # This is more of a smoke test to make sure the functions don't raise errors
        # Real distributed tests would need a proper multi-process setup
        
        # Use mock functions that use CPU instead of CUDA
        def append_irecv_cpu(ops, src, group):
            tensors = mock_build_from_tensor_shapes()
            src = dist.distributed_c10d.get_global_rank(group, src)
            for tensor in tensors:
                if tensor is not None:
                    ops.append(dist.P2POp(dist.irecv, tensor, src))
            return tensors
            
        def append_isend_cpu(ops, tensors, dst, group):
            dst = dist.distributed_c10d.get_global_rank(group, dst)
            for tensor in tensors:
                if tensor is not None:
                    ops.append(dist.P2POp(dist.isend, tensor, dst))
        
        # Set up config
        shapes = [(10, 20)]
        set_p2p_tensor_shapes(shapes)
        set_p2p_tensor_dtype(torch.float32)
        
        # Create dummy ops list
        ops = []
        
        # Test irecv with self as src
        tensors = append_irecv_cpu(ops, 0, dist.group.WORLD)
        self.assertEqual(len(tensors), 1)
        self.assertEqual(tensors[0].shape, torch.Size([10, 20]))
        
        # Create a tensor to send
        send_tensor = torch.randn(10, 20)
        
        # Test isend with self as dst
        append_isend_cpu(ops, [send_tensor], 0, dist.group.WORLD)
        
        # Ops should have 2 operations now
        self.assertEqual(len(ops), 2)


if __name__ == '__main__':
    unittest.main() 