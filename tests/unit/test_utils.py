import unittest
import os
import shutil
import tempfile
import json
import time

import torch

from dualpipe.utils import (
    DualPipeProfiler, 
    enable_profiling, 
    disable_profiling, 
    get_profiler,
    chunk_tensor,
    cat_tensor,
    scatter,
    gather
)


class TestDualPipeProfiler(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        disable_profiling()
        shutil.rmtree(self.test_dir)
    
    def test_profiler_initialization(self):
        """Test that the profiler initializes correctly."""
        profiler = DualPipeProfiler(enabled=True, log_dir=self.test_dir)
        self.assertTrue(profiler.enabled)
        self.assertEqual(profiler.log_dir, self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))
        
    def test_profiler_track_time(self):
        """Test that the profiler can track time for different phases."""
        profiler = DualPipeProfiler(enabled=True)
        
        # Test tracking time for forward compute
        with profiler.track_time("forward_compute"):
            time.sleep(0.01)  # Simulate computation
        
        # Test tracking time for backward compute
        with profiler.track_time("backward_compute"):
            time.sleep(0.01)  # Simulate computation
            
        # Verify timings were recorded
        self.assertEqual(len(profiler.timings["forward_compute"]), 1)
        self.assertEqual(len(profiler.timings["backward_compute"]), 1)
        self.assertTrue(profiler.timings["forward_compute"][0] > 0)
        self.assertTrue(profiler.timings["backward_compute"][0] > 0)
    
    def test_profiler_step_context(self):
        """Test that the step context manager works correctly."""
        profiler = DualPipeProfiler(enabled=True)
        
        with profiler.step_context():
            # Simulate a full DualPipe step
            with profiler.track_time("forward_compute"):
                time.sleep(0.01)
            with profiler.track_time("backward_compute"):
                time.sleep(0.01)
                
        # Verify total step time was recorded
        self.assertEqual(len(profiler.timings["total_step"]), 1)
        self.assertTrue(profiler.timings["total_step"][0] > 0)
    
    def test_profiler_counters(self):
        """Test that the profiler can increment counters."""
        profiler = DualPipeProfiler(enabled=True)
        
        # Increment counters
        profiler.increment_counter("forward_chunks")
        profiler.increment_counter("backward_chunks", 2)
        
        # Verify counters were incremented
        self.assertEqual(profiler.counters["forward_chunks"], 1)
        self.assertEqual(profiler.counters["backward_chunks"], 2)
    
    def test_profiler_summary(self):
        """Test that the profiler can generate a summary."""
        profiler = DualPipeProfiler(enabled=True)
        
        # Simulate a DualPipe step
        with profiler.step_context():
            with profiler.track_time("forward_compute"):
                time.sleep(0.01)
            with profiler.track_time("backward_compute"):
                time.sleep(0.01)
                
        # Get summary
        summary = profiler.get_summary()
        
        # Verify summary contains expected keys
        self.assertIn("avg_time_per_phase_ms", summary)
        self.assertIn("percentage_time_per_phase", summary)
        self.assertIn("total_time_s", summary)
        self.assertIn("step_count", summary)
        self.assertIn("counters", summary)
        
    def test_profiler_save_to_file(self):
        """Test that the profiler can save results to a file."""
        profiler = DualPipeProfiler(enabled=True, log_dir=self.test_dir)
        
        # Simulate a DualPipe step
        with profiler.step_context():
            with profiler.track_time("forward_compute"):
                time.sleep(0.01)
                
        # Save to file
        filepath = profiler.save_to_file("test_profile.json")
        
        # Verify file was created and contains expected data
        self.assertTrue(os.path.exists(filepath))
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.assertIn("avg_time_per_phase_ms", data)
            
    def test_global_profiler_functions(self):
        """Test the global profiler functions."""
        # Test enable_profiling
        profiler = enable_profiling(log_dir=self.test_dir)
        self.assertTrue(profiler.enabled)
        self.assertEqual(profiler.log_dir, self.test_dir)
        
        # Test get_profiler
        current_profiler = get_profiler()
        self.assertEqual(profiler, current_profiler)
        
        # Test disable_profiling
        disable_profiling()
        self.assertFalse(get_profiler().enabled)
        
    def test_tensor_utils(self):
        """Test the tensor utility functions."""
        # Test chunk_tensor
        x = torch.randn(10, 20)
        chunks = chunk_tensor(x, 2, dim=0)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].shape, (5, 20))
        
        # Test chunk_tensor with None
        chunks = chunk_tensor(None, 2, dim=0)
        self.assertEqual(len(chunks), 2)
        self.assertIsNone(chunks[0])
        
        # Test cat_tensor with tensor chunks
        chunks_tensor = [torch.randn(5, 20) for _ in range(2)]
        y = cat_tensor(chunks_tensor, dim=0)
        self.assertEqual(y.shape, (10, 20))
        
        # Test cat_tensor with None
        z = cat_tensor([None, None], dim=0)
        self.assertIsNone(z)
        
        # Test scatter
        inputs = torch.randn(10, 20)
        microbatches = scatter([inputs], 2, dim=0)
        self.assertEqual(len(microbatches), 2)
        self.assertEqual(len(microbatches[0]), 1)
        self.assertEqual(microbatches[0][0].shape, (5, 20))
        
        # Test gather
        outputs = [mb[0] for mb in microbatches]
        gathered = gather(outputs, dim=0)
        self.assertEqual(gathered[0].shape, (10, 20))
        self.assertTrue(torch.allclose(inputs, gathered[0]))


if __name__ == '__main__':
    unittest.main() 