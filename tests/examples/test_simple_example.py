#!/usr/bin/env python
"""
A simple example test for DualPipe that doesn't require distributed computation.

This example demonstrates:
1. Creating a simple model
2. Running inference with profiling
"""

import os
import tempfile
import shutil
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from dualpipe.utils import enable_profiling, get_profiler


class SimpleModel(nn.Module):
    """A simple model for testing."""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def main():
    parser = argparse.ArgumentParser(description="Simple test example")
    parser.add_argument("--input-size", type=int, default=10, help="Input dimension")
    parser.add_argument("--hidden-size", type=int, default=20, help="Hidden dimension")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if CUDA is available")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print(f"Using device: {device}")
    
    # Create temp directory for profiles if needed
    profile_dir = None
    if args.profile:
        profile_dir = tempfile.mkdtemp()
        
    try:
        # Enable profiling if requested
        if args.profile:
            profiler = enable_profiling(log_dir=profile_dir)
            print(f"Profiling enabled, saving to {profile_dir}")
            
        # Create model
        model = SimpleModel(args.input_size, args.hidden_size).to(device)
        print(f"Created model with input size {args.input_size} and hidden size {args.hidden_size}")
        
        # Create random input
        inputs = torch.randn(args.batch_size, args.input_size, device=device)
        
        # Define loss function
        criterion = nn.MSELoss()
        
        print("Running forward pass...")
        
        # Use the profiler context manager to track time
        with profiler.step_context():
            # Forward pass with profiling
            with profiler.track_time("forward_compute"):
                outputs = model(inputs)
                
            # Backward pass with profiling
            with profiler.track_time("backward_compute"):
                # Calculate loss (using inputs as targets for simplicity)
                loss = criterion(outputs, inputs)
                loss.backward()
                
            # Increment counters for statistics
            profiler.increment_counter("forward_chunks")
            profiler.increment_counter("backward_chunks")
        
        print(f"Forward and backward pass complete. Loss: {loss.item():.4f}")
        
        # Print profiling results if enabled
        if args.profile:
            profiler.print_summary()
            
            # Save profile to file
            filepath = profiler.save_to_file("simple_test_profile.json")
            print(f"Saved profile to {filepath}")
        
        return 0
        
    finally:
        # Clean up temp directory
        if profile_dir and os.path.exists(profile_dir):
            shutil.rmtree(profile_dir)


if __name__ == "__main__":
    exit(main()) 