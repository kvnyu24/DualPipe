#!/usr/bin/env python
"""
A simple example test for DualPipe that doesn't require distributed computation.

This example demonstrates:
1. Creating a simple transformer model
2. Using the model adapter to partition it
3. Running inference with DualPipe
4. Using the profiling utility
"""

import os
import tempfile
import shutil
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from dualpipe import (
    DualPipe, 
    set_p2p_tensor_shapes, 
    set_p2p_tensor_dtype,
    enable_profiling,
    get_profiler,
    TransformerLayerAdapter
)


class SimpleTransformerLayer(nn.Module):
    """A simplified transformer layer for demonstration purposes."""
    
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
    def forward(self, x, attention_mask=None):
        # Self-attention
        residual = x
        x = self.norm1(x)
        
        if attention_mask is not None:
            # Convert mask from [batch_size, seq_len] to attention mask
            attn_mask = attention_mask.float().masked_fill(
                attention_mask == 0, float("-inf")
            ).masked_fill(attention_mask == 1, float(0.0))
            x, _ = self.attn(x, x, x, attn_mask=attn_mask)
        else:
            x, _ = self.attn(x, x, x)
            
        x = residual + x
        
        # Feed-forward network
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x


class SimpleTransformer(nn.Module):
    """A simple transformer model for demonstration purposes."""
    
    def __init__(self, vocab_size=1000, hidden_size=128, num_layers=4, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
            
        logits = self.output_proj(x)
        return logits


def main():
    parser = argparse.ArgumentParser(description="Simple DualPipe example")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size of the model")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num-chunks", type=int, default=4, help="Number of micro-batches")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a GPU.")
        return 1
    
    # Create temp directory for profiles if needed
    profile_dir = None
    if args.profile:
        profile_dir = tempfile.mkdtemp()
        
    try:
        # Create model
        model = SimpleTransformer(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers
        ).cuda()
        
        print(f"Created model with {args.num_layers} layers")
        
        # Create adapter and partition model
        adapter = TransformerLayerAdapter(model)
        partitions = adapter.partition(num_partitions=2)
        
        print(f"Partitioned model into 2 parts")
        
        # Create a small batch of random inputs
        batch_size = 4
        seq_length = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_length), device="cuda")
        attention_mask = torch.ones_like(input_ids)
        
        # Set tensor shapes for DualPipe
        set_p2p_tensor_shapes([(batch_size, seq_length, args.hidden_size)])
        set_p2p_tensor_dtype(torch.float32)
        
        # Enable profiling if requested
        if args.profile:
            enable_profiling(log_dir=profile_dir)
            print(f"Profiling enabled, saving to {profile_dir}")
        
        # Create DualPipe with both partitions
        # This is a hack for testing on a single GPU - in practice,
        # you would use different partitions on different GPUs
        dualpipe = DualPipe([partitions[0].cuda(), partitions[0].cuda()])
        
        print("Running inference...")
        
        # Run inference
        with torch.no_grad():
            _, outputs = dualpipe.step(
                input_ids, 
                num_chunks=args.num_chunks,
                return_outputs=True
            )
        
        print(f"Inference complete, output shape: {outputs.shape}")
        
        # Print profiling results if enabled
        if args.profile:
            profiler = get_profiler()
            profiler.print_summary()
            
            # Save profile to file
            filepath = profiler.save_to_file("dualpipe_profile.json")
            print(f"Saved profile to {filepath}")
        
        return 0
        
    finally:
        # Clean up temp directory
        if profile_dir and os.path.exists(profile_dir):
            shutil.rmtree(profile_dir)


if __name__ == "__main__":
    exit(main()) 