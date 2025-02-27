"""
Example of using DualPipe with PyTorch Lightning.

This example demonstrates how to use DualPipe with PyTorch Lightning
for easy integration into Lightning workflows.
"""

import os
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from dualpipe import set_p2p_tensor_shapes, set_p2p_tensor_dtype
from dualpipe.compat import DualPipeLightningModule


class SimpleTransformerLayer(nn.Module):
    """A simplified transformer layer for demonstration purposes."""
    
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Self-attention
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention_out = nn.Linear(hidden_size, hidden_size)
        
        # Feed-forward network
        self.ff1 = nn.Linear(hidden_size, hidden_size * 4)
        self.ff2 = nn.Linear(hidden_size * 4, hidden_size)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        # Self-attention
        residual = x
        x = self.norm1(x)
        
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            # Apply attention mask (1.0 for tokens to attend to, 0.0 for tokens to ignore)
            scores = scores + (attention_mask.unsqueeze(1).unsqueeze(2) - 1.0) * 10000.0
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        attn_output = self.attention_out(context)
        attn_output = self.dropout(attn_output)
        
        # First residual connection
        x = residual + attn_output
        
        # Feed-forward network
        residual = x
        x = self.norm2(x)
        x = self.ff1(x)
        x = F.gelu(x)
        x = self.ff2(x)
        x = self.dropout(x)
        
        # Second residual connection
        x = residual + x
        
        return x


class SimpleTransformer(nn.Module):
    """A simple transformer model for demonstration purposes."""
    
    def __init__(self, vocab_size=30000, hidden_size=768, num_layers=12, num_heads=12, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(512, hidden_size)  # Max sequence length = 512
        
        # Transformer layers
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(hidden_size)
        
        # Output head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Create position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        
        # Combined embeddings
        x = token_embeddings + position_embeddings
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
            
        # Final norm
        x = self.norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        return logits


class SimpleDataset(Dataset):
    """A simple dataset that generates random token IDs and labels."""
    
    def __init__(self, vocab_size, seq_length, size):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Generate random token IDs
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        
        # Generate random labels (next token prediction)
        labels = torch.randint(0, self.vocab_size, (self.seq_length,))
        
        # Create attention mask (all 1s for this simple example)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class SimpleTransformerDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for the SimpleTransformer.
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        seq_length: int = 128,
        train_size: int = 1000,
        val_size: int = 200,
        batch_size: int = 16,
        num_workers: int = 2
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.train_size = train_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        # Create datasets
        self.train_dataset = SimpleDataset(
            vocab_size=self.vocab_size,
            seq_length=self.seq_length,
            size=self.train_size
        )
        
        self.val_dataset = SimpleDataset(
            vocab_size=self.vocab_size,
            seq_length=self.seq_length,
            size=self.val_size
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


class TransformerLightningModule(DualPipeLightningModule):
    """
    PyTorch Lightning module for training a transformer model with DualPipe.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 5e-5,
        *args,
        **kwargs
    ):
        super().__init__(model=model, *args, **kwargs)
        self.learning_rate = learning_rate
        
    def configure_optimizers(self):
        # We need to use local_module instead of model since that's what DualPipe uses
        return torch.optim.Adam(self.dualpipe_wrapper.local_module.parameters(), lr=self.learning_rate)
        
    def criterion(self, outputs, targets):
        """
        Cross-entropy loss for language modeling.
        
        Args:
            outputs: Model logits.
            targets: Target token IDs.
            
        Returns:
            Loss value.
        """
        # Reshape for cross entropy: (batch_size * seq_len, vocab_size)
        logits = outputs.view(-1, outputs.size(-1))
        labels = targets.view(-1)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        return loss


def main():
    parser = argparse.ArgumentParser(description="Train a transformer model with DualPipe and PyTorch Lightning")
    
    # Model parameters
    parser.add_argument("--vocab-size", type=int, default=30000, help="Vocabulary size")
    parser.add_argument("--hidden-size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of attention heads")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--num-chunks", type=int, default=8, help="Number of micro-batches")
    parser.add_argument("--train-size", type=int, default=1000, help="Size of the training dataset")
    parser.add_argument("--val-size", type=int, default=200, help="Size of the validation dataset")
    
    # Other parameters
    parser.add_argument("--num-gpus", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    
    args = parser.parse_args()
    
    # Ensure number of GPUs is even (required by DualPipe)
    assert args.num_gpus % 2 == 0, "Number of GPUs must be even for DualPipe"
    assert args.num_gpus <= torch.cuda.device_count(), f"Requested {args.num_gpus} GPUs but only {torch.cuda.device_count()} available"
    
    # Create the model
    model = SimpleTransformer(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    
    # Create the data module
    datamodule = SimpleTransformerDataModule(
        vocab_size=args.vocab_size,
        seq_length=args.seq_length,
        train_size=args.train_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
    )
    
    # Create the Lightning module
    lightning_module = TransformerLightningModule(
        model=model,
        learning_rate=args.learning_rate,
        num_chunks=args.num_chunks,
        enable_profiling_during_validation=args.profile,
    )
    
    # Create the Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        strategy="ddp",
    )
    
    # Train the model
    trainer.fit(lightning_module, datamodule=datamodule)
    
    # Print final metrics
    metrics = trainer.callback_metrics
    print(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main() 