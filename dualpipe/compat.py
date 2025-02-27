"""
Compatibility module for integrating DualPipe with popular frameworks.

This module provides helper utilities for integrating DualPipe with
popular frameworks like PyTorch Lightning, Hugging Face Transformers,
and other common training libraries.
"""

from typing import Optional, List, Dict, Any, Union, Callable, Tuple
import os
import importlib.util
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist

from dualpipe import DualPipe, set_p2p_tensor_shapes, set_p2p_tensor_dtype
from dualpipe.adapters import get_model_adapter
from dualpipe.utils import enable_profiling, get_profiler


# Check if optional dependencies are available
_LIGHTNING_AVAILABLE = importlib.util.find_spec("pytorch_lightning") is not None
_ACCELERATE_AVAILABLE = importlib.util.find_spec("accelerate") is not None
_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None


class DualPipeWrapper(nn.Module):
    """
    A wrapper module that applies DualPipe to a model.
    
    This class provides a standard PyTorch module interface to a
    DualPipe-parallelized model.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        num_partitions: int,
        batch_dim: int = 0,
        process_group: Optional[dist.ProcessGroup] = None,
        auto_setup: bool = True
    ):
        """
        Initialize the DualPipe wrapper.
        
        Args:
            model: The model to parallelize.
            num_partitions: Number of pipeline partitions.
            batch_dim: Batch dimension in the input tensor.
            process_group: PyTorch process group for distributed communication.
            auto_setup: Automatically set up the DualPipe configuration.
        """
        super().__init__()
        self.model = model
        self.num_partitions = num_partitions
        self.batch_dim = batch_dim
        self.process_group = process_group or dist.distributed_c10d._get_default_group()
        
        # Determine local rank
        self.local_rank = self.process_group.rank()
        
        # Get model adapter and create partitions
        self.adapter = get_model_adapter(model)
        self.partitions = self.adapter.partition(num_partitions)
        
        # Move partition to the current device
        self.local_module = self.partitions[self.local_rank].to(f"cuda:{self.local_rank}")
        
        # Create DualPipe
        self.dualpipe = DualPipe(
            [self.local_module, self.local_module],
            batch_dim=batch_dim,
            process_group=process_group
        )
        
        # Auto setup if requested
        if auto_setup:
            self._auto_setup()
            
    def _auto_setup(self):
        """Automatically determine tensor shapes and dtype for communication."""
        # Try to infer tensor shapes from the model architecture
        # This is highly model-dependent, so we will implement a heuristic
        # that works for common model architectures
        
        try:
            # For transformer models, we can guess hidden size
            if hasattr(self.model, 'hidden_size'):
                hidden_size = self.model.hidden_size
                # Guess a reasonable batch_size and seq_len
                batch_size = 1  
                seq_len = 128
                set_p2p_tensor_shapes([(batch_size, seq_len, hidden_size)])
                set_p2p_tensor_dtype(torch.float32)
                warnings.warn(
                    "DualPipeWrapper: Automatically set tensor shapes for a transformer model. "
                    "If this doesn't match your model's actual input shapes, you need to manually "
                    "call set_p2p_tensor_shapes and set_p2p_tensor_dtype."
                )
                return
                
            # For vision models, we can guess feature dimensions
            if hasattr(self.model, 'num_channels'):
                num_channels = self.model.num_channels
                # Guess a reasonable batch_size and spatial dims
                batch_size = 1
                height, width = 224, 224  # Common image size
                set_p2p_tensor_shapes([(batch_size, num_channels, height, width)])
                set_p2p_tensor_dtype(torch.float32)
                warnings.warn(
                    "DualPipeWrapper: Automatically set tensor shapes for a vision model. "
                    "If this doesn't match your model's actual input shapes, you need to manually "
                    "call set_p2p_tensor_shapes and set_p2p_tensor_dtype."
                )
                return
                
        except Exception as e:
            warnings.warn(
                f"DualPipeWrapper: Failed to automatically determine tensor shapes: {str(e)}. "
                "You need to manually call set_p2p_tensor_shapes and set_p2p_tensor_dtype."
            )
            
    def forward(
        self, 
        x: torch.Tensor, 
        *args, 
        num_chunks: int = 8,
        criterion: Optional[Callable] = None,
        labels: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass using DualPipe.
        
        Args:
            x: Input tensor.
            *args: Additional positional arguments.
            num_chunks: Number of micro-batches for pipeline parallelism.
            criterion: Loss function to use.
            labels: Labels for the loss function.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Model output and optionally loss if criterion is provided.
        """
        # Always return outputs
        return_outputs = True
        
        # Set up labels
        if labels is None:
            labels = []
        elif not isinstance(labels, list):
            labels = [labels]
            
        # Do the DualPipe step
        loss, outputs = self.dualpipe.step(
            x, *args,
            num_chunks=num_chunks,
            criterion=criterion,
            labels=labels,
            return_outputs=return_outputs
        )
        
        if criterion is not None and loss is not None:
            return outputs, loss
        return outputs


if _LIGHTNING_AVAILABLE:
    import pytorch_lightning as pl
    
    class DualPipeLightningModule(pl.LightningModule):
        """
        PyTorch Lightning module for DualPipe.
        
        This class provides an integration with PyTorch Lightning for
        easier training of DualPipe models.
        """
        
        def __init__(
            self, 
            model: nn.Module,
            num_partitions: Optional[int] = None,
            num_chunks: int = 8,
            batch_dim: int = 0,
            enable_profiling_during_validation: bool = False,
            *args, 
            **kwargs
        ):
            """
            Initialize the Lightning module.
            
            Args:
                model: The model to parallelize.
                num_partitions: Number of pipeline partitions. If None, will use world_size.
                num_chunks: Number of micro-batches for pipeline parallelism.
                batch_dim: Batch dimension in the input tensor.
                enable_profiling_during_validation: Whether to enable profiling during validation.
                *args, **kwargs: Additional arguments passed to LightningModule.
            """
            super().__init__(*args, **kwargs)
            self.model = model
            self.num_chunks = num_chunks
            self.batch_dim = batch_dim
            self.enable_profiling_during_validation = enable_profiling_during_validation
            self._dualpipe_initialized = False
            self._num_partitions = num_partitions
            
        def setup(self, stage: Optional[str] = None):
            """
            Set up the module for training.
            
            This method is called by Lightning when training begins.
            """
            if self._dualpipe_initialized:
                return
                
            # Get the world size from the trainer
            world_size = self.trainer.world_size
            
            # Set up DualPipe
            num_partitions = self._num_partitions or world_size
            
            # We need an even number of partitions for DualPipe
            if num_partitions % 2 != 0:
                num_partitions -= 1
                warnings.warn(
                    f"DualPipeLightningModule: Adjusted num_partitions to {num_partitions} "
                    "to ensure it's an even number as required by DualPipe."
                )
                
            # Create the DualPipe wrapper
            self.dualpipe_wrapper = DualPipeWrapper(
                self.model,
                num_partitions=num_partitions,
                batch_dim=self.batch_dim,
                auto_setup=False  # We'll set it up manually
            )
            
            # Manually set up tensor shapes
            # This would typically be inferred from the first batch, but we make a best guess
            input_size = self._infer_input_size()
            set_p2p_tensor_shapes([input_size])
            set_p2p_tensor_dtype(torch.float32)
            
            self._dualpipe_initialized = True
            
        def _infer_input_size(self) -> Tuple[int, ...]:
            """Infer the input tensor size based on the model architecture."""
            # This is highly model-dependent, so we implement a heuristic
            # that works for common model architectures
            
            # For transformer models
            if hasattr(self.model, 'hidden_size'):
                hidden_size = self.model.hidden_size
                batch_size = self.trainer.datamodule.batch_size // self.trainer.world_size
                seq_len = 128  # Reasonable default
                return (batch_size, seq_len, hidden_size)
                
            # For vision models
            if hasattr(self.model, 'num_channels'):
                num_channels = self.model.num_channels
                batch_size = self.trainer.datamodule.batch_size // self.trainer.world_size
                height, width = 224, 224  # Common image size
                return (batch_size, num_channels, height, width)
                
            # Default case - we'll need to update this during the first forward pass
            warnings.warn(
                "DualPipeLightningModule: Could not infer input size. Using placeholder values. "
                "You should manually call set_p2p_tensor_shapes and set_p2p_tensor_dtype with "
                "the correct input tensor shapes."
            )
            return (1, 64, 64, 64)  # Placeholder
            
        def forward(self, x, *args, **kwargs):
            """
            Forward pass using DualPipe.
            
            Args:
                x: Input tensor.
                *args, **kwargs: Additional arguments passed to the model.
                
            Returns:
                Model output.
            """
            return self.dualpipe_wrapper(
                x, *args, 
                num_chunks=self.num_chunks,
                **kwargs
            )
            
        def training_step(self, batch, batch_idx):
            """
            Training step using DualPipe.
            
            Args:
                batch: Input batch.
                batch_idx: Batch index.
                
            Returns:
                Loss value.
            """
            # Get inputs and labels from batch
            inputs, labels = self._unpack_batch(batch)
            
            # Forward pass with criterion
            outputs, loss = self.dualpipe_wrapper(
                inputs,
                num_chunks=self.num_chunks,
                criterion=self.criterion,
                labels=[labels]
            )
            
            # Log metrics
            if loss is not None:
                self.log('train_loss', loss.mean().item())
                
            return loss
            
        def validation_step(self, batch, batch_idx):
            """
            Validation step using DualPipe.
            
            Args:
                batch: Input batch.
                batch_idx: Batch index.
                
            Returns:
                Loss value.
            """
            # Enable profiling during validation if requested
            if self.enable_profiling_during_validation and batch_idx == 0:
                enable_profiling(log_dir="./lightning_profiles")
                
            # Get inputs and labels from batch
            inputs, labels = self._unpack_batch(batch)
            
            # Forward pass with criterion
            outputs, loss = self.dualpipe_wrapper(
                inputs,
                num_chunks=self.num_chunks,
                criterion=self.criterion,
                labels=[labels]
            )
            
            # Log metrics
            if loss is not None:
                self.log('val_loss', loss.mean().item())
                
            # Save profiling results
            if self.enable_profiling_during_validation and batch_idx == 0:
                profiler = get_profiler()
                profiler.save_to_file(f"dualpipe_profile_rank{self.trainer.global_rank}.json")
                
            return loss
            
        def _unpack_batch(self, batch):
            """Unpack batch into inputs and labels."""
            if isinstance(batch, tuple) and len(batch) == 2:
                return batch[0], batch[1]
            elif isinstance(batch, dict):
                if 'input_ids' in batch and 'labels' in batch:
                    return batch['input_ids'], batch['labels']
                elif 'inputs' in batch and 'labels' in batch:
                    return batch['inputs'], batch['labels']
                elif 'input' in batch and 'label' in batch:
                    return batch['input'], batch['label']
                elif 'x' in batch and 'y' in batch:
                    return batch['x'], batch['y']
            
            # Default case: assume first element is input, second is label
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                return batch[0], batch[1]
                
            # If we can't unpack, just return the batch and None
            warnings.warn(
                "DualPipeLightningModule: Could not unpack batch into inputs and labels. "
                "Override _unpack_batch method to customize batch unpacking logic."
            )
            return batch, None
            
        def criterion(self, outputs, targets):
            """
            Default criterion function.
            
            Override this method to define your own criterion.
            
            Args:
                outputs: Model outputs.
                targets: Target labels.
                
            Returns:
                Loss value.
            """
            raise NotImplementedError(
                "DualPipeLightningModule: You must implement the criterion method "
                "or pass a criterion function to forward/training_step."
            )


if _TRANSFORMERS_AVAILABLE:
    import transformers
    
    class DualPipeTrainer(transformers.Trainer):
        """
        Extension of HuggingFace Transformers Trainer to use DualPipe.
        
        This class overrides the training loop to use DualPipe for pipeline parallelism.
        """
        
        def __init__(
            self,
            model=None,
            num_partitions: Optional[int] = None,
            num_chunks: int = 8,
            batch_dim: int = 0,
            enable_profiling: bool = False,
            *args,
            **kwargs
        ):
            """
            Initialize the DualPipe trainer.
            
            Args:
                model: The model to train.
                num_partitions: Number of pipeline partitions. If None, will use world_size.
                num_chunks: Number of micro-batches for pipeline parallelism.
                batch_dim: Batch dimension in the input tensor.
                enable_profiling: Whether to enable profiling.
                *args, **kwargs: Additional arguments passed to transformers.Trainer.
            """
            # Initialize with base model
            super().__init__(model=model, *args, **kwargs)
            
            # Store DualPipe-specific parameters
            self.num_chunks = num_chunks
            self.batch_dim = batch_dim
            self.do_enable_profiling = enable_profiling
            self._num_partitions = num_partitions
            self._dualpipe_initialized = False
            
        def _wrap_with_dualpipe(self):
            """Wrap the model with DualPipe."""
            if self._dualpipe_initialized:
                return
                
            # Determine number of partitions
            world_size = self.args.world_size
            num_partitions = self._num_partitions or world_size
            
            # We need an even number of partitions for DualPipe
            if num_partitions % 2 != 0:
                num_partitions -= 1
                warnings.warn(
                    f"DualPipeTrainer: Adjusted num_partitions to {num_partitions} "
                    "to ensure it's an even number as required by DualPipe."
                )
                
            # Create the DualPipe wrapper
            self.model = DualPipeWrapper(
                self.model,
                num_partitions=num_partitions,
                batch_dim=self.batch_dim,
                auto_setup=False  # We'll set it up manually
            )
            
            # Set up tensor shapes based on the model type
            if isinstance(self.model, transformers.PreTrainedModel):
                # For transformer models, use the hidden size and configuration
                config = self.model.config
                hidden_size = config.hidden_size
                
                # Determine batch size
                batch_size = self.args.per_device_train_batch_size
                
                # For sequence models, use sequence length
                seq_length = getattr(config, 'max_position_embeddings', 512)
                
                # Set tensor shapes
                set_p2p_tensor_shapes([(batch_size, seq_length, hidden_size)])
                set_p2p_tensor_dtype(torch.float32)
            else:
                warnings.warn(
                    "DualPipeTrainer: Could not automatically determine tensor shapes. "
                    "You need to manually call set_p2p_tensor_shapes and set_p2p_tensor_dtype."
                )
                
            self._dualpipe_initialized = True
            
        def train(self, *args, **kwargs):
            """
            Train the model with DualPipe.
            
            This method wraps the model with DualPipe before training.
            """
            # Wrap model with DualPipe
            self._wrap_with_dualpipe()
            
            # Enable profiling if requested
            if self.do_enable_profiling:
                enable_profiling(log_dir="./transformers_profiles")
                
            # Call parent's train method
            results = super().train(*args, **kwargs)
            
            # Save profiling results
            if self.do_enable_profiling:
                profiler = get_profiler()
                profiler.save_to_file(f"dualpipe_profile_rank{self.args.local_rank}.json")
                
            return results
            
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            """
            Perform a prediction step using DualPipe.
            
            This method ensures the model is wrapped with DualPipe before prediction.
            """
            # Wrap model with DualPipe if not already done
            if not self._dualpipe_initialized:
                self._wrap_with_dualpipe()
                
            # Call parent's prediction_step method
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys) 