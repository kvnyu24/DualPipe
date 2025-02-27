"""
Adapters for integrating DualPipe with common model architectures.

This module provides pre-built adaptors that make it easier to use DualPipe
with popular model architectures and frameworks like PyTorch, HuggingFace Transformers,
and more.
"""

from typing import List, Optional, Callable, Tuple, Dict, Any, Union
import inspect
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelAdapterBase:
    """Base class for model adapters that help integrate models with DualPipe."""
    
    def __init__(self, model: nn.Module):
        """
        Initialize the adapter with a model.
        
        Args:
            model: The model to adapt for DualPipe.
        """
        self.model = model
        
    def partition(self, num_partitions: int) -> List[nn.Module]:
        """
        Partition the model into the specified number of pipeline stages.
        
        Args:
            num_partitions: Number of partitions to create.
            
        Returns:
            A list of model partitions that can be distributed across the pipeline.
        """
        raise NotImplementedError("Subclasses must implement partition")
    
    def forward_backward_overlap(self, 
                                module0: nn.Module,
                                inputs0: List[torch.Tensor],
                                criterion0: Optional[Callable],
                                labels0: Optional[List[torch.Tensor]],
                                module1: nn.Module,
                                loss1: Optional[torch.Tensor],
                                outputs1: Optional[List[torch.Tensor]],
                                output_grads1: Optional[List[torch.Tensor]]) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Implement custom forward-backward overlap for the specific model architecture.
        
        Args:
            module0: The module for forward pass
            inputs0: Inputs for forward pass
            criterion0: Loss function to use if this is the last stage
            labels0: Labels for the loss function if this is the last stage
            module1: The module for backward pass
            loss1: Loss to backward if this is the last stage
            outputs1: Outputs for backward pass if this is not the last stage
            output_grads1: Output gradients for backward pass if this is not the last stage
            
        Returns:
            Tuple of (outputs from forward pass, loss if last stage else None)
        """
        outputs0 = module0(*inputs0)
        outputs0 = [outputs0] if isinstance(outputs0, torch.Tensor) else outputs0
        if criterion0 is not None:
            loss0 = criterion0(*outputs0, *labels0)
        else:
            loss0 = None

        if loss1 is not None:
            loss1.backward()
            loss1.detach_()
        else:
            from dualpipe.utils import run_backward
            run_backward(outputs1, output_grads1)

        return outputs0, loss0


class TransformerLayerAdapter(ModelAdapterBase):
    """Adapter for transformer models with layer-wise partitioning."""
    
    def __init__(self, model: nn.Module, layer_attr: str = "layers"):
        """
        Initialize the adapter with a transformer model.
        
        Args:
            model: The transformer model to adapt.
            layer_attr: The attribute name that contains the transformer layers.
        """
        super().__init__(model)
        self.layer_attr = layer_attr
        
        # Verify the model has the expected structure
        if not hasattr(model, layer_attr):
            raise ValueError(f"Model does not have a '{layer_attr}' attribute")
        
        self.layers = getattr(model, layer_attr)
        if not isinstance(self.layers, (nn.ModuleList, nn.Sequential)):
            raise ValueError(f"Model's {layer_attr} is not a ModuleList or Sequential")
            
    def partition(self, num_partitions: int) -> List[nn.Module]:
        """
        Partition the transformer model into pipeline stages based on layers.
        
        Args:
            num_partitions: Number of partitions to create.
            
        Returns:
            A list of model partitions.
        """
        num_layers = len(self.layers)
        if num_layers < num_partitions:
            raise ValueError(f"Cannot partition {num_layers} layers into {num_partitions} partitions")
            
        # Calculate layers per partition
        layers_per_partition = num_layers // num_partitions
        remainder = num_layers % num_partitions
        
        partitions = []
        start_idx = 0
        
        # Create a deep copy of the model structure without the layers
        for i in range(num_partitions):
            # Allocate extra layer to early partitions if not evenly divisible
            partition_size = layers_per_partition + (1 if i < remainder else 0)
            end_idx = start_idx + partition_size
            
            # Create a partition with the appropriate layers
            partition = self._create_partition(start_idx, end_idx)
            partitions.append(partition)
            
            start_idx = end_idx
            
        return partitions
    
    def _create_partition(self, start_idx: int, end_idx: int) -> nn.Module:
        """
        Create a partition from a subset of layers.
        
        Args:
            start_idx: Starting layer index (inclusive)
            end_idx: Ending layer index (exclusive)
            
        Returns:
            A module containing the specified layers.
        """
        # Create a wrapper class to properly handle the partition
        class TransformerPartition(nn.Module):
            def __init__(self, parent_model, start, end, layer_attr):
                super().__init__()
                self.parent_model = parent_model
                self.start = start
                self.end = end
                self.layer_attr = layer_attr
                
                # Extract the relevant layers
                self.layers = nn.ModuleList(
                    [getattr(parent_model, layer_attr)[i] for i in range(start, end)]
                )
                
            def forward(self, hidden_states, *args, **kwargs):
                for layer in self.layers:
                    # Handle different transformer implementations
                    if inspect.ismethod(layer.forward) and 'attention_mask' in inspect.signature(layer.forward).parameters:
                        # For HuggingFace-style transformers
                        if 'attention_mask' in kwargs:
                            hidden_states = layer(hidden_states, kwargs['attention_mask'])
                        else:
                            hidden_states = layer(hidden_states)
                    else:
                        # Generic case
                        hidden_states = layer(hidden_states)
                        
                return hidden_states
                
            @classmethod
            def overlaped_forward_backward(
                cls,
                module0,
                inputs0,
                criterion0,
                labels0,
                module1,
                loss1,
                outputs1,
                output_grads1,
            ):
                # Forward pass
                outputs0 = module0(*inputs0)
                outputs0 = [outputs0] if isinstance(outputs0, torch.Tensor) else outputs0
                if criterion0 is not None:
                    loss0 = criterion0(*outputs0, *labels0)
                else:
                    loss0 = None

                # Backward pass
                if loss1 is not None:
                    loss1.backward()
                    loss1.detach_()
                else:
                    from dualpipe.utils import run_backward
                    run_backward(outputs1, output_grads1)

                return outputs0, loss0
            
        return TransformerPartition(self.model, start_idx, end_idx, self.layer_attr)


class VisionModelAdapter(ModelAdapterBase):
    """Adapter for vision models that typically have sequential stages."""
    
    def __init__(self, model: nn.Module):
        """
        Initialize the adapter with a vision model.
        
        Args:
            model: The vision model to adapt.
        """
        super().__init__(model)
        
        # Try to automatically detect model structure
        self.model_structure = self._detect_model_structure()
        
    def _detect_model_structure(self) -> List[Tuple[str, nn.Module]]:
        """
        Detect the structure of a vision model to identify logical partitioning points.
        
        Returns:
            A list of (name, module) tuples representing meaningful components.
        """
        structure = []
        
        # Common patterns in vision models
        candidates = [
            # Look for these patterns in the model
            ('stem', ['conv1', 'bn1', 'stem', 'conv_stem']),
            ('blocks', ['layer1', 'layer2', 'layer3', 'layer4', 'blocks', 'stages']),
            ('head', ['fc', 'classifier', 'head'])
        ]
        
        for section_name, attr_names in candidates:
            for attr in attr_names:
                if hasattr(self.model, attr):
                    module = getattr(self.model, attr)
                    if isinstance(module, nn.Module):
                        structure.append((f"{section_name}_{attr}", module))
        
        # Fall back to modules with parameters if structure detection failed
        if not structure:
            warnings.warn("Could not automatically detect model structure. Falling back to generic partitioning.")
            structure = [(name, module) for name, module in self.model.named_children() 
                        if any(p.requires_grad for p in module.parameters())]
            
        return structure
        
    def partition(self, num_partitions: int) -> List[nn.Module]:
        """
        Partition the vision model into pipeline stages.
        
        Args:
            num_partitions: Number of partitions to create.
            
        Returns:
            A list of model partitions.
        """
        if not self.model_structure:
            raise ValueError("Could not detect model structure for partitioning")
            
        if num_partitions > len(self.model_structure):
            warnings.warn(f"Requested {num_partitions} partitions, but model only has {len(self.model_structure)} logical components. Some partitions will be empty.")
            
        # Create wrapper partitions
        partitions = []
        components_per_partition = max(1, len(self.model_structure) // num_partitions)
        
        for i in range(0, num_partitions):
            start_idx = i * components_per_partition
            end_idx = min((i + 1) * components_per_partition, len(self.model_structure))
            
            # Skip empty partitions
            if start_idx >= len(self.model_structure):
                # Create dummy partition to maintain count
                partition = self._create_empty_partition()
            else:
                partition = self._create_partition(start_idx, end_idx)
                
            partitions.append(partition)
            
        return partitions
    
    def _create_partition(self, start_idx: int, end_idx: int) -> nn.Module:
        """Create a partition from components between start_idx and end_idx."""
        components = self.model_structure[start_idx:end_idx]
        
        class VisionModelPartition(nn.Module):
            def __init__(self, components):
                super().__init__()
                self.components = nn.ModuleList([module for _, module in components])
                self.component_names = [name for name, _ in components]
                
            def forward(self, x):
                for component in self.components:
                    x = component(x)
                return x
                
            @classmethod
            def overlaped_forward_backward(
                cls,
                module0,
                inputs0,
                criterion0,
                labels0,
                module1,
                loss1,
                outputs1,
                output_grads1,
            ):
                # Forward pass
                outputs0 = module0(*inputs0)
                outputs0 = [outputs0] if isinstance(outputs0, torch.Tensor) else outputs0
                if criterion0 is not None:
                    loss0 = criterion0(*outputs0, *labels0)
                else:
                    loss0 = None

                # Backward pass
                if loss1 is not None:
                    loss1.backward()
                    loss1.detach_()
                else:
                    from dualpipe.utils import run_backward
                    run_backward(outputs1, output_grads1)

                return outputs0, loss0
                
        return VisionModelPartition(components)
    
    def _create_empty_partition(self) -> nn.Module:
        """Create an empty partition that simply passes inputs through."""
        class EmptyPartition(nn.Module):
            def forward(self, x):
                return x
                
            @classmethod
            def overlaped_forward_backward(
                cls,
                module0,
                inputs0,
                criterion0,
                labels0,
                module1,
                loss1,
                outputs1,
                output_grads1,
            ):
                outputs0 = [inputs0[0]]  # Identity operation
                if criterion0 is not None:
                    loss0 = criterion0(*outputs0, *labels0)
                else:
                    loss0 = None

                if loss1 is not None:
                    loss1.backward()
                    loss1.detach_()
                else:
                    from dualpipe.utils import run_backward
                    run_backward(outputs1, output_grads1)

                return outputs0, loss0
                
        return EmptyPartition()


def get_model_adapter(model: nn.Module) -> ModelAdapterBase:
    """
    Factory function to get the appropriate adapter for a model.
    
    Args:
        model: The model to create an adapter for.
        
    Returns:
        An instance of a ModelAdapterBase subclass appropriate for the model.
    """
    # Try to detect the model type
    if hasattr(model, 'layers') or hasattr(model, 'encoder') or hasattr(model, 'decoder'):
        # Looks like a transformer
        if hasattr(model, 'layers'):
            return TransformerLayerAdapter(model, 'layers')
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            return TransformerLayerAdapter(model.encoder, 'layers')
        else:
            # Default transformer attributes
            for attr in ['blocks', 'layer']:
                if hasattr(model, attr):
                    return TransformerLayerAdapter(model, attr)
    
    # Vision model detection
    if any(hasattr(model, attr) for attr in ['conv1', 'features', 'backbone']):
        return VisionModelAdapter(model)
    
    # Default to generic adapter
    return ModelAdapterBase(model) 