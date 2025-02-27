"""
Communication module for DualPipe.

This module handles the point-to-point communication between pipeline stages
in the DualPipe algorithm.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


@dataclass
class DualPipeConfig:
    """Configuration for DualPipe communication.
    
    This class holds the configuration for tensor shapes and datatypes used in
    point-to-point communication.
    """
    tensor_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    tensor_dtype: Optional[torch.dtype] = None
    
    def validate(self) -> bool:
        """
        Validate that the configuration is complete and valid.
        
        Returns:
            bool: True if the configuration is valid
        """
        return len(self.tensor_shapes) > 0 and self.tensor_dtype is not None


# Global configuration instance
_config = DualPipeConfig()


def set_p2p_tensor_shapes(shapes: List[Tuple[int, ...]]) -> None:
    """
    Set the tensor shapes for point-to-point communication.
    
    Args:
        shapes: List of tensor shapes that will be used in communication.
    """
    global _config
    _config.tensor_shapes = shapes


def set_p2p_tensor_dtype(dtype: torch.dtype) -> None:
    """
    Set the tensor dtype for point-to-point communication.
    
    Args:
        dtype: The data type for tensors used in communication.
    """
    global _config
    _config.tensor_dtype = dtype


def get_config() -> DualPipeConfig:
    """
    Get the current DualPipe configuration.
    
    Returns:
        The current configuration object.
    """
    return _config


def build_from_tensor_shapes() -> List[torch.Tensor]:
    """
    Build tensors based on the configured shapes and dtype.
    
    Returns:
        List of empty tensors with the configured shapes and dtype.
    """
    if not _config.validate():
        raise RuntimeError(
            "Communication configuration is incomplete. "
            "You need to call set_p2p_tensor_shapes and set_p2p_tensor_dtype before building tensors."
        )
    
    return [
        torch.empty(
            s, 
            dtype=_config.tensor_dtype, 
            device="cuda", 
            requires_grad=True
        ) for s in _config.tensor_shapes
    ]


def append_irecv(ops: List[dist.P2POp], src: int, group: dist.ProcessGroup) -> List[torch.Tensor]:
    """
    Append an irecv operation to the list of communication operations.
    
    Args:
        ops: List of communication operations to append to.
        src: Source rank to receive from.
        group: Process group for communication.
        
    Returns:
        List of tensors that will receive the data.
    """
    tensors = build_from_tensor_shapes()
    src = dist.distributed_c10d.get_global_rank(group, src)
    for tensor in tensors:
        if tensor is not None:
            ops.append(dist.P2POp(dist.irecv, tensor, src))
    return tensors


def append_isend(ops: List[dist.P2POp], tensors: List[torch.Tensor], dst: int, group: dist.ProcessGroup) -> None:
    """
    Append an isend operation to the list of communication operations.
    
    Args:
        ops: List of communication operations to append to.
        tensors: Tensors to send.
        dst: Destination rank to send to.
        group: Process group for communication.
    """
    dst = dist.distributed_c10d.get_global_rank(group, dst)
    for tensor in tensors:
        if tensor is not None:
            ops.append(dist.P2POp(dist.isend, tensor, dst))
