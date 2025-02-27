import queue
from typing import List, Callable, Dict, Any, Optional
import time
import json
import os
from contextlib import contextmanager

import torch
from torch.autograd import Variable


class WeightGradStore:

    enabled: bool = False
    cache: List[Callable] = []
    funcs_queue = queue.Queue()

    @classmethod
    def put(cls, func: Callable) -> None:
        cls.cache.append(func)

    @classmethod
    def flush(cls) -> None:
        cls.funcs_queue.put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls) -> None:
        assert not cls.funcs_queue.empty(), "Pop empty queue."
        funcs = cls.funcs_queue.get()
        for func in funcs:
            func()

    @classmethod
    def clear(cls) -> None:
        cls.cache = []
        cls.funcs_queue = queue.Queue()


def run_backward(tensors: List[torch.Tensor], grad_tensors: List[torch.Tensor]) -> None:
    kwargs = dict(
        keep_graph=False,
        create_graph=False,
        allow_unreachable=True,
        accumulate_grad=True,
    )
    Variable._execution_engine.run_backward(tuple(tensors), tuple(grad_tensors), **kwargs)


def chunk_tensor(x, chunks, dim):
    if x is None:
        return [None for _ in range(chunks)]
    return x.tensor_split(chunks, dim=dim)


def cat_tensor(x, dim):
    if (isinstance(x, tuple) or isinstance(x, list)):
        if len(x) == 1:
            return x[0]
        elif x[0] is None:
            assert all(y is None for y in x)
            return None
    return torch.cat(x, dim=dim)


def scatter(inputs, chunks, dim):
    assert isinstance(inputs, (torch.Tensor, tuple, list))
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)
    assert all(x is None or isinstance(x, torch.Tensor) for x in inputs)
    inputs = [chunk_tensor(x, chunks, dim) for x in inputs]
    microbatches = [microbatch for microbatch in zip(*inputs)]
    if len(microbatches) == 0:
        microbatches = [() for _ in range(chunks)]
    return microbatches


def gather(micro_outputs, dim):
    assert isinstance(micro_outputs[0], (torch.Tensor, tuple, list))
    if isinstance(micro_outputs[0], torch.Tensor):
        micro_outputs = [(x,) for x in micro_outputs]
    outputs = [x for x in zip(*micro_outputs)]
    outputs = tuple(cat_tensor(x, dim=dim) for x in outputs)
    return outputs


class DualPipeProfiler:
    """
    A profiling utility for DualPipe execution that tracks time spent in each phase
    (forward computation, backward computation, communication) and provides insights
    on performance optimization opportunities.
    """
    
    def __init__(self, enabled: bool = True, log_dir: Optional[str] = None):
        """
        Initialize the profiler.
        
        Args:
            enabled: Whether profiling is enabled.
            log_dir: Directory to save profiling results. If None, results are only kept in memory.
        """
        self.enabled = enabled
        self.log_dir = log_dir
        self.reset()
        
        if self.log_dir and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def reset(self):
        """Reset all profiling data."""
        self.timings = {
            "forward_compute": [],
            "backward_compute": [],
            "forward_communication": [],
            "backward_communication": [],
            "overlapped_compute": [],
            "total_step": []
        }
        self.counters = {
            "forward_chunks": 0,
            "backward_chunks": 0,
            "communication_ops": 0
        }
        self.current_step = None
    
    @contextmanager
    def track_time(self, phase: str):
        """Context manager to track time spent in a specific phase."""
        if not self.enabled:
            yield
            return
            
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            if phase in self.timings:
                self.timings[phase].append(duration)
    
    @contextmanager
    def step_context(self):
        """Context manager for tracking a complete DualPipe step."""
        if not self.enabled:
            yield
            return
            
        start_time = time.time()
        self.current_step = {
            "start_time": start_time,
            "phases": {}
        }
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            self.timings["total_step"].append(duration)
            self.current_step["end_time"] = end_time
            self.current_step["duration"] = duration
            
    def increment_counter(self, counter: str, increment: int = 1):
        """Increment a counter by the specified amount."""
        if not self.enabled:
            return
            
        if counter in self.counters:
            self.counters[counter] += increment
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate a summary of profiling results."""
        if not self.enabled or not self.timings.get("total_step"):
            return {}
            
        summary = {
            "avg_time_per_phase_ms": {},
            "percentage_time_per_phase": {},
            "total_time_s": sum(self.timings["total_step"]),
            "step_count": len(self.timings["total_step"]),
            "counters": self.counters
        }
        
        total_time = sum(self.timings["total_step"])
        for phase, times in self.timings.items():
            if not times:
                continue
                
            avg_time = sum(times) / len(times) * 1000  # convert to ms
            summary["avg_time_per_phase_ms"][phase] = avg_time
            
            if phase != "total_step":
                percentage = sum(times) / total_time * 100 if total_time > 0 else 0
                summary["percentage_time_per_phase"][phase] = percentage
        
        # Calculate potential speedup from better overlap
        compute_time = sum(self.timings.get("forward_compute", [])) + sum(self.timings.get("backward_compute", []))
        communication_time = sum(self.timings.get("forward_communication", [])) + sum(self.timings.get("backward_communication", []))
        overlapped_time = sum(self.timings.get("overlapped_compute", []))
        
        if compute_time > 0 and communication_time > 0:
            min_time = min(compute_time, communication_time)
            if min_time > 0:
                current_overlap_efficiency = (compute_time + communication_time - total_time) / min_time * 100
                summary["current_overlap_efficiency_percent"] = current_overlap_efficiency
        
        return summary
    
    def save_to_file(self, filename: Optional[str] = None):
        """Save profiling results to a file."""
        if not self.enabled or not self.log_dir:
            return
            
        summary = self.get_summary()
        if not summary:
            return
            
        if filename is None:
            filename = f"dualpipe_profile_{int(time.time())}.json"
            
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return filepath
    
    def print_summary(self):
        """Print a summary of profiling results to stdout."""
        if not self.enabled:
            return
            
        summary = self.get_summary()
        if not summary:
            print("No profiling data available.")
            return
            
        print("\n====== DualPipe Profiling Summary ======")
        print(f"Total steps: {summary['step_count']}")
        print(f"Total time: {summary['total_time_s']:.3f}s")
        print("\nAverage time per phase (ms):")
        for phase, time_ms in summary["avg_time_per_phase_ms"].items():
            print(f"  {phase}: {time_ms:.3f}ms ({summary['percentage_time_per_phase'].get(phase, 0):.1f}%)")
        
        print("\nCounters:")
        for counter, value in summary["counters"].items():
            print(f"  {counter}: {value}")
            
        if "current_overlap_efficiency_percent" in summary:
            print(f"\nCurrent compute-communication overlap efficiency: {summary['current_overlap_efficiency_percent']:.1f}%")
            
        print("========================================\n")


# Global profiler instance
profiler = DualPipeProfiler(enabled=False)

def enable_profiling(log_dir: Optional[str] = None):
    """Enable DualPipe profiling."""
    global profiler
    profiler = DualPipeProfiler(enabled=True, log_dir=log_dir)
    return profiler

def disable_profiling():
    """Disable DualPipe profiling."""
    global profiler
    profiler.enabled = False

def get_profiler() -> DualPipeProfiler:
    """Get the global profiler instance."""
    return profiler
