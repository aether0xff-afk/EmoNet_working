"""EmoNet v7 adaptive sparse RSNN core."""

from .adaptive_rsnn import AdaptiveSparseRSNN, SNNState, TickTrace, create_recurrent_mask
from .event_encoder import EventEncoder
from .schemas import Event
from .surrogate import spike_with_surrogate_gradient
from .trace_encoder import TraceEncoder
from .training_window import DifferentiableWindow, run_differentiable_window

__all__ = [
    "AdaptiveSparseRSNN",
    "SNNState",
    "TickTrace",
    "create_recurrent_mask",
    "EventEncoder",
    "Event",
    "TraceEncoder",
    "DifferentiableWindow",
    "run_differentiable_window",
    "spike_with_surrogate_gradient",
]
