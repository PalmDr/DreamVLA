"""
ACTOR Extension for DreamVLA.

Adds Action Consistency Loss (L3) to DreamVLA's training:
- If world model predicts next state z', inverse dynamics should recover original action
- This provides a self-consistency check that DreamVLA lacks

Key Components:
- InverseDynamicsHead: Predicts action from (z_t, z_{t+1})
- ActionConsistencyLoss: L3 = ||ID(z_t, z'_pred) - a||²
- ACTORDreamVLA: Extended model with L3 loss
"""

from .inverse_dynamics import InverseDynamicsHead
from .action_consistency_loss import ActionConsistencyLoss
from .actor_dreamvla import ACTORDreamVLA

__all__ = [
    "InverseDynamicsHead",
    "ActionConsistencyLoss",
    "ACTORDreamVLA",
]
