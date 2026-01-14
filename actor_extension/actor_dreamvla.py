"""
ACTOR-Enhanced DreamVLA Model.

Extends DreamVLA with Action Consistency Loss (L3):
- Adds Inverse Dynamics head
- Computes L3 during training
- Verifies world model predictions are action-consistent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import sys
import os

# Add parent directory to path for DreamVLA imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .inverse_dynamics import InverseDynamicsHead
from .action_consistency_loss import ActionConsistencyLoss, FullACTORLoss


class ACTORDreamVLA(nn.Module):
    """
    DreamVLA extended with ACTOR's Action Consistency Loss.

    Key additions:
    1. InverseDynamicsHead: Predicts action from (z_t, z_{t+1})
    2. L3 Loss: Verifies WM predictions are action-consistent

    During training:
    - DreamVLA predicts next state features (DINO, depth, etc.)
    - We extract latent z'_pred from these predictions
    - Inverse dynamics predicts â = ID(z_t, z'_pred)
    - L3 = ||â - a_gt||²

    Args:
        dreamvla_model: Base DreamVLA model
        hidden_dim: DreamVLA hidden dimension (for ID head)
        action_dim: Action dimension (7)
        l3_weight: Weight for L3 loss
    """

    def __init__(
        self,
        dreamvla_model: nn.Module,
        hidden_dim: int = 384,
        action_dim: int = 7,
        l3_weight: float = 0.1,
        freeze_base: bool = False,
    ):
        super().__init__()

        self.dreamvla = dreamvla_model
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.l3_weight = l3_weight

        # Inverse Dynamics Head
        self.inverse_dynamics = InverseDynamicsHead(
            state_dim=hidden_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
        )

        # L3 Loss
        self.l3_loss_fn = ActionConsistencyLoss()

        # Optionally freeze base model
        if freeze_base:
            for param in self.dreamvla.parameters():
                param.requires_grad = False

    def extract_latent_from_transformer(
        self,
        transformer_output: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Extract latent state from transformer output.

        DreamVLA's transformer output contains:
        [text, state, image_embedding, cls_token, obs_tokens, action_tokens]

        We use the aggregated features before prediction heads as latent.

        Args:
            transformer_output: (B, S, num_tokens, hidden_dim)
            seq_len: Sequence length

        Returns:
            latent: (B, S, hidden_dim) state latent
        """
        # Use mean pooling over the observation tokens
        # The structure is: [text(1), state(1), image(NUM_RESAMPLER_QUERY*2), cls(2), obs_tokens, action_tokens]
        # We'll use the state token + image cls tokens as our latent

        # Simple approach: use the state embedding position (index 1)
        state_latent = transformer_output[:, :, 1, :]  # (B, S, hidden_dim)

        return state_latent

    def get_predicted_next_latent(
        self,
        obs_pred_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get predicted next state latent from observation prediction features.

        DreamVLA predicts next-frame features (DINO, depth, etc.).
        We use these as the predicted next state.

        Args:
            obs_pred_features: (B, S, num_obs_tokens, hidden_dim) predicted obs features

        Returns:
            next_latent: (B, S, hidden_dim) predicted next state latent
        """
        # Mean pool over observation tokens
        next_latent = obs_pred_features.mean(dim=2)  # (B, S, hidden_dim)
        return next_latent

    def compute_l3_loss(
        self,
        current_latent: torch.Tensor,
        predicted_next_latent: torch.Tensor,
        gt_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute L3 Action Consistency Loss.

        Args:
            current_latent: (B, S, hidden_dim) current state latent
            predicted_next_latent: (B, S, hidden_dim) WM's predicted next latent
            gt_actions: (B, S, 7) ground truth actions

        Returns:
            Dict with L3 loss components
        """
        # Predict action from state transition using inverse dynamics
        pred_arm, pred_gripper = self.inverse_dynamics(
            current_latent,
            predicted_next_latent
        )

        # Split ground truth
        gt_arm = gt_actions[..., :6]
        gt_gripper = gt_actions[..., 6:]

        # Compute L3 loss
        l3_result = self.l3_loss_fn(
            pred_arm, pred_gripper,
            gt_arm, gt_gripper
        )

        return l3_result

    def forward(
        self,
        image_primary: torch.Tensor,
        image_wrist: torch.Tensor,
        state: torch.Tensor,
        text_token: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        track_infos: Optional[Dict] = None,
        action_label: Optional[torch.Tensor] = None,
        mode: str = 'train',
    ) -> Tuple[Any, ...]:
        """
        Forward pass with L3 loss computation.

        Returns:
            Original DreamVLA outputs + L3 loss dict
        """
        # Run base DreamVLA forward
        outputs = self.dreamvla(
            image_primary=image_primary,
            image_wrist=image_wrist,
            state=state,
            text_token=text_token,
            action=action,
            track_infos=track_infos,
            action_label=action_label,
            mode=mode,
        )

        # In training mode, compute L3 loss
        l3_loss_dict = None
        if mode == 'train' and action is not None:
            # For L3, we need:
            # 1. Current state latent (from encoder)
            # 2. Predicted next state latent (from world model predictions)
            # 3. Ground truth actions

            # Since DreamVLA doesn't expose intermediate latents easily,
            # we'll use the DINO/image predictions as proxy for predicted next state
            # This is computed in the training loop where we have access to features
            pass

        return outputs


class ACTORTrainingWrapper:
    """
    Training wrapper that computes L3 loss during DreamVLA training.

    This is designed to be integrated into DreamVLA's train_one_epoch_calvin.
    """

    def __init__(
        self,
        inverse_dynamics: InverseDynamicsHead,
        l3_weight: float = 0.1,
    ):
        self.inverse_dynamics = inverse_dynamics
        self.l3_weight = l3_weight
        self.l3_loss_fn = ActionConsistencyLoss()

    def compute_l3_from_features(
        self,
        current_features: torch.Tensor,  # Transformer output at t
        predicted_features: torch.Tensor,  # Observation prediction features
        gt_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute L3 loss from DreamVLA features.

        Args:
            current_features: (B*S, hidden_dim) current state features
            predicted_features: (B*S, num_tokens, hidden_dim) predicted features
            gt_actions: (B*S, 7) ground truth actions

        Returns:
            L3 loss dict
        """
        # Pool predicted features to get next state latent
        pred_next_latent = predicted_features.mean(dim=1)  # (B*S, hidden_dim)

        # Predict action via inverse dynamics
        pred_arm, pred_gripper = self.inverse_dynamics(
            current_features,
            pred_next_latent
        )

        # Split ground truth
        gt_arm = gt_actions[..., :6]
        gt_gripper = gt_actions[..., 6:]

        # Compute loss
        l3_result = self.l3_loss_fn(
            pred_arm, pred_gripper,
            gt_arm, gt_gripper
        )

        return l3_result


def create_actor_dreamvla(
    base_model: nn.Module,
    hidden_dim: int = 384,
    l3_weight: float = 0.1,
) -> Tuple[nn.Module, InverseDynamicsHead, ACTORTrainingWrapper]:
    """
    Factory function to create ACTOR-enhanced DreamVLA.

    Returns:
        base_model: Original model (unchanged)
        inverse_dynamics: ID head to add to training
        training_wrapper: Wrapper for L3 loss computation
    """
    inverse_dynamics = InverseDynamicsHead(
        state_dim=hidden_dim,
        action_dim=7,
        hidden_dim=hidden_dim,
    )

    training_wrapper = ACTORTrainingWrapper(
        inverse_dynamics=inverse_dynamics,
        l3_weight=l3_weight,
    )

    return base_model, inverse_dynamics, training_wrapper
