"""
Action Consistency Loss (L3) - ACTOR's Core Contribution.

If the world model's prediction is realistic, inverse dynamics
should be able to recover the original action.

L3 = ||ID(z_t, WM(z_t, a)) - a||²

This provides a self-consistency check that verifies world model
predictions are physically plausible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ActionConsistencyLoss(nn.Module):
    """
    Action Consistency Loss (L3) from ACTOR.

    Given:
    - z_t: current state latent
    - a: ground truth action
    - z'_pred: world model's predicted next state

    Compute:
    - â = ID(z_t, z'_pred): inverse dynamics prediction
    - L3 = ||â - a||²

    This loss ensures that world model predictions are action-consistent:
    if the prediction is realistic, we should be able to recover the action.

    Args:
        arm_weight: Weight for arm action loss
        gripper_weight: Weight for gripper action loss
        use_smooth_l1: Use smooth L1 loss instead of MSE
    """

    def __init__(
        self,
        arm_weight: float = 1.0,
        gripper_weight: float = 1.0,
        use_smooth_l1: bool = True,
    ):
        super().__init__()

        self.arm_weight = arm_weight
        self.gripper_weight = gripper_weight
        self.use_smooth_l1 = use_smooth_l1

    def forward(
        self,
        pred_arm_action: torch.Tensor,
        pred_gripper_action: torch.Tensor,
        target_arm_action: torch.Tensor,
        target_gripper_action: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute action consistency loss.

        Args:
            pred_arm_action: (B, S, 6) ID's predicted arm action from WM output
            pred_gripper_action: (B, S, 1) ID's predicted gripper action
            target_arm_action: (B, S, 6) ground truth arm action
            target_gripper_action: (B, S, 1) ground truth gripper action
            mask: Optional (B, S) mask for valid timesteps

        Returns:
            Dict with 'loss', 'arm_loss', 'gripper_loss'
        """
        # Arm action loss
        if self.use_smooth_l1:
            arm_loss = F.smooth_l1_loss(pred_arm_action, target_arm_action, reduction='none')
        else:
            arm_loss = F.mse_loss(pred_arm_action, target_arm_action, reduction='none')

        # Gripper action loss (binary cross entropy)
        gripper_loss = F.binary_cross_entropy(
            pred_gripper_action,
            target_gripper_action,
            reduction='none'
        )

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (B, S, 1)
            arm_loss = arm_loss * mask
            gripper_loss = gripper_loss * mask

            # Mean over valid entries
            num_valid = mask.sum().clamp(min=1)
            arm_loss = arm_loss.sum() / (num_valid * 6)
            gripper_loss = gripper_loss.sum() / num_valid
        else:
            arm_loss = arm_loss.mean()
            gripper_loss = gripper_loss.mean()

        # Combined loss
        total_loss = self.arm_weight * arm_loss + self.gripper_weight * gripper_loss

        return {
            'loss': total_loss,
            'arm_loss': arm_loss,
            'gripper_loss': gripper_loss,
        }


class FullACTORLoss(nn.Module):
    """
    Full ACTOR Loss combining all six training signals.

    L1: Imagination Loss (WM → VLA) - Train VLA on WM rollouts
    L2: Action Feasibility (VLA → WM) - Entropy consistency
    L3: Action Consistency (WM → ID) - Core contribution [MAIN]
    L4: Prediction Error (ID → WM) - Gradient signal back to WM
    L5: Forward Verification (VLA → ID) - VLA and ID should agree
    L6: Action Prior (ID → VLA) - KL regularization

    For DreamVLA integration, we focus on L3 as the key addition.
    """

    def __init__(
        self,
        l3_weight: float = 1.0,  # Action Consistency (main)
        l5_weight: float = 0.1,  # Forward Verification
        use_l5: bool = True,
    ):
        super().__init__()

        self.l3_weight = l3_weight
        self.l5_weight = l5_weight
        self.use_l5 = use_l5

        self.l3_loss = ActionConsistencyLoss()

    def forward(
        self,
        # L3 inputs: Action Consistency
        id_pred_arm: torch.Tensor,  # ID(z_t, z'_pred)
        id_pred_gripper: torch.Tensor,
        gt_arm: torch.Tensor,  # Ground truth action
        gt_gripper: torch.Tensor,
        # L5 inputs: Forward Verification (optional)
        vla_pred_arm: Optional[torch.Tensor] = None,
        vla_pred_gripper: Optional[torch.Tensor] = None,
        id_from_gt_arm: Optional[torch.Tensor] = None,  # ID(z_t, z_{t+1}_real)
        id_from_gt_gripper: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute full ACTOR loss.

        Returns:
            Dict with all loss components
        """
        losses = {}

        # L3: Action Consistency (CORE)
        # If WM predicts z', can ID recover action a?
        l3_result = self.l3_loss(
            id_pred_arm, id_pred_gripper,
            gt_arm, gt_gripper,
            mask
        )
        losses['l3_action_consistency'] = l3_result['loss']
        losses['l3_arm'] = l3_result['arm_loss']
        losses['l3_gripper'] = l3_result['gripper_loss']

        # L5: Forward Verification (optional)
        # VLA's action should match what ID predicts from real transitions
        if self.use_l5 and vla_pred_arm is not None and id_from_gt_arm is not None:
            l5_result = self.l3_loss(  # Reuse same loss function
                vla_pred_arm, vla_pred_gripper,
                id_from_gt_arm.detach(), id_from_gt_gripper.detach(),
                mask
            )
            losses['l5_forward_verification'] = l5_result['loss']
        else:
            losses['l5_forward_verification'] = torch.tensor(0.0, device=gt_arm.device)

        # Total loss
        total = (
            self.l3_weight * losses['l3_action_consistency'] +
            self.l5_weight * losses['l5_forward_verification']
        )
        losses['total_actor_loss'] = total

        return losses
