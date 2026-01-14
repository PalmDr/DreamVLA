"""
Tests for Action Consistency Loss (L3).
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from actor_extension.action_consistency_loss import ActionConsistencyLoss, FullACTORLoss


class TestActionConsistencyLoss:
    """Tests for L3 Action Consistency Loss."""

    @pytest.fixture
    def loss_fn(self):
        return ActionConsistencyLoss()

    def test_init(self, loss_fn):
        """Test initialization."""
        assert loss_fn.arm_weight == 1.0
        assert loss_fn.gripper_weight == 1.0

    def test_perfect_prediction(self, loss_fn):
        """Test arm loss is zero for identical arm predictions.

        Note: BCE(p, p) = -p*log(p) - (1-p)*log(1-p) is the entropy, not zero.
        So we only check arm loss is zero for identical inputs.
        """
        B, S = 4, 10
        pred_arm = torch.randn(B, S, 6)
        target_arm = pred_arm.clone()

        # For gripper, use deterministic values
        pred_gripper = torch.ones(B, S, 1) * 0.5
        target_gripper = pred_gripper.clone()

        result = loss_fn(
            pred_arm, pred_gripper,
            target_arm, target_gripper
        )

        # Arm loss should be ~0 for identical predictions
        assert result['arm_loss'].item() < 1e-6

    def test_nonzero_loss(self, loss_fn):
        """Test loss is nonzero for different predictions."""
        B, S = 4, 10
        pred_arm = torch.randn(B, S, 6)
        pred_gripper = torch.rand(B, S, 1)
        target_arm = torch.randn(B, S, 6)
        target_gripper = torch.rand(B, S, 1)

        result = loss_fn(
            pred_arm, pred_gripper,
            target_arm, target_gripper
        )

        assert result['loss'].item() > 0
        assert result['arm_loss'].item() > 0
        assert result['gripper_loss'].item() > 0

    def test_with_mask(self, loss_fn):
        """Test loss with mask."""
        B, S = 4, 10
        pred_arm = torch.randn(B, S, 6)
        pred_gripper = torch.rand(B, S, 1)
        target_arm = torch.randn(B, S, 6)
        target_gripper = torch.rand(B, S, 1)

        # Mask out half the sequence
        mask = torch.zeros(B, S)
        mask[:, :5] = 1.0

        result = loss_fn(
            pred_arm, pred_gripper,
            target_arm, target_gripper,
            mask=mask
        )

        assert result['loss'].item() > 0

    def test_gradient_flow(self, loss_fn):
        """Test gradients flow through loss."""
        B, S = 4, 10
        pred_arm = torch.randn(B, S, 6, requires_grad=True)
        pred_gripper = torch.rand(B, S, 1, requires_grad=True)
        target_arm = torch.randn(B, S, 6)
        target_gripper = torch.rand(B, S, 1)

        result = loss_fn(
            pred_arm, pred_gripper,
            target_arm, target_gripper
        )

        result['loss'].backward()

        assert pred_arm.grad is not None
        assert pred_gripper.grad is not None

    def test_loss_decreases_with_closer_prediction(self, loss_fn):
        """Test loss decreases as prediction gets closer to target."""
        B, S = 4, 10
        target_arm = torch.randn(B, S, 6)
        target_gripper = torch.rand(B, S, 1)

        # Far prediction
        pred_arm_far = target_arm + torch.randn_like(target_arm) * 2
        pred_gripper_far = torch.rand(B, S, 1)

        # Close prediction
        pred_arm_close = target_arm + torch.randn_like(target_arm) * 0.1
        pred_gripper_close = target_gripper + torch.randn_like(target_gripper) * 0.1
        pred_gripper_close = pred_gripper_close.clamp(0, 1)

        loss_far = loss_fn(pred_arm_far, pred_gripper_far, target_arm, target_gripper)
        loss_close = loss_fn(pred_arm_close, pred_gripper_close, target_arm, target_gripper)

        assert loss_close['arm_loss'] < loss_far['arm_loss']


class TestFullACTORLoss:
    """Tests for combined ACTOR loss."""

    @pytest.fixture
    def loss_fn(self):
        return FullACTORLoss(l3_weight=1.0, l5_weight=0.1)

    def test_l3_only(self, loss_fn):
        """Test L3 loss computation."""
        B, S = 4, 10
        id_pred_arm = torch.randn(B, S, 6)
        id_pred_gripper = torch.rand(B, S, 1)
        gt_arm = torch.randn(B, S, 6)
        gt_gripper = torch.rand(B, S, 1)

        result = loss_fn(
            id_pred_arm, id_pred_gripper,
            gt_arm, gt_gripper
        )

        assert 'l3_action_consistency' in result
        assert 'total_actor_loss' in result
        assert result['l3_action_consistency'].item() > 0

    def test_with_l5(self, loss_fn):
        """Test with L5 forward verification."""
        B, S = 4, 10
        id_pred_arm = torch.randn(B, S, 6)
        id_pred_gripper = torch.rand(B, S, 1)
        gt_arm = torch.randn(B, S, 6)
        gt_gripper = torch.rand(B, S, 1)
        vla_pred_arm = torch.randn(B, S, 6)
        vla_pred_gripper = torch.rand(B, S, 1)
        id_from_gt_arm = torch.randn(B, S, 6)
        id_from_gt_gripper = torch.rand(B, S, 1)

        result = loss_fn(
            id_pred_arm, id_pred_gripper,
            gt_arm, gt_gripper,
            vla_pred_arm, vla_pred_gripper,
            id_from_gt_arm, id_from_gt_gripper
        )

        assert result['l5_forward_verification'].item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
