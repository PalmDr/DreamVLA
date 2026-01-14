"""
Tests for Inverse Dynamics Head.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from actor_extension.inverse_dynamics import InverseDynamicsHead, TransformerInverseDynamics


class TestInverseDynamicsHead:
    """Tests for MLP-based Inverse Dynamics."""

    @pytest.fixture
    def model(self):
        return InverseDynamicsHead(
            state_dim=384,
            action_dim=7,
            hidden_dim=256,
            num_layers=3,
        )

    def test_init(self, model):
        """Test model initialization."""
        assert model.state_dim == 384
        assert model.action_dim == 7

    def test_forward_single(self, model):
        """Test forward pass with single timestep."""
        B = 4
        z_t = torch.randn(B, 384)
        z_next = torch.randn(B, 384)

        arm_action, gripper_action = model(z_t, z_next)

        assert arm_action.shape == (B, 6)
        assert gripper_action.shape == (B, 1)
        # Arm should be in [-1, 1] due to tanh
        assert arm_action.min() >= -1 and arm_action.max() <= 1
        # Gripper should be in [0, 1] due to sigmoid
        assert gripper_action.min() >= 0 and gripper_action.max() <= 1

    def test_forward_sequence(self, model):
        """Test forward pass with sequence."""
        B, S = 4, 10
        z_t = torch.randn(B, S, 384)
        z_next = torch.randn(B, S, 384)

        arm_action, gripper_action = model(z_t, z_next)

        assert arm_action.shape == (B, S, 6)
        assert gripper_action.shape == (B, S, 1)

    def test_predict_action(self, model):
        """Test predict_action helper."""
        B = 4
        z_t = torch.randn(B, 384)
        z_next = torch.randn(B, 384)

        action = model.predict_action(z_t, z_next)

        assert action.shape == (B, 7)

    def test_gradient_flow(self, model):
        """Test gradients flow properly."""
        B = 4
        z_t = torch.randn(B, 384, requires_grad=True)
        z_next = torch.randn(B, 384, requires_grad=True)

        arm_action, gripper_action = model(z_t, z_next)
        loss = arm_action.mean() + gripper_action.mean()
        loss.backward()

        assert z_t.grad is not None
        assert z_next.grad is not None
        assert z_t.grad.abs().sum() > 0

    def test_deterministic(self, model):
        """Test model is deterministic in eval mode."""
        model.eval()
        B = 4
        z_t = torch.randn(B, 384)
        z_next = torch.randn(B, 384)

        with torch.no_grad():
            arm1, gripper1 = model(z_t, z_next)
            arm2, gripper2 = model(z_t, z_next)

        assert torch.allclose(arm1, arm2)
        assert torch.allclose(gripper1, gripper2)


class TestTransformerInverseDynamics:
    """Tests for Transformer-based Inverse Dynamics."""

    @pytest.fixture
    def model(self):
        return TransformerInverseDynamics(
            state_dim=384,
            action_dim=7,
            hidden_dim=256,
            num_heads=4,
            num_layers=2,
        )

    def test_forward(self, model):
        """Test forward pass."""
        B, S = 4, 10
        z_t = torch.randn(B, S, 384)
        z_next = torch.randn(B, S, 384)

        arm_action, gripper_action = model(z_t, z_next)

        assert arm_action.shape == (B, S, 6)
        assert gripper_action.shape == (B, S, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
