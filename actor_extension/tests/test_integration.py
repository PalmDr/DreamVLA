"""
Integration tests for ACTOR + DreamVLA.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from actor_extension.inverse_dynamics import InverseDynamicsHead
from actor_extension.action_consistency_loss import ActionConsistencyLoss
from actor_extension.actor_dreamvla import ACTORTrainingWrapper, create_actor_dreamvla


class TestACTORTrainingWrapper:
    """Tests for training wrapper."""

    @pytest.fixture
    def wrapper(self):
        inverse_dynamics = InverseDynamicsHead(
            state_dim=384,
            action_dim=7,
        )
        return ACTORTrainingWrapper(
            inverse_dynamics=inverse_dynamics,
            l3_weight=0.1,
        )

    def test_compute_l3_from_features(self, wrapper):
        """Test L3 computation from DreamVLA features."""
        B_S = 40  # Batch * Sequence
        hidden_dim = 384
        num_tokens = 20

        current_features = torch.randn(B_S, hidden_dim)
        predicted_features = torch.randn(B_S, num_tokens, hidden_dim)
        gt_actions = torch.randn(B_S, 7)
        gt_actions[..., 6:] = torch.sigmoid(gt_actions[..., 6:])  # Gripper in [0, 1]

        result = wrapper.compute_l3_from_features(
            current_features,
            predicted_features,
            gt_actions
        )

        assert 'loss' in result
        assert result['loss'].item() > 0

    def test_gradient_flow_through_wrapper(self, wrapper):
        """Test gradients flow through wrapper."""
        B_S = 40
        hidden_dim = 384

        current_features = torch.randn(B_S, hidden_dim, requires_grad=True)
        predicted_features = torch.randn(B_S, 20, hidden_dim, requires_grad=True)
        gt_actions = torch.randn(B_S, 7)
        gt_actions[..., 6:] = torch.sigmoid(gt_actions[..., 6:])

        result = wrapper.compute_l3_from_features(
            current_features,
            predicted_features,
            gt_actions
        )

        result['loss'].backward()

        # Gradients should flow to predicted_features (the WM output)
        assert predicted_features.grad is not None
        assert predicted_features.grad.abs().sum() > 0


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_l3_loss_decreases_training(self):
        """Test that L3 loss decreases during training simulation."""
        # Setup
        hidden_dim = 384
        batch_size = 8
        seq_len = 5

        inverse_dynamics = InverseDynamicsHead(state_dim=hidden_dim, action_dim=7)
        l3_loss_fn = ActionConsistencyLoss()
        optimizer = torch.optim.Adam(inverse_dynamics.parameters(), lr=1e-3)

        # Fixed target data
        z_t = torch.randn(batch_size, seq_len, hidden_dim)
        z_next = torch.randn(batch_size, seq_len, hidden_dim)
        gt_arm = torch.randn(batch_size, seq_len, 6) * 0.5  # Small actions
        gt_gripper = torch.rand(batch_size, seq_len, 1)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()

            pred_arm, pred_gripper = inverse_dynamics(z_t, z_next)
            result = l3_loss_fn(pred_arm, pred_gripper, gt_arm, gt_gripper)
            loss = result['loss']

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_combined_with_mock_dreamvla(self):
        """Test ACTOR integration with mock DreamVLA-like model."""

        class MockDreamVLA(nn.Module):
            """Mock model that mimics DreamVLA's structure."""

            def __init__(self, hidden_dim=384):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.encoder = nn.Linear(3 * 224 * 224, hidden_dim)
                self.world_model = nn.Linear(hidden_dim, hidden_dim)
                self.action_head = nn.Linear(hidden_dim, 7)

            def forward(self, image):
                B = image.shape[0]
                x = image.flatten(1)
                z_t = self.encoder(x)
                z_next_pred = self.world_model(z_t)
                action = self.action_head(z_t)
                return {
                    'z_t': z_t,
                    'z_next_pred': z_next_pred,
                    'action': action,
                }

        # Setup
        hidden_dim = 384
        mock_model = MockDreamVLA(hidden_dim)
        inverse_dynamics = InverseDynamicsHead(state_dim=hidden_dim, action_dim=7)
        l3_loss_fn = ActionConsistencyLoss()

        optimizer = torch.optim.Adam(
            list(mock_model.parameters()) + list(inverse_dynamics.parameters()),
            lr=1e-3
        )

        # Training step
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        gt_arm = torch.randn(batch_size, 6) * 0.5
        gt_gripper = torch.rand(batch_size, 1)

        optimizer.zero_grad()

        # Forward
        outputs = mock_model(images)
        z_t = outputs['z_t']
        z_next_pred = outputs['z_next_pred']

        # L3: Action consistency
        pred_arm, pred_gripper = inverse_dynamics(z_t, z_next_pred)
        l3_result = l3_loss_fn(pred_arm, pred_gripper, gt_arm, gt_gripper)

        # Total loss (action prediction + L3)
        action_loss = nn.functional.mse_loss(outputs['action'][:, :6], gt_arm)
        total_loss = action_loss + 0.1 * l3_result['loss']

        total_loss.backward()
        optimizer.step()

        # Verify gradients propagate to world model
        assert mock_model.world_model.weight.grad is not None
        assert mock_model.world_model.weight.grad.abs().sum() > 0


class TestMemoryEfficiency:
    """Test memory usage."""

    def test_no_memory_leak(self):
        """Test no memory accumulation over iterations."""
        import gc

        hidden_dim = 384
        inverse_dynamics = InverseDynamicsHead(state_dim=hidden_dim, action_dim=7)
        l3_loss_fn = ActionConsistencyLoss()

        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        for i in range(100):
            z_t = torch.randn(8, 10, hidden_dim)
            z_next = torch.randn(8, 10, hidden_dim)
            gt_arm = torch.randn(8, 10, 6)
            gt_gripper = torch.rand(8, 10, 1)

            pred_arm, pred_gripper = inverse_dynamics(z_t, z_next)
            result = l3_loss_fn(pred_arm, pred_gripper, gt_arm, gt_gripper)

            # Don't accumulate gradients
            del z_t, z_next, pred_arm, pred_gripper, result

            if i % 10 == 0:
                gc.collect()

        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Memory should not grow significantly
        assert final_memory <= initial_memory * 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
