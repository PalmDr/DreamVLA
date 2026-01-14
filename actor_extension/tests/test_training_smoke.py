"""
Smoke tests for ACTOR + DreamVLA training.

These tests verify that:
1. L3 loss can be computed with DreamVLA-like features
2. Training loop runs without errors
3. Loss decreases over training steps
4. Gradients flow correctly to all components
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from actor_extension.inverse_dynamics import InverseDynamicsHead
from actor_extension.action_consistency_loss import ActionConsistencyLoss


class MockDreamVLAFeatureExtractor(nn.Module):
    """
    Mock that simulates DreamVLA's feature extraction for testing.

    Produces:
    - state_features: Current state latent (from encoder)
    - pred_features: Predicted next state features (from world model)
    - action_pred: Predicted actions
    """

    def __init__(self, hidden_dim=384, num_obs_tokens=20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_obs_tokens = num_obs_tokens

        # Simplified encoder
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, hidden_dim),
        )

        # State encoder
        self.state_encoder = nn.Linear(8, hidden_dim)

        # World model (predicts next observation features)
        self.world_model = nn.Sequential(
            nn.Linear(hidden_dim + 7, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * num_obs_tokens),
        )

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 7),
        )

    def forward(self, images, states, actions=None):
        """
        Forward pass mimicking DreamVLA.

        Args:
            images: (B, 3, 224, 224)
            states: (B, 8)
            actions: (B, 7) optional for world model

        Returns:
            dict with state_features, pred_features, action_pred
        """
        B = images.shape[0]

        # Encode current state
        img_features = self.image_encoder(images)
        state_features = self.state_encoder(states)
        current_latent = img_features + state_features  # (B, hidden_dim)

        # Predict action
        action_pred = self.action_head(current_latent)

        # World model: predict next observation features
        if actions is None:
            actions = action_pred.detach()

        wm_input = torch.cat([current_latent, actions], dim=-1)
        pred_features = self.world_model(wm_input)
        pred_features = pred_features.view(B, self.num_obs_tokens, self.hidden_dim)

        return {
            'state_features': current_latent,
            'pred_features': pred_features,
            'action_pred': action_pred,
        }


class TestTrainingSmoke:
    """Smoke tests for training pipeline."""

    @pytest.fixture
    def setup(self):
        """Setup models and optimizer."""
        hidden_dim = 384

        dreamvla_mock = MockDreamVLAFeatureExtractor(hidden_dim=hidden_dim)
        inverse_dynamics = InverseDynamicsHead(state_dim=hidden_dim, action_dim=7)
        l3_loss_fn = ActionConsistencyLoss()

        all_params = list(dreamvla_mock.parameters()) + list(inverse_dynamics.parameters())
        optimizer = torch.optim.Adam(all_params, lr=1e-4)

        return {
            'dreamvla': dreamvla_mock,
            'inverse_dynamics': inverse_dynamics,
            'l3_loss_fn': l3_loss_fn,
            'optimizer': optimizer,
            'hidden_dim': hidden_dim,
        }

    def test_single_training_step(self, setup):
        """Test a single training step completes without errors."""
        dreamvla = setup['dreamvla']
        inverse_dynamics = setup['inverse_dynamics']
        l3_loss_fn = setup['l3_loss_fn']
        optimizer = setup['optimizer']

        # Create batch
        B = 4
        images = torch.randn(B, 3, 224, 224)
        states = torch.randn(B, 8)
        gt_actions = torch.randn(B, 7)
        gt_actions[..., 6:] = torch.sigmoid(gt_actions[..., 6:])

        # Forward
        outputs = dreamvla(images, states)
        state_features = outputs['state_features']
        pred_features = outputs['pred_features']
        action_pred = outputs['action_pred']

        # Compute L3 loss
        pred_next_latent = pred_features.mean(dim=1)  # Pool to (B, hidden)
        pred_arm, pred_gripper = inverse_dynamics(state_features, pred_next_latent)

        l3_result = l3_loss_fn(
            pred_arm, pred_gripper,
            gt_actions[..., :6], gt_actions[..., 6:]
        )

        # Action prediction loss
        action_loss = F.mse_loss(action_pred, gt_actions)

        # Total loss
        total_loss = action_loss + 0.1 * l3_result['loss']

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Verify loss is finite
        assert torch.isfinite(total_loss)
        assert torch.isfinite(l3_result['loss'])

    def test_loss_decreases_over_steps(self, setup):
        """Test that loss decreases over multiple training steps."""
        dreamvla = setup['dreamvla']
        inverse_dynamics = setup['inverse_dynamics']
        l3_loss_fn = setup['l3_loss_fn']
        optimizer = setup['optimizer']

        # Fixed data for consistent training
        torch.manual_seed(42)
        B = 8
        images = torch.randn(B, 3, 224, 224)
        states = torch.randn(B, 8)
        gt_actions = torch.randn(B, 7)
        gt_actions[..., 6:] = torch.sigmoid(gt_actions[..., 6:])

        losses = []
        l3_losses = []

        for step in range(30):
            # Forward
            outputs = dreamvla(images, states)
            state_features = outputs['state_features']
            pred_features = outputs['pred_features']
            action_pred = outputs['action_pred']

            # L3
            pred_next_latent = pred_features.mean(dim=1)
            pred_arm, pred_gripper = inverse_dynamics(state_features, pred_next_latent)
            l3_result = l3_loss_fn(
                pred_arm, pred_gripper,
                gt_actions[..., :6], gt_actions[..., 6:]
            )

            # Action loss
            action_loss = F.mse_loss(action_pred, gt_actions)
            total_loss = action_loss + 0.1 * l3_result['loss']

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            losses.append(total_loss.item())
            l3_losses.append(l3_result['loss'].item())

        # Loss should decrease
        avg_first_10 = sum(losses[:10]) / 10
        avg_last_10 = sum(losses[-10:]) / 10
        assert avg_last_10 < avg_first_10, f"Loss did not decrease: {avg_first_10:.4f} -> {avg_last_10:.4f}"

        # L3 specifically should also improve
        l3_first_10 = sum(l3_losses[:10]) / 10
        l3_last_10 = sum(l3_losses[-10:]) / 10
        print(f"L3 loss: {l3_first_10:.4f} -> {l3_last_10:.4f}")

    def test_gradients_flow_to_world_model(self, setup):
        """Test that L3 gradients flow back to world model."""
        dreamvla = setup['dreamvla']
        inverse_dynamics = setup['inverse_dynamics']
        l3_loss_fn = setup['l3_loss_fn']

        B = 4
        images = torch.randn(B, 3, 224, 224)
        states = torch.randn(B, 8)
        gt_actions = torch.randn(B, 7)
        gt_actions[..., 6:] = torch.sigmoid(gt_actions[..., 6:])

        # Forward
        outputs = dreamvla(images, states)
        state_features = outputs['state_features']
        pred_features = outputs['pred_features']

        # L3 loss (only)
        pred_next_latent = pred_features.mean(dim=1)
        pred_arm, pred_gripper = inverse_dynamics(state_features.detach(), pred_next_latent)
        l3_result = l3_loss_fn(
            pred_arm, pred_gripper,
            gt_actions[..., :6], gt_actions[..., 6:]
        )

        # Backward
        l3_result['loss'].backward()

        # Check gradients flow to world model
        wm_grad_sum = sum(
            p.grad.abs().sum().item()
            for p in dreamvla.world_model.parameters()
            if p.grad is not None
        )
        assert wm_grad_sum > 0, "No gradients flowed to world model"

    def test_no_nan_gradients(self, setup):
        """Test no NaN gradients during training."""
        dreamvla = setup['dreamvla']
        inverse_dynamics = setup['inverse_dynamics']
        l3_loss_fn = setup['l3_loss_fn']
        optimizer = setup['optimizer']

        B = 4
        images = torch.randn(B, 3, 224, 224)
        states = torch.randn(B, 8)
        gt_actions = torch.randn(B, 7)
        gt_actions[..., 6:] = torch.sigmoid(gt_actions[..., 6:])

        for _ in range(10):
            outputs = dreamvla(images, states)
            pred_next_latent = outputs['pred_features'].mean(dim=1)
            pred_arm, pred_gripper = inverse_dynamics(
                outputs['state_features'], pred_next_latent
            )
            l3_result = l3_loss_fn(
                pred_arm, pred_gripper,
                gt_actions[..., :6], gt_actions[..., 6:]
            )

            action_loss = F.mse_loss(outputs['action_pred'], gt_actions)
            total_loss = action_loss + 0.1 * l3_result['loss']

            optimizer.zero_grad()
            total_loss.backward()

            # Check for NaNs
            for name, param in dreamvla.named_parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

            for name, param in inverse_dynamics.named_parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(), f"NaN gradient in ID {name}"

            optimizer.step()

    def test_checkpoint_save_load(self, setup, tmp_path):
        """Test checkpoint saving and loading."""
        dreamvla = setup['dreamvla']
        inverse_dynamics = setup['inverse_dynamics']
        optimizer = setup['optimizer']

        # Train a few steps
        B = 4
        images = torch.randn(B, 3, 224, 224)
        states = torch.randn(B, 8)

        for _ in range(5):
            outputs = dreamvla(images, states)
            loss = outputs['action_pred'].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save checkpoint
        ckpt_path = tmp_path / "checkpoint.pth"
        torch.save({
            'dreamvla_state_dict': dreamvla.state_dict(),
            'inverse_dynamics_state_dict': inverse_dynamics.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, ckpt_path)

        # Load checkpoint into new models
        dreamvla_new = MockDreamVLAFeatureExtractor()
        inverse_dynamics_new = InverseDynamicsHead(state_dim=384, action_dim=7)

        ckpt = torch.load(ckpt_path)
        dreamvla_new.load_state_dict(ckpt['dreamvla_state_dict'])
        inverse_dynamics_new.load_state_dict(ckpt['inverse_dynamics_state_dict'])

        # Verify outputs match
        with torch.no_grad():
            out1 = dreamvla(images, states)
            out2 = dreamvla_new(images, states)
            assert torch.allclose(out1['action_pred'], out2['action_pred'])


class TestL3LossProperties:
    """Test specific properties of L3 loss."""

    def test_l3_provides_signal_when_wm_wrong(self):
        """
        Test that L3 loss is high when world model prediction is wrong.

        Key insight: If WM predicts incorrect next state, ID cannot
        recover the correct action, so L3 loss is high.
        """
        hidden_dim = 384
        inverse_dynamics = InverseDynamicsHead(state_dim=hidden_dim, action_dim=7)
        l3_loss_fn = ActionConsistencyLoss()

        B = 8
        current_state = torch.randn(B, hidden_dim)
        correct_next_state = torch.randn(B, hidden_dim)
        wrong_next_state = torch.randn(B, hidden_dim)  # Random, uncorrelated
        gt_action = torch.randn(B, 7)
        gt_action[..., 6:] = torch.sigmoid(gt_action[..., 6:])

        # Train ID to predict action from correct transitions
        optimizer = torch.optim.Adam(inverse_dynamics.parameters(), lr=1e-3)
        for _ in range(100):
            pred_arm, pred_gripper = inverse_dynamics(current_state, correct_next_state)
            result = l3_loss_fn(pred_arm, pred_gripper, gt_action[..., :6], gt_action[..., 6:])
            optimizer.zero_grad()
            result['loss'].backward()
            optimizer.step()

        # Now test: L3 should be lower for correct prediction
        with torch.no_grad():
            pred_arm_correct, pred_gripper_correct = inverse_dynamics(
                current_state, correct_next_state
            )
            loss_correct = l3_loss_fn(
                pred_arm_correct, pred_gripper_correct,
                gt_action[..., :6], gt_action[..., 6:]
            )

            pred_arm_wrong, pred_gripper_wrong = inverse_dynamics(
                current_state, wrong_next_state
            )
            loss_wrong = l3_loss_fn(
                pred_arm_wrong, pred_gripper_wrong,
                gt_action[..., :6], gt_action[..., 6:]
            )

        print(f"L3 loss with correct prediction: {loss_correct['loss'].item():.4f}")
        print(f"L3 loss with wrong prediction: {loss_wrong['loss'].item():.4f}")

        # Correct should have lower loss
        assert loss_correct['loss'] < loss_wrong['loss']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
