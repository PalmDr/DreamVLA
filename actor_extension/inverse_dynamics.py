"""
Inverse Dynamics Head for ACTOR Extension.

Predicts action from state transition (z_t, z_{t+1}).
Used for L3 Action Consistency Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class InverseDynamicsHead(nn.Module):
    """
    Inverse Dynamics model that predicts action from consecutive states.

    Given latent states z_t and z_{t+1}, predicts the action that caused
    the transition. This is used for:
    - L3: Action Consistency - verify WM predictions are action-consistent
    - L5: Forward Verification - verify VLA actions match ID predictions

    Architecture: MLP with skip connections

    Args:
        state_dim: Dimension of latent state (DreamVLA hidden_dim)
        action_dim: Dimension of action (7 for robot arm + gripper)
        hidden_dim: MLP hidden dimension
        num_layers: Number of MLP layers
    """

    def __init__(
        self,
        state_dim: int = 384,
        action_dim: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Input: concatenation of z_t and z_{t+1}
        input_dim = state_dim * 2

        # Build MLP layers
        layers = []
        for i in range(num_layers):
            in_features = input_dim if i == 0 else hidden_dim
            out_features = hidden_dim

            layers.extend([
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features),
                nn.GELU(),
                nn.Dropout(dropout),
            ])

        self.mlp = nn.Sequential(*layers)

        # Separate heads for arm and gripper
        self.arm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 6),  # 6 DoF arm
            nn.Tanh(),  # Normalized actions
        )

        self.gripper_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),  # Binary gripper
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        z_next: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict action from state transition.

        Args:
            z_t: Current state latent (B, state_dim) or (B, S, state_dim)
            z_next: Next state latent (B, state_dim) or (B, S, state_dim)

        Returns:
            arm_action: (B, 6) or (B, S, 6) predicted arm action
            gripper_action: (B, 1) or (B, S, 1) predicted gripper action
        """
        # Handle sequence dimension
        has_seq_dim = z_t.dim() == 3
        if has_seq_dim:
            B, S, D = z_t.shape
            z_t = z_t.reshape(B * S, D)
            z_next = z_next.reshape(B * S, D)

        # Concatenate states
        x = torch.cat([z_t, z_next], dim=-1)  # (B, state_dim * 2)

        # MLP forward
        features = self.mlp(x)  # (B, hidden_dim)

        # Predict actions
        arm_action = self.arm_head(features)  # (B, 6)
        gripper_action = self.gripper_head(features)  # (B, 1)

        # Reshape back if needed
        if has_seq_dim:
            arm_action = arm_action.reshape(B, S, 6)
            gripper_action = gripper_action.reshape(B, S, 1)

        return arm_action, gripper_action

    def predict_action(
        self,
        z_t: torch.Tensor,
        z_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict full action (arm + gripper concatenated).

        Returns:
            action: (B, 7) or (B, S, 7) full action
        """
        arm_action, gripper_action = self.forward(z_t, z_next)
        return torch.cat([arm_action, gripper_action], dim=-1)


class TransformerInverseDynamics(nn.Module):
    """
    Transformer-based Inverse Dynamics for sequence modeling.

    Can process entire sequences at once, useful for action chunk prediction.
    """

    def __init__(
        self,
        state_dim: int = 384,
        action_dim: int = 7,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Project states to hidden dim
        self.state_proj = nn.Linear(state_dim, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Action prediction heads
        self.arm_head = nn.Linear(hidden_dim, 6)
        self.gripper_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        z_t: torch.Tensor,
        z_next: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process state pairs through transformer.

        Args:
            z_t: (B, S, state_dim) current states
            z_next: (B, S, state_dim) next states

        Returns:
            arm_action: (B, S, 6)
            gripper_action: (B, S, 1)
        """
        B, S, D = z_t.shape

        # Project and stack states: [z_t, z_next] for each timestep
        z_t_proj = self.state_proj(z_t)  # (B, S, hidden)
        z_next_proj = self.state_proj(z_next)  # (B, S, hidden)

        # Interleave: [z_0, z_1, z_1, z_2, z_2, z_3, ...]
        # Or simply concatenate features
        features = z_t_proj + z_next_proj  # Residual combination

        # Transformer
        features = self.transformer(features)  # (B, S, hidden)

        # Predict actions
        arm_action = torch.tanh(self.arm_head(features))
        gripper_action = torch.sigmoid(self.gripper_head(features))

        return arm_action, gripper_action
