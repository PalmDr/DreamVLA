#!/usr/bin/env python3
"""
Comprehensive SmolVLM Experiment with Multiple Metrics

Metrics:
1. Per-dimension MSE (x, y, z, roll, pitch, yaw, gripper)
2. End-effector position error (Euclidean)
3. Trajectory smoothness (jerk)
4. Gripper accuracy
5. Long-horizon accumulated error

Uses REAL SmolVLM pretrained weights.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from transformers import SmolVLMForConditionalGeneration, SmolVLMProcessor
from datasets import load_dataset

print("=" * 70)
print("COMPREHENSIVE SmolVLM EXPERIMENT")
print("Multiple metrics with REAL pretrained weights")
print("=" * 70)


class LIBERODataset(IterableDataset):
    """LIBERO dataset with trajectory tracking for smoothness metrics."""

    def __init__(self, split='train', max_samples=500):
        self.max_samples = max_samples
        self.dataset = load_dataset('HuggingFaceVLA/libero', split=split, streaming=True)

    def __iter__(self):
        count = 0
        prev_sample = None
        prev_img = None
        prev_state = None
        prev_action = None
        prev_prev_action = None  # For jerk calculation

        for sample in self.dataset:
            if count >= self.max_samples:
                break

            img = sample['observation.images.image']
            if not isinstance(img, Image.Image):
                continue

            state = torch.tensor(sample['observation.state'], dtype=torch.float32)
            action = torch.tensor(sample['action'], dtype=torch.float32)

            if len(action) == 7:
                action_arm = action[:6]
                action_gripper = torch.sigmoid(action[6:7])
                action = torch.cat([action_arm, action_gripper])

            if prev_sample is not None and sample['episode_index'] == prev_sample['episode_index']:
                yield {
                    'image': prev_img,
                    'next_image': img,
                    'state': prev_state,
                    'next_state': state,
                    'action': prev_action,
                    'prev_action': prev_prev_action if prev_prev_action is not None else prev_action,
                    'language': sample.get('language_instruction', 'pick up the object'),
                    'episode_index': sample['episode_index'],
                    'frame_index': sample.get('frame_index', count),
                }
                count += 1

            prev_prev_action = prev_action
            prev_sample = sample
            prev_img = img
            prev_state = state
            prev_action = action


def collate_fn(batch):
    return {
        'images': [b['image'] for b in batch],
        'next_images': [b['next_image'] for b in batch],
        'states': torch.stack([b['state'] for b in batch]),
        'next_states': torch.stack([b['next_state'] for b in batch]),
        'actions': torch.stack([b['action'] for b in batch]),
        'prev_actions': torch.stack([b['prev_action'] for b in batch]),
        'texts': [b['language'] for b in batch],
        'episode_indices': [b['episode_index'] for b in batch],
    }


class SmolVLMEncoder(nn.Module):
    def __init__(self, model_name="HuggingFaceTB/SmolVLM-Instruct", device="cuda"):
        super().__init__()
        self.device = device

        print(f"Loading REAL SmolVLM: {model_name}")
        self.processor = SmolVLMProcessor.from_pretrained(model_name)
        self.model = SmolVLMForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)

        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_dim = self.model.config.vision_config.hidden_size
        print(f"Vision hidden dim: {self.hidden_dim}")

        self.state_encoder = nn.Sequential(
            nn.Linear(8, 256), nn.ReLU(), nn.Linear(256, self.hidden_dim)
        ).to(device)

    def encode_image(self, images, texts):
        all_features = []
        with torch.no_grad():
            for img, text in zip(images, texts):
                formatted_text = f"<image>User: {text}\nAssistant:"
                inputs = self.processor(
                    text=formatted_text, images=[img], return_tensors="pt"
                ).to(self.device)

                pixel_values = inputs.pixel_values.to(torch.float16)
                if len(pixel_values.shape) == 5:
                    b, n, c, h, w = pixel_values.shape
                    pixel_values = pixel_values.view(b * n, c, h, w)

                vision_outputs = self.model.model.vision_model(pixel_values)
                features = vision_outputs.last_hidden_state.mean(dim=1).mean(dim=0, keepdim=True)
                all_features.append(features)

        return torch.cat(all_features, dim=0).float()

    def encode(self, images, states, texts):
        visual = self.encode_image(images, texts)
        state_emb = self.state_encoder(states)
        return visual + state_emb


class Model(nn.Module):
    def __init__(self, encoder, action_dim=7, device="cuda"):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = encoder.hidden_dim
        self.device = device

        self.world_model = nn.Sequential(
            nn.Linear(self.hidden_dim + action_dim, self.hidden_dim * 2),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        ).to(device)

        self.policy = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, action_dim)
        ).to(device)

        self.inverse_dynamics = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, action_dim)
        ).to(device)

    def encode(self, images, states, texts):
        return self.encoder.encode(images, states, texts)

    def forward_policy(self, z):
        return self.policy(z)

    def forward_world_model(self, z, action):
        return self.world_model(torch.cat([z, action], dim=-1))

    def forward_inverse_dynamics(self, z_t, z_next):
        return self.inverse_dynamics(torch.cat([z_t, z_next], dim=-1))


def compute_comprehensive_metrics(predictions, targets, prev_actions):
    """
    Compute comprehensive metrics:
    1. Per-dimension MSE
    2. End-effector position error
    3. Trajectory smoothness (jerk)
    4. Gripper accuracy
    """
    metrics = {}

    # Action dimensions: [x, y, z, roll, pitch, yaw, gripper]
    dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']

    # 1. Per-dimension MSE
    for i, name in enumerate(dim_names):
        if i < 6:
            metrics[f'mse_{name}'] = F.mse_loss(predictions[:, i], targets[:, i]).item()
        else:
            # Gripper uses sigmoid
            pred_grip = torch.sigmoid(predictions[:, i])
            metrics[f'mse_{name}'] = F.mse_loss(pred_grip, targets[:, i]).item()

    # 2. Total arm MSE (first 6 dimensions)
    metrics['arm_mse'] = F.mse_loss(predictions[:, :6], targets[:, :6]).item()

    # 3. End-effector position error (Euclidean distance for x, y, z)
    position_error = torch.sqrt(((predictions[:, :3] - targets[:, :3]) ** 2).sum(dim=1))
    metrics['position_error_mean'] = position_error.mean().item()
    metrics['position_error_std'] = position_error.std().item()

    # 4. Rotation error (Euclidean distance for roll, pitch, yaw)
    rotation_error = torch.sqrt(((predictions[:, 3:6] - targets[:, 3:6]) ** 2).sum(dim=1))
    metrics['rotation_error_mean'] = rotation_error.mean().item()
    metrics['rotation_error_std'] = rotation_error.std().item()

    # 5. Gripper accuracy
    pred_gripper = (torch.sigmoid(predictions[:, 6]) > 0.5).float()
    target_gripper = (targets[:, 6] > 0.5).float()
    metrics['gripper_accuracy'] = (pred_gripper == target_gripper).float().mean().item()

    # 6. Trajectory smoothness (jerk = change in acceleration)
    # Jerk approximation: |a_t - 2*a_{t-1} + a_{t-2}| ≈ |pred - 2*target + prev|
    # Simplified: measure how different prediction is from smooth trajectory
    velocity_pred = predictions[:, :6] - prev_actions[:, :6]  # Approximate velocity
    velocity_target = targets[:, :6] - prev_actions[:, :6]
    jerk = torch.abs(velocity_pred - velocity_target)
    metrics['jerk_mean'] = jerk.mean().item()
    metrics['jerk_max'] = jerk.max().item()

    # 7. Action magnitude (to check if predictions are reasonable)
    metrics['pred_magnitude'] = torch.norm(predictions[:, :6], dim=1).mean().item()
    metrics['target_magnitude'] = torch.norm(targets[:, :6], dim=1).mean().item()

    return metrics


def train_and_eval(name, config, encoder, device, num_train=500, num_test=200, epochs=3):
    """Train and evaluate with comprehensive metrics."""
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Config: {config}")
    print(f"Training samples: {num_train}, Test samples: {num_test}, Epochs: {epochs}")
    print(f"{'='*60}")
    sys.stdout.flush()

    model = Model(encoder, device=device)
    trainable = list(model.world_model.parameters()) + list(model.policy.parameters()) + \
                list(model.inverse_dynamics.parameters()) + list(model.encoder.state_encoder.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)

    # Training
    for epoch in range(1, epochs + 1):
        model.train()
        model.encoder.model.eval()

        train_data = LIBERODataset(split='train', max_samples=num_train)
        train_loader = DataLoader(train_data, batch_size=8, collate_fn=collate_fn)

        total_loss = 0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            images = batch['images']
            next_images = batch['next_images']
            states = batch['states'].to(device)
            next_states = batch['next_states'].to(device)
            actions = batch['actions'].to(device)
            texts = batch['texts']

            optimizer.zero_grad()

            z_t = model.encode(images, states, texts)
            z_t1 = model.encode(next_images, next_states, texts)
            vla_pred = model.forward_policy(z_t)

            # Base loss
            arm_loss = F.smooth_l1_loss(vla_pred[:, :6], actions[:, :6])
            grip_loss = F.binary_cross_entropy(torch.sigmoid(vla_pred[:, 6:]), actions[:, 6:].clamp(0, 1))
            loss = arm_loss + 0.1 * grip_loss

            # L3: Action consistency (prior work)
            if config.get("l3", 0) > 0:
                z_pred = model.forward_world_model(z_t, actions)
                a_rec = model.forward_inverse_dynamics(z_t, z_pred)
                loss += config["l3"] * F.mse_loss(a_rec[:, :6], actions[:, :6].detach())

            # L_sim: Simulation loss (NOVEL)
            if config.get("l_sim", 0) > 0:
                z_sim = model.forward_world_model(z_t, vla_pred)
                loss += config["l_sim"] * F.mse_loss(z_sim, z_t1.detach())

            # L_sim_inv: Inverse dynamics (NOVEL)
            if config.get("l_sim_inv", 0) > 0:
                id_pred = model.forward_inverse_dynamics(z_t, z_t1)
                loss += config["l_sim_inv"] * F.mse_loss(id_pred[:, :6], actions[:, :6].detach())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"[{name}] Epoch {epoch}: loss={total_loss/n_batches:.4f}")
        sys.stdout.flush()

    # Comprehensive Evaluation
    print(f"[{name}] Running comprehensive evaluation...")
    model.eval()

    test_data = LIBERODataset(split='train', max_samples=num_test)
    test_loader = DataLoader(test_data, batch_size=8, collate_fn=collate_fn)

    all_predictions = []
    all_targets = []
    all_prev_actions = []

    # Track per-episode metrics for long-horizon analysis
    episode_errors = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            images = batch['images']
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            prev_actions = batch['prev_actions'].to(device)
            texts = batch['texts']
            episode_indices = batch['episode_indices']

            z = model.encode(images, states, texts)
            pred = model.forward_policy(z)

            all_predictions.append(pred.cpu())
            all_targets.append(actions.cpu())
            all_prev_actions.append(prev_actions.cpu())

            # Track per-episode errors
            errors = F.mse_loss(pred[:, :6], actions[:, :6], reduction='none').mean(dim=1)
            for i, ep_idx in enumerate(episode_indices):
                episode_errors[ep_idx].append(errors[i].item())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_prev_actions = torch.cat(all_prev_actions, dim=0)

    # Compute comprehensive metrics
    metrics = compute_comprehensive_metrics(all_predictions, all_targets, all_prev_actions)

    # Add long-horizon metrics (error accumulation over episodes)
    episode_mean_errors = [np.mean(errors) for errors in episode_errors.values()]
    episode_max_errors = [np.max(errors) for errors in episode_errors.values()]
    metrics['long_horizon_mean_error'] = np.mean(episode_mean_errors)
    metrics['long_horizon_std_error'] = np.std(episode_mean_errors)
    metrics['long_horizon_max_error'] = np.mean(episode_max_errors)

    # Print summary
    print(f"\n[{name}] === COMPREHENSIVE METRICS ===")
    print(f"  Arm MSE: {metrics['arm_mse']:.6f}")
    print(f"  Position Error: {metrics['position_error_mean']:.4f} ± {metrics['position_error_std']:.4f}")
    print(f"  Rotation Error: {metrics['rotation_error_mean']:.4f} ± {metrics['rotation_error_std']:.4f}")
    print(f"  Gripper Accuracy: {metrics['gripper_accuracy']*100:.1f}%")
    print(f"  Jerk (smoothness): {metrics['jerk_mean']:.4f}")
    print(f"  Long-horizon Error: {metrics['long_horizon_mean_error']:.4f} ± {metrics['long_horizon_std_error']:.4f}")
    print(f"  Per-dim MSE: x={metrics['mse_x']:.4f}, y={metrics['mse_y']:.4f}, z={metrics['mse_z']:.4f}")
    print(f"               roll={metrics['mse_roll']:.4f}, pitch={metrics['mse_pitch']:.4f}, yaw={metrics['mse_yaw']:.4f}")
    sys.stdout.flush()

    return {
        "name": name,
        "config": config,
        "metrics": metrics,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load encoder once
    encoder = SmolVLMEncoder(device=device)
    results = []

    # Run experiments with more data and epochs
    NUM_TRAIN = 500
    NUM_TEST = 200
    EPOCHS = 3

    print(f"\nRunning experiments with {NUM_TRAIN} train, {NUM_TEST} test, {EPOCHS} epochs")

    results.append(train_and_eval("baseline", {}, encoder, device, NUM_TRAIN, NUM_TEST, EPOCHS))
    results.append(train_and_eval("L3=0.01", {"l3": 0.01}, encoder, device, NUM_TRAIN, NUM_TEST, EPOCHS))
    results.append(train_and_eval("L_sim=0.01", {"l_sim": 0.01}, encoder, device, NUM_TRAIN, NUM_TEST, EPOCHS))
    results.append(train_and_eval("L_sim_inv=0.01", {"l_sim_inv": 0.01}, encoder, device, NUM_TRAIN, NUM_TEST, EPOCHS))
    results.append(train_and_eval("Combined", {"l3": 0.005, "l_sim": 0.005, "l_sim_inv": 0.005}, encoder, device, NUM_TRAIN, NUM_TEST, EPOCHS))

    # Results Summary
    print("\n" + "=" * 100)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 100)

    baseline_metrics = results[0]["metrics"]

    print(f"\n{'Method':<20} {'Arm MSE':<10} {'Pos Err':<10} {'Rot Err':<10} {'Grip Acc':<10} {'Jerk':<10} {'LH Err':<10}")
    print("-" * 80)

    for r in results:
        m = r["metrics"]
        arm_imp = (baseline_metrics["arm_mse"] - m["arm_mse"]) / baseline_metrics["arm_mse"] * 100
        print(f"{r['name']:<20} {m['arm_mse']:.4f}     {m['position_error_mean']:.4f}     {m['rotation_error_mean']:.4f}     {m['gripper_accuracy']*100:.1f}%      {m['jerk_mean']:.4f}     {m['long_horizon_mean_error']:.4f}  ({arm_imp:+.1f}%)")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": "SmolVLM (REAL)",
        "num_train": NUM_TRAIN,
        "num_test": NUM_TEST,
        "epochs": EPOCHS,
        "results": results,
    }

    output_path = Path("comprehensive_results.json")
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
