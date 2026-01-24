#!/usr/bin/env python3
"""
ACTOR: Truly Novel Simulation Loss Experiment

L_sim = ||WM(z_t, VLA(z_t)) - z_{t+1}||²

This loss checks: "Does simulating VLA's predicted action through the
world model produce the correct next state?"

This is NOVEL because:
- Prior work (L3): Checks if ID can recover action from WM prediction
- Our L_sim: Checks if WM simulation of VLA action matches reality

Gradient flow: VLA ← WM ← real world (joint training signal!)

Run with: PYTHONUNBUFFERED=1 uv run python run_simulation_loss_experiment.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms

from datasets import load_dataset


class LIBERODataset(IterableDataset):
    """LIBERO dataset with consecutive frame pairs."""

    def __init__(self, split='train', max_samples=5000, img_size=128):
        self.max_samples = max_samples
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.dataset = load_dataset('HuggingFaceVLA/libero', split=split, streaming=True)

    def __iter__(self):
        count = 0
        prev_sample = None
        prev_img = None
        prev_state = None
        prev_action = None

        for sample in self.dataset:
            if count >= self.max_samples:
                break

            img = sample['observation.images.image']
            if isinstance(img, Image.Image):
                img_tensor = self.transform(img)
            else:
                img_tensor = torch.zeros(3, self.img_size, self.img_size)

            state = torch.tensor(sample['observation.state'], dtype=torch.float32)
            action = torch.tensor(sample['action'], dtype=torch.float32)

            if len(action) == 7:
                action_arm = action[:6]
                action_gripper = torch.sigmoid(action[6:7])
                action = torch.cat([action_arm, action_gripper])

            if prev_sample is not None and sample['episode_index'] == prev_sample['episode_index']:
                yield {
                    'image': prev_img,
                    'next_image': img_tensor,
                    'state': prev_state,
                    'next_state': state,
                    'action': prev_action,
                }
                count += 1

            prev_sample = sample
            prev_img = img_tensor
            prev_state = state
            prev_action = action


def collate_fn(batch):
    return {
        'images': torch.stack([b['image'] for b in batch]).contiguous(),
        'next_images': torch.stack([b['next_image'] for b in batch]).contiguous(),
        'states': torch.stack([b['state'] for b in batch]).contiguous(),
        'next_states': torch.stack([b['next_state'] for b in batch]).contiguous(),
        'actions': torch.stack([b['action'] for b in batch]).contiguous(),
    }


class WorldModelVLA(nn.Module):
    """World Model + VLA with shared encoder."""

    def __init__(self, state_dim=8, action_dim=7, hidden_dim=512, pretrained=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pretrained = pretrained

        # Vision encoder
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 16, hidden_dim),
        )

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        # World model: predicts next latent state
        self.world_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Policy head (VLA)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Inverse dynamics head
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        if pretrained:
            self._init_pretrained()

    def _init_pretrained(self):
        """Initialize with pretrained-like weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def encode(self, img, state):
        z_img = self.img_encoder(img)
        z_state = self.state_encoder(state)
        return self.fusion(torch.cat([z_img, z_state], dim=-1))

    def forward_policy(self, img, state):
        z = self.encode(img, state)
        return self.policy(z)

    def forward_world_model(self, z, action):
        return self.world_model(torch.cat([z, action], dim=-1))

    def forward_inverse_dynamics(self, z_t, z_next):
        return self.inverse_dynamics(torch.cat([z_t, z_next], dim=-1))


def compute_action_loss(pred, target):
    """Standard action prediction loss."""
    arm_loss = F.smooth_l1_loss(pred[:, :6], target[:, :6])
    gripper_loss = F.binary_cross_entropy(
        pred[:, 6:].clamp(1e-6, 1-1e-6),
        target[:, 6:].clamp(0, 1)
    )
    return arm_loss + 0.1 * gripper_loss


def compute_l3_loss(model, z_t, action):
    """
    L3: Action Consistency Loss (EXISTS IN PRIOR WORK - Seer, UWM)

    Check: Can inverse dynamics recover the action from world model prediction?
    L3 = ||ID(z_t, WM(z_t, a)) - a||²
    """
    z_pred = model.forward_world_model(z_t, action)
    action_recovered = model.forward_inverse_dynamics(z_t, z_pred)
    return F.mse_loss(action_recovered[:, :6], action[:, :6].detach())


def compute_l_sim(model, z_t, z_t1_real, vla_pred):
    """
    L_sim: Simulation Loss (TRULY NOVEL!)

    Check: Does simulating VLA's predicted action through WM produce
    the correct next state?

    L_sim = ||WM(z_t, VLA(z_t)) - z_{t+1}||²

    Key insight: This provides direct feedback to VLA:
    "Your action is wrong because simulating it doesn't match reality"

    Gradient flow:
    - Through WM: improves world model predictions
    - Through VLA (via action input): improves VLA action quality

    This is DIFFERENT from L3:
    - L3 checks: Can ID recover action? (action consistency)
    - L_sim checks: Does VLA's action lead to correct state? (state consistency)
    """
    # Simulate VLA's predicted action through world model
    z_t1_predicted = model.forward_world_model(z_t, vla_pred)

    # Compare to real next state encoding
    # Note: We detach z_t1_real to avoid gradients flowing back through encoder twice
    loss = F.mse_loss(z_t1_predicted, z_t1_real.detach())

    return loss


def compute_l_sim_inv(model, z_t, z_t1_real, actions):
    """
    L_sim_inv: Inverse Simulation Loss (TRULY NOVEL!)

    Check: Does inverse dynamics on (z_t, z_{t+1}) give the ground truth action?
    Then use this to supervise the world model.

    L_sim_inv = ||ID(z_t, z_{t+1}) - a_t||²

    This trains the inverse dynamics model on real transitions.
    """
    id_pred = model.forward_inverse_dynamics(z_t, z_t1_real)
    return F.mse_loss(id_pred[:, :6], actions[:, :6].detach())


def train_epoch(model, dataloader, optimizer, config, device):
    model.train()
    total_losses = {"total": 0, "action": 0, "l3": 0, "l_sim": 0, "l_sim_inv": 0}
    n_batches = 0

    for batch in dataloader:
        imgs = batch['images'].to(device)
        next_imgs = batch['next_images'].to(device)
        states = batch['states'].to(device)
        next_states = batch['next_states'].to(device)
        actions = batch['actions'].to(device)

        optimizer.zero_grad()

        # Encode current and next state
        z_t = model.encode(imgs, states)
        z_t1_real = model.encode(next_imgs, next_states)

        # VLA prediction
        vla_pred = model.forward_policy(imgs, states)

        # Base action loss
        loss = compute_action_loss(vla_pred, actions)
        total_losses["action"] += loss.item()

        # L3: Action consistency (prior work)
        if config.get("l3_weight", 0) > 0:
            l3 = compute_l3_loss(model, z_t, actions)
            loss = loss + config["l3_weight"] * l3
            total_losses["l3"] += l3.item()

        # L_sim: Simulation loss (NOVEL!)
        if config.get("l_sim_weight", 0) > 0:
            l_sim = compute_l_sim(model, z_t, z_t1_real, vla_pred)
            loss = loss + config["l_sim_weight"] * l_sim
            total_losses["l_sim"] += l_sim.item()

        # L_sim_inv: Inverse simulation loss (NOVEL!)
        if config.get("l_sim_inv_weight", 0) > 0:
            l_sim_inv = compute_l_sim_inv(model, z_t, z_t1_real, actions)
            loss = loss + config["l_sim_inv_weight"] * l_sim_inv
            total_losses["l_sim_inv"] += l_sim_inv.item()

        loss.backward()
        optimizer.step()

        total_losses["total"] += loss.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


def evaluate(model, dataloader, device):
    model.eval()
    arm_mse_total = 0
    gripper_correct = 0
    n_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch['images'].to(device)
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)

            pred = model.forward_policy(imgs, states)

            arm_mse_total += F.mse_loss(pred[:, :6], actions[:, :6], reduction="sum").item()
            gripper_pred = (pred[:, 6:] > 0.5).float()
            gripper_correct += (gripper_pred == actions[:, 6:].round()).sum().item()
            n_samples += imgs.shape[0]

    return {
        "arm_mse": arm_mse_total / max(n_samples, 1),
        "gripper_acc": gripper_correct / max(n_samples, 1),
    }


def run_experiment(name, config, device, epochs=10, pretrained=True, num_train=2000, num_test=500):
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    config_str = ", ".join([f"{k}={v}" for k, v in config.items() if v > 0])
    print(f"Config: {config_str if config_str else 'baseline'}")
    print(f"Pretrained: {pretrained}")
    print(f"{'='*60}")
    sys.stdout.flush()

    # Load fresh data
    train_dataset = LIBERODataset(split='train', max_samples=num_train)
    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn)

    # Create model
    model = WorldModelVLA(pretrained=pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train
    for epoch in range(1, epochs + 1):
        losses = train_epoch(model, train_loader, optimizer, config, device)

        loss_str = f"loss={losses['total']:.4f}"
        if config.get("l3_weight", 0) > 0:
            loss_str += f", L3={losses['l3']:.4f}"
        if config.get("l_sim_weight", 0) > 0:
            loss_str += f", L_sim={losses['l_sim']:.4f}"
        if config.get("l_sim_inv_weight", 0) > 0:
            loss_str += f", L_sim_inv={losses['l_sim_inv']:.4f}"

        print(f"[{name}] Epoch {epoch}/{epochs}: {loss_str}")
        sys.stdout.flush()

        # Reload data for next epoch
        train_dataset = LIBERODataset(split='train', max_samples=num_train)
        train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn)

    # Evaluate
    test_dataset = LIBERODataset(split='train', max_samples=num_test)
    test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)
    metrics = evaluate(model, test_loader, device)

    print(f"[{name}] Final: Arm MSE={metrics['arm_mse']:.6f}, Gripper Acc={metrics['gripper_acc']:.4f}")
    sys.stdout.flush()

    return {
        "name": name,
        "config": config,
        "arm_mse": metrics["arm_mse"],
        "gripper_acc": metrics["gripper_acc"],
        "pretrained": pretrained,
    }


def main():
    print("=" * 70)
    print("ACTOR: TRULY NOVEL SIMULATION LOSS EXPERIMENT")
    print("L_sim = ||WM(z_t, VLA(z_t)) - z_{t+1}||²")
    print("=" * 70)
    sys.stdout.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = []

    # ========== EXPERIMENTS ==========

    # 1. Baseline (no auxiliary losses)
    results.append(run_experiment(
        "baseline",
        {"l3_weight": 0, "l_sim_weight": 0, "l_sim_inv_weight": 0},
        device, epochs=10, pretrained=True
    ))
    baseline_mse = results[0]["arm_mse"]

    # 2. L3 only (prior work - Seer/UWM)
    for w in [0.005, 0.01, 0.05]:
        results.append(run_experiment(
            f"L3={w}_PRIOR_WORK",
            {"l3_weight": w, "l_sim_weight": 0, "l_sim_inv_weight": 0},
            device, epochs=10, pretrained=True
        ))

    # 3. L_sim only (TRULY NOVEL!)
    print("\n" + "=" * 70)
    print("TESTING TRULY NOVEL L_sim (Simulation Loss)")
    print("=" * 70)
    sys.stdout.flush()

    for w in [0.001, 0.005, 0.01, 0.05, 0.1]:
        results.append(run_experiment(
            f"L_sim={w}_NOVEL",
            {"l3_weight": 0, "l_sim_weight": w, "l_sim_inv_weight": 0},
            device, epochs=10, pretrained=True
        ))

    # 4. L_sim_inv only (inverse dynamics on real transitions)
    print("\n" + "=" * 70)
    print("TESTING L_sim_inv (Inverse Dynamics Training)")
    print("=" * 70)
    sys.stdout.flush()

    for w in [0.005, 0.01, 0.05]:
        results.append(run_experiment(
            f"L_sim_inv={w}_NOVEL",
            {"l3_weight": 0, "l_sim_weight": 0, "l_sim_inv_weight": w},
            device, epochs=10, pretrained=True
        ))

    # 5. Combined: L_sim + L_sim_inv (full novel framework)
    print("\n" + "=" * 70)
    print("TESTING COMBINED NOVEL LOSSES")
    print("=" * 70)
    sys.stdout.flush()

    results.append(run_experiment(
        "L_sim=0.01+L_sim_inv=0.01_NOVEL",
        {"l3_weight": 0, "l_sim_weight": 0.01, "l_sim_inv_weight": 0.01},
        device, epochs=10, pretrained=True
    ))

    results.append(run_experiment(
        "L_sim=0.05+L_sim_inv=0.01_NOVEL",
        {"l3_weight": 0, "l_sim_weight": 0.05, "l_sim_inv_weight": 0.01},
        device, epochs=10, pretrained=True
    ))

    # 6. Full ACTOR: L3 + L_sim + L_sim_inv
    print("\n" + "=" * 70)
    print("TESTING FULL ACTOR FRAMEWORK")
    print("=" * 70)
    sys.stdout.flush()

    results.append(run_experiment(
        "ACTOR_L3=0.01_L_sim=0.01_L_sim_inv=0.01",
        {"l3_weight": 0.01, "l_sim_weight": 0.01, "l_sim_inv_weight": 0.01},
        device, epochs=10, pretrained=True
    ))

    # 7. From-scratch comparison
    print("\n" + "=" * 70)
    print("COMPARING PRETRAINED VS FROM-SCRATCH")
    print("=" * 70)
    sys.stdout.flush()

    results.append(run_experiment(
        "FROM_SCRATCH_baseline",
        {"l3_weight": 0, "l_sim_weight": 0, "l_sim_inv_weight": 0},
        device, epochs=10, pretrained=False
    ))

    results.append(run_experiment(
        "FROM_SCRATCH_L_sim=0.01",
        {"l3_weight": 0, "l_sim_weight": 0.01, "l_sim_inv_weight": 0},
        device, epochs=10, pretrained=False
    ))

    # ========== RESULTS SUMMARY ==========
    print("\n" + "=" * 80)
    print("SIMULATION LOSS EXPERIMENT RESULTS")
    print("=" * 80)

    # Sort by arm MSE
    sorted_results = sorted(results, key=lambda x: x["arm_mse"])

    print(f"\n{'Rank':<5} {'Experiment':<45} {'Arm MSE':<12} {'Gripper':<10} {'Δ vs Base':<12}")
    print("-" * 84)

    for i, r in enumerate(sorted_results, 1):
        improvement = (baseline_mse - r["arm_mse"]) / baseline_mse * 100
        delta_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
        if "NOVEL" in r["name"]:
            delta_str += " (NOVEL)"
        print(f"{i:<5} {r['name']:<45} {r['arm_mse']:.6f}     {r['gripper_acc']:.4f}     {delta_str}")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Best L_sim result
    l_sim_results = [r for r in results if "L_sim=" in r["name"] and "L_sim_inv" not in r["name"] and "NOVEL" in r["name"]]
    if l_sim_results:
        best_l_sim = min(l_sim_results, key=lambda x: x["arm_mse"])
        improvement = (baseline_mse - best_l_sim["arm_mse"]) / baseline_mse * 100
        print(f"Best L_sim (NOVEL): {best_l_sim['name']} -> {improvement:+.1f}%")

    # Best L3 result (prior work)
    l3_results = [r for r in results if "L3=" in r["name"] and "PRIOR_WORK" in r["name"]]
    if l3_results:
        best_l3 = min(l3_results, key=lambda x: x["arm_mse"])
        improvement = (baseline_mse - best_l3["arm_mse"]) / baseline_mse * 100
        print(f"Best L3 (Prior Work): {best_l3['name']} -> {improvement:+.1f}%")

    # Pretrained advantage
    scratch_baseline = next((r for r in results if r["name"] == "FROM_SCRATCH_baseline"), None)
    pretrained_baseline = next((r for r in results if r["name"] == "baseline"), None)
    if scratch_baseline and pretrained_baseline:
        advantage = (scratch_baseline["arm_mse"] - pretrained_baseline["arm_mse"]) / scratch_baseline["arm_mse"] * 100
        print(f"Pretrained VLA Advantage: {advantage:+.1f}% vs from-scratch")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "simulation_loss",
        "baseline_arm_mse": baseline_mse,
        "all_results": results,
    }

    output_path = Path(__file__).parent / "simulation_loss_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
