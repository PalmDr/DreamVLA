#!/usr/bin/env python3
"""
ACTOR: Simulation Loss Experiment with REAL SmolVLM2 Model

This uses the ACTUAL pretrained SmolVLM2 model (the VLM backbone of SmolVLA)
with real pretrained weights from HuggingFace.

SmolVLA architecture:
- VLM backbone: SmolVLM2 (SigLIP vision encoder + SmolLM2 language decoder)
- Action expert: Transformer with flow matching

We use SmolVLM2 as the encoder and add World Model + Inverse Dynamics heads.

Run with: PYTHONUNBUFFERED=1 uv run python run_real_smolvla_experiment.py
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
from tqdm import tqdm

# Check for required packages
try:
    from transformers import SmolVLMForConditionalGeneration, SmolVLMProcessor
    print("Transformers loaded successfully (SmolVLM classes available)")
except ImportError as e:
    print(f"Error importing transformers: {e}")
    print("Please install: pip install transformers>=4.45.0")
    sys.exit(1)

from datasets import load_dataset


class LIBERODataset(IterableDataset):
    """LIBERO dataset with consecutive frame pairs for real SmolVLM2."""

    def __init__(self, split='train', max_samples=2000, processor=None):
        self.max_samples = max_samples
        self.processor = processor
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
                    'language': sample.get('language_instruction', 'pick up the object'),
                }
                count += 1

            prev_sample = sample
            prev_img = img
            prev_state = state
            prev_action = action


class SmolVLM2Encoder(nn.Module):
    """
    Encoder using REAL pretrained SmolVLM2 model.

    SmolVLM2 is the VLM backbone used in SmolVLA.
    We extract features from the vision encoder for our world model.
    """

    def __init__(self, model_name="HuggingFaceTB/SmolVLM-Instruct", device="cuda"):
        super().__init__()
        self.device = device

        print(f"Loading REAL SmolVLM model: {model_name}")
        print("This uses actual pretrained weights from HuggingFace!")
        sys.stdout.flush()

        # Load the real pretrained model using SmolVLM-specific classes
        self.processor = SmolVLMProcessor.from_pretrained(model_name)
        self.model = SmolVLMForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(device)

        # Freeze the pretrained model - we only train the heads
        for param in self.model.parameters():
            param.requires_grad = False

        # Get the vision hidden dimension (different from text hidden dimension!)
        self.vision_hidden_dim = self.model.config.vision_config.hidden_size
        self.text_hidden_dim = self.model.config.text_config.hidden_size
        print(f"SmolVLM vision hidden dim: {self.vision_hidden_dim}")
        print(f"SmolVLM text hidden dim: {self.text_hidden_dim}")

        # Use vision hidden dimension as our latent space
        self.hidden_dim = self.vision_hidden_dim

        # State encoder to match vision hidden dimension
        self.state_encoder = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(),
            nn.Linear(256, self.hidden_dim),
        ).to(device)

        print("SmolVLM encoder initialized with REAL pretrained weights!")
        sys.stdout.flush()

    def encode_image(self, images, texts=None):
        """
        Encode images using the real SmolVLM vision encoder.
        Returns the pooled visual features.
        """
        if texts is None:
            texts = ["describe this robot scene"] * len(images)

        # Process images one at a time to avoid batching issues with SmolVLM
        all_features = []

        with torch.no_grad():
            for img, text in zip(images, texts):
                # Format text with image placeholder
                formatted_text = f"<image>User: {text}\nAssistant:"

                # Process single image
                inputs = self.processor(
                    text=formatted_text,
                    images=[img],
                    return_tensors="pt",
                ).to(self.device)

                # Get pixel values
                pixel_values = inputs.pixel_values.to(torch.float16)

                # Handle different pixel_values shapes
                # SmolVLM can output (batch, num_images, channels, height, width)
                # or (batch, num_crops, channels, height, width)
                original_shape = pixel_values.shape

                if len(pixel_values.shape) == 5:
                    # Shape: (batch, num_images/crops, C, H, W)
                    b, n, c, h, w = pixel_values.shape
                    pixel_values = pixel_values.view(b * n, c, h, w)

                # Get vision encoder outputs directly
                vision_outputs = self.model.model.vision_model(pixel_values)

                # vision_outputs.last_hidden_state shape: (num_patches, seq_len, hidden_dim)
                # We want one feature vector per original image

                # Pool across all patches and sequence length
                # First pool over sequence length (dim=1), then over num_patches (dim=0)
                features = vision_outputs.last_hidden_state  # (N, seq, hidden)
                features = features.mean(dim=1)  # (N, hidden)
                features = features.mean(dim=0, keepdim=True)  # (1, hidden)

                all_features.append(features)

        # Stack all features
        visual_features = torch.cat(all_features, dim=0)  # (batch, hidden_dim)
        return visual_features.float()

    def encode(self, images, states, texts=None):
        """
        Encode images and states into a joint latent representation.
        """
        visual_features = self.encode_image(images, texts)
        state_features = self.state_encoder(states)

        # Combine visual and state features
        combined = visual_features + state_features  # Residual combination

        return combined


class WorldModelVLAReal(nn.Module):
    """
    World Model + VLA with REAL SmolVLM2 encoder.

    Uses actual pretrained SmolVLM2 weights for encoding,
    with trainable World Model, Policy, and Inverse Dynamics heads.
    """

    def __init__(self, encoder, action_dim=7, device="cuda"):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = encoder.hidden_dim
        self.device = device

        # World model: predicts next latent state given current state and action
        self.world_model = nn.Sequential(
            nn.Linear(self.hidden_dim + action_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        ).to(device)

        # Policy head (VLA action prediction)
        self.policy = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, action_dim),
        ).to(device)

        # Inverse dynamics head
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, action_dim),
        ).to(device)

        print(f"WorldModelVLAReal initialized:")
        print(f"  - Encoder: SmolVLM2 (PRETRAINED, FROZEN)")
        print(f"  - Hidden dim: {self.hidden_dim}")
        print(f"  - Action dim: {action_dim}")
        print(f"  - Trainable params: World Model, Policy, Inverse Dynamics heads")
        sys.stdout.flush()

    def encode(self, images, states, texts=None):
        return self.encoder.encode(images, states, texts)

    def forward_policy(self, z):
        return self.policy(z)

    def forward_world_model(self, z, action):
        return self.world_model(torch.cat([z, action], dim=-1))

    def forward_inverse_dynamics(self, z_t, z_next):
        return self.inverse_dynamics(torch.cat([z_t, z_next], dim=-1))


def compute_action_loss(pred, target):
    """Standard action prediction loss."""
    arm_loss = F.smooth_l1_loss(pred[:, :6], target[:, :6])
    gripper_loss = F.binary_cross_entropy(
        torch.sigmoid(pred[:, 6:]),
        target[:, 6:].clamp(0, 1)
    )
    return arm_loss + 0.1 * gripper_loss


def compute_l3_loss(model, z_t, action):
    """
    L3: Action Consistency Loss (EXISTS IN PRIOR WORK - Seer, UWM)
    L3 = ||ID(z_t, WM(z_t, a)) - a||²
    """
    z_pred = model.forward_world_model(z_t, action)
    action_recovered = model.forward_inverse_dynamics(z_t, z_pred)
    return F.mse_loss(action_recovered[:, :6], action[:, :6].detach())


def compute_l_sim(model, z_t, z_t1_real, vla_pred):
    """
    L_sim: Simulation Loss (TRULY NOVEL!)
    L_sim = ||WM(z_t, VLA(z_t)) - z_{t+1}||²

    Checks if simulating VLA's predicted action produces correct next state.
    """
    z_t1_predicted = model.forward_world_model(z_t, vla_pred)
    return F.mse_loss(z_t1_predicted, z_t1_real.detach())


def compute_l_sim_inv(model, z_t, z_t1_real, actions):
    """
    L_sim_inv: Inverse Dynamics Training on real transitions
    L_sim_inv = ||ID(z_t, z_{t+1}) - a_t||²
    """
    id_pred = model.forward_inverse_dynamics(z_t, z_t1_real)
    return F.mse_loss(id_pred[:, :6], actions[:, :6].detach())


def train_epoch(model, dataloader, optimizer, config, device):
    """Train one epoch with the real SmolVLM2 model."""
    model.train()
    # Keep encoder in eval mode (frozen)
    model.encoder.model.eval()

    total_losses = {"total": 0, "action": 0, "l3": 0, "l_sim": 0, "l_sim_inv": 0}
    n_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch['images']  # List of PIL images
        next_images = batch['next_images']
        states = batch['states'].to(device)
        next_states = batch['next_states'].to(device)
        actions = batch['actions'].to(device)
        texts = batch['texts']

        optimizer.zero_grad()

        # Encode using REAL SmolVLM2
        z_t = model.encode(images, states, texts)
        z_t1_real = model.encode(next_images, next_states, texts)

        # VLA prediction
        vla_pred = model.forward_policy(z_t)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_losses["total"] += loss.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    arm_mse_total = 0
    gripper_correct = 0
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch['images']
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            texts = batch['texts']

            z = model.encode(images, states, texts)
            pred = model.forward_policy(z)

            arm_mse_total += F.mse_loss(pred[:, :6], actions[:, :6], reduction="sum").item()
            gripper_pred = (torch.sigmoid(pred[:, 6:]) > 0.5).float()
            gripper_correct += (gripper_pred == actions[:, 6:].round()).sum().item()
            n_samples += len(images)

    return {
        "arm_mse": arm_mse_total / max(n_samples, 1),
        "gripper_acc": gripper_correct / max(n_samples, 1),
    }


def collate_fn(batch):
    """Collate function that keeps images as PIL for SmolVLM2 processor."""
    return {
        'images': [b['image'] for b in batch],
        'next_images': [b['next_image'] for b in batch],
        'states': torch.stack([b['state'] for b in batch]),
        'next_states': torch.stack([b['next_state'] for b in batch]),
        'actions': torch.stack([b['action'] for b in batch]),
        'texts': [b['language'] for b in batch],
    }


def run_experiment(name, config, encoder, device, epochs=3, num_train=500, num_test=100):
    """Run a single experiment with the real SmolVLM2 model."""
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    config_str = ", ".join([f"{k}={v}" for k, v in config.items() if v > 0])
    print(f"Config: {config_str if config_str else 'baseline'}")
    print(f"Using REAL SmolVLM2 pretrained encoder!")
    print(f"{'='*60}")
    sys.stdout.flush()

    # Create model with shared encoder
    model = WorldModelVLAReal(encoder, device=device)

    # Only optimize the trainable heads
    trainable_params = list(model.world_model.parameters()) + \
                       list(model.policy.parameters()) + \
                       list(model.inverse_dynamics.parameters()) + \
                       list(model.encoder.state_encoder.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    # Train
    for epoch in range(1, epochs + 1):
        # Reload data each epoch (streaming dataset)
        train_dataset = LIBERODataset(split='train', max_samples=num_train)
        train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn)

        losses = train_epoch(model, train_loader, optimizer, config, device)

        loss_str = f"loss={losses['total']:.4f}, action={losses['action']:.4f}"
        if config.get("l3_weight", 0) > 0:
            loss_str += f", L3={losses['l3']:.4f}"
        if config.get("l_sim_weight", 0) > 0:
            loss_str += f", L_sim={losses['l_sim']:.4f}"
        if config.get("l_sim_inv_weight", 0) > 0:
            loss_str += f", L_sim_inv={losses['l_sim_inv']:.4f}"

        print(f"[{name}] Epoch {epoch}/{epochs}: {loss_str}")
        sys.stdout.flush()

    # Evaluate
    test_dataset = LIBERODataset(split='train', max_samples=num_test)
    test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)
    metrics = evaluate(model, test_loader, device)

    print(f"[{name}] FINAL: Arm MSE={metrics['arm_mse']:.6f}, Gripper Acc={metrics['gripper_acc']:.4f}")
    sys.stdout.flush()

    return {
        "name": name,
        "config": config,
        "arm_mse": metrics["arm_mse"],
        "gripper_acc": metrics["gripper_acc"],
        "model": "SmolVLM2-256M-Video-Instruct (REAL PRETRAINED)",
    }


def main():
    print("=" * 70)
    print("ACTOR: REAL SmolVLM2 EXPERIMENT")
    print("Using ACTUAL pretrained SmolVLM2 weights from HuggingFace!")
    print("=" * 70)
    sys.stdout.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cpu":
        print("WARNING: Running on CPU will be very slow!")

    # Load the REAL pretrained SmolVLM2 encoder (shared across experiments)
    print("\n" + "=" * 70)
    print("LOADING REAL PRETRAINED SmolVLM2 MODEL")
    print("=" * 70)
    encoder = SmolVLM2Encoder(device=device)

    results = []

    # ========== KEY EXPERIMENTS (FAST VERSION) ==========
    # Focus on baseline vs L_sim vs L3 comparison

    # 1. Baseline (no auxiliary losses)
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: BASELINE (VLA only)")
    print("=" * 70)
    results.append(run_experiment(
        "baseline",
        {"l3_weight": 0, "l_sim_weight": 0, "l_sim_inv_weight": 0},
        encoder, device
    ))
    baseline_mse = results[0]["arm_mse"]

    # 2. L3 (prior work - Seer/UWM)
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: L3 (PRIOR WORK)")
    print("=" * 70)
    results.append(run_experiment(
        "L3=0.01_PRIOR_WORK",
        {"l3_weight": 0.01, "l_sim_weight": 0, "l_sim_inv_weight": 0},
        encoder, device
    ))

    # 3. L_sim (NOVEL!)
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: L_sim (TRULY NOVEL!)")
    print("=" * 70)
    results.append(run_experiment(
        "L_sim=0.01_NOVEL",
        {"l3_weight": 0, "l_sim_weight": 0.01, "l_sim_inv_weight": 0},
        encoder, device
    ))

    # 4. L_sim_inv (NOVEL!)
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: L_sim_inv (NOVEL!)")
    print("=" * 70)
    results.append(run_experiment(
        "L_sim_inv=0.01_NOVEL",
        {"l3_weight": 0, "l_sim_weight": 0, "l_sim_inv_weight": 0.01},
        encoder, device
    ))

    # ========== RESULTS SUMMARY ==========
    print("\n" + "=" * 80)
    print("REAL SmolVLM2 EXPERIMENT RESULTS")
    print("Using ACTUAL pretrained weights!")
    print("=" * 80)

    sorted_results = sorted(results, key=lambda x: x["arm_mse"])

    print(f"\n{'Rank':<5} {'Experiment':<40} {'Arm MSE':<12} {'Gripper':<10} {'vs Base':<12}")
    print("-" * 80)

    for i, r in enumerate(sorted_results, 1):
        improvement = (baseline_mse - r["arm_mse"]) / baseline_mse * 100
        delta_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
        if "NOVEL" in r["name"]:
            delta_str += " *"
        print(f"{i:<5} {r['name']:<40} {r['arm_mse']:.6f}     {r['gripper_acc']:.4f}     {delta_str}")

    print("\n* = NOVEL contribution (not in prior work)")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS (REAL SmolVLM2)")
    print("=" * 80)

    # Best L_sim
    l_sim_results = [r for r in results if "L_sim=" in r["name"] and "L_sim_inv" not in r["name"]]
    if l_sim_results:
        best = min(l_sim_results, key=lambda x: x["arm_mse"])
        imp = (baseline_mse - best["arm_mse"]) / baseline_mse * 100
        print(f"Best L_sim (NOVEL): {best['name']} -> {imp:+.1f}%")

    # Best L3
    l3_results = [r for r in results if "L3=" in r["name"]]
    if l3_results:
        best = min(l3_results, key=lambda x: x["arm_mse"])
        imp = (baseline_mse - best["arm_mse"]) / baseline_mse * 100
        print(f"Best L3 (Prior): {best['name']} -> {imp:+.1f}%")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "real_smolvlm2_simulation_loss",
        "model": "SmolVLM2-256M-Video-Instruct (REAL PRETRAINED)",
        "baseline_arm_mse": baseline_mse,
        "all_results": results,
    }

    output_path = Path(__file__).parent / "real_smolvlm2_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("All results use REAL SmolVLM2 pretrained weights!")
    print("=" * 80)


if __name__ == "__main__":
    main()
