#!/usr/bin/env python3
"""
Multi-seed Experiment Runner for ACTOR Paper
Runs comprehensive experiments with multiple seeds for statistical significance.
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

# Set up proper seeding
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from transformers import SmolVLMForConditionalGeneration, SmolVLMProcessor
from datasets import load_dataset

print("=" * 70)
print("ACTOR MULTI-SEED EXPERIMENT")
print("=" * 70)


class SmolVLMEncoder(nn.Module):
    """Encoder using real pretrained SmolVLM."""

    def __init__(self, model_name="HuggingFaceTB/SmolVLM-Instruct", device="cuda"):
        super().__init__()
        self.device = device

        print(f"Loading SmolVLM: {model_name}")
        self.processor = SmolVLMProcessor.from_pretrained(model_name)
        self.model = SmolVLMForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)

        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_dim = self.model.config.vision_config.hidden_size
        print(f"Vision hidden dim: {self.hidden_dim}")

    def build_state_encoder(self):
        """Build fresh state encoder for each seed."""
        return nn.Sequential(
            nn.Linear(8, 256), nn.ReLU(), nn.Linear(256, self.hidden_dim)
        ).to(self.device)

    def encode_image(self, images, texts):
        """Encode images with text context."""
        all_features = []
        with torch.no_grad():
            for img, text in zip(images, texts):
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img.astype(np.uint8))

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


class ACTORModel(nn.Module):
    """ACTOR model with world model and policy."""

    def __init__(self, encoder, state_encoder, action_dim=7, device="cuda"):
        super().__init__()
        self.encoder = encoder
        self.state_encoder = state_encoder
        self.hidden_dim = encoder.hidden_dim
        self.device = device

        self.world_model = nn.Sequential(
            nn.Linear(self.hidden_dim + action_dim, self.hidden_dim * 2),
            nn.ReLU(), nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        ).to(device)

        self.policy = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(), nn.Linear(self.hidden_dim // 2, action_dim)
        ).to(device)

        self.inverse_dynamics = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(), nn.Linear(self.hidden_dim, action_dim)
        ).to(device)

    def encode(self, images, states, texts):
        visual = self.encoder.encode_image(images, texts)
        state_emb = self.state_encoder(states)
        return visual + state_emb

    def forward_policy(self, z):
        return self.policy(z)

    def forward_world_model(self, z, action):
        return self.world_model(torch.cat([z, action], dim=-1))

    def forward_inverse_dynamics(self, z_t, z_next):
        return self.inverse_dynamics(torch.cat([z_t, z_next], dim=-1))


def compute_metrics(model, test_loader, device):
    """Compute comprehensive evaluation metrics."""
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['images']
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            texts = batch['texts']

            z = model.encode(images, states, texts)
            pred = model.forward_policy(z)

            all_preds.append(pred.cpu())
            all_targets.append(actions.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # MSE per dimension
    mse_per_dim = F.mse_loss(preds, targets, reduction='none').mean(dim=0)

    # Arm MSE (first 6 dimensions)
    arm_mse = F.mse_loss(preds[:, :6], targets[:, :6]).item()

    # Position error (L2 norm of position delta)
    pos_error = torch.norm(preds[:, :3] - targets[:, :3], dim=-1)

    # Rotation error (L2 norm of rotation delta)
    rot_error = torch.norm(preds[:, 3:6] - targets[:, 3:6], dim=-1)

    # Gripper accuracy
    pred_gripper = (torch.sigmoid(preds[:, 6]) > 0.5).float()
    target_gripper = (targets[:, 6] > 0.5).float()
    gripper_acc = (pred_gripper == target_gripper).float().mean().item()

    # Prediction magnitude ratio (important for understanding policy behavior)
    pred_mag = torch.norm(preds[:, :6], dim=-1).mean().item()
    target_mag = torch.norm(targets[:, :6], dim=-1).mean().item()

    return {
        'arm_mse': arm_mse,
        'position_error_mean': pos_error.mean().item(),
        'position_error_std': pos_error.std().item(),
        'rotation_error_mean': rot_error.mean().item(),
        'rotation_error_std': rot_error.std().item(),
        'gripper_accuracy': gripper_acc,
        'pred_magnitude': pred_mag,
        'target_magnitude': target_mag,
        'mse_x': mse_per_dim[0].item(),
        'mse_y': mse_per_dim[1].item(),
        'mse_z': mse_per_dim[2].item(),
    }


def train_and_evaluate(config, encoder, device, seed, num_train=1000, num_test=300, epochs=5, batch_size=8):
    """Train a model with given configuration and seed, return metrics."""
    from torch.utils.data import DataLoader, IterableDataset

    set_seed(seed)

    class LIBERODataset(IterableDataset):
        def __init__(self, split='train', max_samples=200):
            self.max_samples = max_samples
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

    def collate_fn(batch):
        return {
            'images': [b['image'] for b in batch],
            'next_images': [b['next_image'] for b in batch],
            'states': torch.stack([b['state'] for b in batch]),
            'next_states': torch.stack([b['next_state'] for b in batch]),
            'actions': torch.stack([b['action'] for b in batch]),
            'texts': [b['language'] for b in batch],
        }

    # Fresh state encoder for this seed
    state_encoder = encoder.build_state_encoder()
    model = ACTORModel(encoder, state_encoder, device=device)

    trainable = (
        list(model.world_model.parameters()) +
        list(model.policy.parameters()) +
        list(model.inverse_dynamics.parameters()) +
        list(model.state_encoder.parameters())
    )
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)

    # Training
    for epoch in range(1, epochs + 1):
        model.train()
        model.encoder.model.eval()

        train_data = LIBERODataset(split='train', max_samples=num_train)
        train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)

        total_loss = 0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Seed {seed} Epoch {epoch}", leave=False):
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

    # Evaluation
    test_data = LIBERODataset(split='train', max_samples=num_test)  # Using later samples as test
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)

    metrics = compute_metrics(model, test_loader, device)
    metrics['final_train_loss'] = total_loss / max(n_batches, 1)

    return metrics, model


def compute_significance(baseline_values, treatment_values):
    """Compute statistical significance using paired t-test."""
    if len(baseline_values) < 2 or len(treatment_values) < 2:
        return {'p_value': 1.0, 'significant': False}

    t_stat, p_value = stats.ttest_ind(baseline_values, treatment_values)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(baseline_values)**2 + np.std(treatment_values)**2) / 2)
    cohens_d = (np.mean(treatment_values) - np.mean(baseline_values)) / pooled_std if pooled_std > 0 else 0

    return {
        'p_value': p_value,
        't_statistic': t_stat,
        'cohens_d': cohens_d,
        'significant_005': p_value < 0.05,
        'significant_001': p_value < 0.01,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Experiment settings
    NUM_SEEDS = 5
    NUM_TRAIN = 1000  # Increased from 500
    NUM_TEST = 300
    EPOCHS = 5  # Increased from 3

    print(f"\nExperiment Settings:")
    print(f"  Seeds: {NUM_SEEDS}")
    print(f"  Training samples: {NUM_TRAIN}")
    print(f"  Test samples: {NUM_TEST}")
    print(f"  Epochs: {EPOCHS}")

    # Load encoder (shared across all experiments)
    encoder = SmolVLMEncoder(device=device)

    # Configurations to evaluate
    configs = [
        ("Baseline", {}),
        ("L3=0.01", {"l3": 0.01}),
        ("L_sim=0.01", {"l_sim": 0.01}),
        ("L_sim_inv=0.01", {"l_sim_inv": 0.01}),
        ("Combined", {"l3": 0.005, "l_sim": 0.005, "l_sim_inv": 0.005}),
    ]

    all_results = {}

    for name, config in configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {name}")
        print(f"{'='*60}")

        seed_results = []

        for seed in range(NUM_SEEDS):
            print(f"\n--- Seed {seed} ---")

            metrics, model = train_and_evaluate(
                config, encoder, device,
                seed=seed,
                num_train=NUM_TRAIN,
                num_test=NUM_TEST,
                epochs=EPOCHS
            )

            seed_results.append(metrics)
            print(f"  Arm MSE: {metrics['arm_mse']:.5f}")
            print(f"  Position Error: {metrics['position_error_mean']:.4f} +/- {metrics['position_error_std']:.4f}")
            print(f"  Gripper Acc: {metrics['gripper_accuracy']:.2%}")

            # Save checkpoint
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save({
                'config': config,
                'seed': seed,
                'state_encoder': model.state_encoder.state_dict(),
                'world_model': model.world_model.state_dict(),
                'policy': model.policy.state_dict(),
                'inverse_dynamics': model.inverse_dynamics.state_dict(),
                'metrics': metrics,
            }, checkpoint_dir / f"{name.replace('=', '_').replace('.', 'p')}_seed{seed}.pt")

        # Aggregate across seeds
        all_results[name] = {
            'config': config,
            'seeds': seed_results,
            'arm_mse_mean': np.mean([r['arm_mse'] for r in seed_results]),
            'arm_mse_std': np.std([r['arm_mse'] for r in seed_results]),
            'position_error_mean': np.mean([r['position_error_mean'] for r in seed_results]),
            'position_error_std': np.std([r['position_error_mean'] for r in seed_results]),
            'gripper_accuracy_mean': np.mean([r['gripper_accuracy'] for r in seed_results]),
            'gripper_accuracy_std': np.std([r['gripper_accuracy'] for r in seed_results]),
        }

    # Compute statistical significance vs baseline
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*70)

    baseline_arm_mse = [r['arm_mse'] for r in all_results['Baseline']['seeds']]

    for name in configs[1:]:  # Skip baseline
        name = name[0]
        treatment_arm_mse = [r['arm_mse'] for r in all_results[name]['seeds']]

        sig_results = compute_significance(baseline_arm_mse, treatment_arm_mse)
        all_results[name]['significance'] = sig_results

        improvement = (1 - all_results[name]['arm_mse_mean'] / all_results['Baseline']['arm_mse_mean']) * 100

        print(f"\n{name} vs Baseline:")
        print(f"  Improvement: {improvement:+.1f}%")
        print(f"  p-value: {sig_results['p_value']:.4f}")
        print(f"  Cohen's d: {sig_results['cohens_d']:.3f}")
        print(f"  Significant (p<0.05): {'YES' if sig_results['significant_005'] else 'NO'}")
        print(f"  Significant (p<0.01): {'YES' if sig_results['significant_001'] else 'NO'}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE (for paper)")
    print("="*70)
    print(f"\n{'Method':<20} {'Arm MSE':<20} {'Position Error':<20} {'Gripper Acc':<15}")
    print("-"*75)

    for name, results in all_results.items():
        arm_str = f"{results['arm_mse_mean']:.4f} +/- {results['arm_mse_std']:.4f}"
        pos_str = f"{results['position_error_mean']:.3f} +/- {results['position_error_std']:.3f}"
        grip_str = f"{results['gripper_accuracy_mean']:.1%} +/- {results['gripper_accuracy_std']:.1%}"
        print(f"{name:<20} {arm_str:<20} {pos_str:<20} {grip_str:<15}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "num_seeds": NUM_SEEDS,
            "num_train": NUM_TRAIN,
            "num_test": NUM_TEST,
            "epochs": EPOCHS,
        },
        "results": {k: {kk: vv for kk, vv in v.items() if kk != 'seeds'}
                   for k, v in all_results.items()},
        "seed_results": {k: v['seeds'] for k, v in all_results.items()},
    }

    output_path = Path("multiseed_results.json")
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {output_path}")

    # Print LaTeX table
    print("\n" + "="*70)
    print("LATEX TABLE")
    print("="*70)
    print(r"""
\begin{table}[h]
\centering
\caption{Action prediction performance on LIBERO (5 seeds). Best results in \textbf{bold}.}
\begin{tabular}{lccc}
\toprule
Method & Arm MSE $\downarrow$ & Position Error $\downarrow$ & Gripper Acc $\uparrow$ \\
\midrule""")

    best_arm = min(all_results.items(), key=lambda x: x[1]['arm_mse_mean'])[0]
    best_pos = min(all_results.items(), key=lambda x: x[1]['position_error_mean'])[0]
    best_grip = max(all_results.items(), key=lambda x: x[1]['gripper_accuracy_mean'])[0]

    for name, results in all_results.items():
        arm_str = f"{results['arm_mse_mean']:.4f}$\\pm${results['arm_mse_std']:.4f}"
        pos_str = f"{results['position_error_mean']:.3f}$\\pm${results['position_error_std']:.3f}"
        grip_str = f"{results['gripper_accuracy_mean']*100:.1f}$\\pm${results['gripper_accuracy_std']*100:.1f}"

        if name == best_arm:
            arm_str = r"\textbf{" + arm_str + "}"
        if name == best_pos:
            pos_str = r"\textbf{" + pos_str + "}"
        if name == best_grip:
            grip_str = r"\textbf{" + grip_str + "}"

        print(f"{name} & {arm_str} & {pos_str} & {grip_str} \\\\")

    print(r"""\bottomrule
\end{tabular}
\label{tab:action_prediction}
\end{table}""")


if __name__ == "__main__":
    main()
