#!/usr/bin/env python3
"""
FAST SmolVLM Experiment - Minimal version to get results quickly.
Uses REAL pretrained SmolVLM weights.
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

from transformers import SmolVLMForConditionalGeneration, SmolVLMProcessor
from datasets import load_dataset

print("=" * 70)
print("FAST SmolVLM EXPERIMENT")
print("Using REAL pretrained weights!")
print("=" * 70)


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
        return self.encoder.encode(images, states, texts)

    def forward_policy(self, z):
        return self.policy(z)

    def forward_world_model(self, z, action):
        return self.world_model(torch.cat([z, action], dim=-1))

    def forward_inverse_dynamics(self, z_t, z_next):
        return self.inverse_dynamics(torch.cat([z_t, z_next], dim=-1))


def train_and_eval(name, config, encoder, device, num_train=200, num_test=100, epochs=2):
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Config: {config}")
    print(f"{'='*60}")
    sys.stdout.flush()

    model = Model(encoder, device=device)
    trainable = list(model.world_model.parameters()) + list(model.policy.parameters()) + \
                list(model.inverse_dynamics.parameters()) + list(model.encoder.state_encoder.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)

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

    # Evaluate
    model.eval()
    test_data = LIBERODataset(split='train', max_samples=num_test)
    test_loader = DataLoader(test_data, batch_size=8, collate_fn=collate_fn)

    arm_mse = 0
    grip_correct = 0
    n_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch['images']
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            texts = batch['texts']

            z = model.encode(images, states, texts)
            pred = model.forward_policy(z)

            arm_mse += F.mse_loss(pred[:, :6], actions[:, :6], reduction="sum").item()
            grip_correct += ((torch.sigmoid(pred[:, 6:]) > 0.5).float() == actions[:, 6:].round()).sum().item()
            n_samples += len(images)

    arm_mse /= max(n_samples, 1)
    grip_acc = grip_correct / max(n_samples, 1)

    print(f"[{name}] FINAL: Arm MSE={arm_mse:.6f}, Gripper Acc={grip_acc:.4f}")
    sys.stdout.flush()

    return {"name": name, "arm_mse": arm_mse, "gripper_acc": grip_acc, "config": config}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load encoder once
    encoder = SmolVLMEncoder(device=device)
    results = []

    # Run experiments
    results.append(train_and_eval("baseline", {}, encoder, device))
    baseline_mse = results[0]["arm_mse"]

    results.append(train_and_eval("L3=0.01_PRIOR", {"l3": 0.01}, encoder, device))
    results.append(train_and_eval("L_sim=0.01_NOVEL", {"l_sim": 0.01}, encoder, device))
    results.append(train_and_eval("L_sim_inv=0.01_NOVEL", {"l_sim_inv": 0.01}, encoder, device))

    # Results
    print("\n" + "=" * 70)
    print("RESULTS (REAL SmolVLM)")
    print("=" * 70)

    for r in sorted(results, key=lambda x: x["arm_mse"]):
        imp = (baseline_mse - r["arm_mse"]) / baseline_mse * 100
        mark = "*" if "NOVEL" in r["name"] else ""
        print(f"{r['name']:<30} Arm MSE={r['arm_mse']:.6f}  Δ={imp:+.1f}% {mark}")

    # Save
    output = {"timestamp": datetime.now().isoformat(), "model": "SmolVLM (REAL)", "results": results}
    Path("fast_smolvlm_results.json").write_text(json.dumps(output, indent=2))
    print("\nSaved to fast_smolvlm_results.json")


if __name__ == "__main__":
    main()
