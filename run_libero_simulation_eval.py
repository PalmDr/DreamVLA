#!/usr/bin/env python3
"""
LIBERO Simulation Evaluation
Evaluates trained models on task success rate in LIBERO simulation environments.
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add LIBERO to path
sys.path.insert(0, '/home/jfy/dev/LIBERO')

from transformers import SmolVLMForConditionalGeneration, SmolVLMProcessor
from datasets import load_dataset

# LIBERO imports
try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    LIBERO_AVAILABLE = True
except ImportError:
    print("WARNING: LIBERO not available, will use mock evaluation")
    LIBERO_AVAILABLE = False

print("=" * 70)
print("LIBERO SIMULATION EVALUATION")
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

        self.state_encoder = nn.Sequential(
            nn.Linear(8, 256), nn.ReLU(), nn.Linear(256, self.hidden_dim)
        ).to(device)

    def encode_image(self, images, texts):
        """Encode images with text context."""
        all_features = []
        with torch.no_grad():
            for img, text in zip(images, texts):
                # Handle both PIL images and numpy arrays
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

    def encode(self, images, states, texts):
        """Encode images and states."""
        visual = self.encode_image(images, texts)
        state_emb = self.state_encoder(states)
        return visual + state_emb


class ACTORModel(nn.Module):
    """ACTOR model with world model and policy."""

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


def get_libero_image(obs, target_size=224):
    """Extract and preprocess image from LIBERO observation."""
    img = obs["agentview_image"]
    # Rotate 180 degrees (LIBERO convention)
    img = img[::-1, ::-1]
    # Resize if needed
    if img.shape[0] != target_size:
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray(img)
        img_pil = img_pil.resize((target_size, target_size), PILImage.LANCZOS)
        img = np.array(img_pil)
    return img


def get_libero_state(obs):
    """Extract state from LIBERO observation."""
    # LIBERO provides robot state in observation
    if "robot0_proprio-state" in obs:
        state = obs["robot0_proprio-state"]
    else:
        # Fallback: construct from joint positions and gripper
        state = np.zeros(8)
        if "robot0_eef_pos" in obs:
            state[:3] = obs["robot0_eef_pos"]
        if "robot0_eef_quat" in obs:
            state[3:7] = obs["robot0_eef_quat"]
        if "robot0_gripper_qpos" in obs:
            state[7] = obs["robot0_gripper_qpos"][0]
    return state


def evaluate_in_simulation(model, task, num_episodes=10, max_steps=300, device="cuda"):
    """Evaluate model on a single LIBERO task."""
    if not LIBERO_AVAILABLE:
        print("LIBERO not available, returning mock results")
        return {"success_rate": 0.0, "avg_steps": 0, "episodes": []}

    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    print(f"\nTask: {task_description}")
    print(f"BDDL: {task_bddl_file}")

    successes = []
    episode_lengths = []

    for episode in range(num_episodes):
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 256,
            "camera_widths": 256
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(episode)

        obs = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            # Get observation
            img = get_libero_image(obs)
            state = get_libero_state(obs)

            # Convert to tensors
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            # Get action from policy
            with torch.no_grad():
                z = model.encode([img], state_tensor, [task_description])
                action_pred = model.forward_policy(z)

            # Convert action to numpy
            action = action_pred[0].cpu().numpy()

            # Process gripper action (sigmoid -> binary)
            action[6] = 1.0 if action[6] > 0 else -1.0

            # Step environment
            obs, reward, done, info = env.step(action)
            step += 1

            if reward > 0:
                done = True
                successes.append(True)
                episode_lengths.append(step)
                break

        if not reward > 0:
            successes.append(False)
            episode_lengths.append(step)

        env.close()

        status = "SUCCESS" if successes[-1] else "FAIL"
        print(f"  Episode {episode+1}/{num_episodes}: {status} (steps: {episode_lengths[-1]})")

    success_rate = sum(successes) / len(successes) if successes else 0.0
    avg_steps = np.mean(episode_lengths) if episode_lengths else 0.0

    return {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "episodes": [{"success": s, "steps": l} for s, l in zip(successes, episode_lengths)]
    }


def train_model(config, encoder, device, num_train=500, epochs=3):
    """Train a model with given configuration."""
    from torch.utils.data import DataLoader, IterableDataset

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
                if not isinstance(img, Image):
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

    model = ACTORModel(encoder, device=device)
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

        print(f"  Epoch {epoch}: loss={total_loss/n_batches:.4f}")

    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Check LIBERO availability
    if not LIBERO_AVAILABLE:
        print("\nWARNING: LIBERO simulation not available.")
        print("This script requires LIBERO to be properly installed with mujoco.")
        print("Proceeding with action prediction evaluation only.\n")

    # Load encoder
    encoder = SmolVLMEncoder(device=device)

    # Configurations to evaluate
    configs = [
        ("baseline", {}),
        ("L3=0.01", {"l3": 0.01}),
        ("L_sim=0.01", {"l_sim": 0.01}),
        ("L_sim_inv=0.01", {"l_sim_inv": 0.01}),
        ("Combined", {"l3": 0.005, "l_sim": 0.005, "l_sim_inv": 0.005}),
    ]

    results = []

    # Get LIBERO benchmark
    if LIBERO_AVAILABLE:
        BenchmarkClass = benchmark.get_benchmark_dict()['libero_spatial']
        bm = BenchmarkClass()
        task_names = bm.get_task_names()
        # Use first 3 tasks for evaluation (can be expanded)
        eval_tasks = [bm.get_task(i) for i in range(min(3, len(task_names)))]
    else:
        eval_tasks = []

    for name, config in configs:
        print(f"\n{'='*60}")
        print(f"Training and evaluating: {name}")
        print(f"Config: {config}")
        print(f"{'='*60}")

        # Train model
        model = train_model(config, encoder, device, num_train=300, epochs=2)

        result = {
            "name": name,
            "config": config,
            "task_results": []
        }

        if LIBERO_AVAILABLE and eval_tasks:
            # Evaluate in simulation
            for task in eval_tasks:
                task_result = evaluate_in_simulation(
                    model, task,
                    num_episodes=5,  # 5 episodes per task for quick eval
                    max_steps=200,
                    device=device
                )
                task_result["task_name"] = task.language
                result["task_results"].append(task_result)

            # Aggregate results
            all_success_rates = [r["success_rate"] for r in result["task_results"]]
            result["avg_success_rate"] = np.mean(all_success_rates)
            result["std_success_rate"] = np.std(all_success_rates)
        else:
            result["avg_success_rate"] = None
            result["note"] = "LIBERO simulation not available"

        results.append(result)
        print(f"\n[{name}] Avg Success Rate: {result.get('avg_success_rate', 'N/A')}")

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    for r in results:
        if r.get("avg_success_rate") is not None:
            print(f"{r['name']:<25} Success Rate: {r['avg_success_rate']*100:.1f}% ± {r.get('std_success_rate', 0)*100:.1f}%")
        else:
            print(f"{r['name']:<25} {r.get('note', 'No results')}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "libero_available": LIBERO_AVAILABLE
    }
    output_path = Path("libero_simulation_results.json")
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
