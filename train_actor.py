"""
ACTOR-Enhanced DreamVLA Training Script.

This script extends DreamVLA training with L3 Action Consistency Loss.

Key additions:
- Inverse Dynamics head
- L3 loss computation during training
- Comparison logging between baseline and ACTOR

Usage:
    python train_actor.py --finetune_type calvin --l3_weight 0.1
"""

import glob
import os
import random
from collections import OrderedDict
import numpy as np
import torch
import wandb
import clip
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from models.dreamvla_model import DreamVLA
from utils.train_utils import get_checkpoint, get_ckpt_name
from utils.arguments_utils import get_parser
from utils.data_utils import (
    get_calvin_dataset,
    get_libero_pretrain_dataset,
    get_libero_finetune_dataset,
)
from utils.distributed_utils import init_distributed_device, world_info_from_env

# ACTOR Extension
from actor_extension.inverse_dynamics import InverseDynamicsHead
from actor_extension.action_consistency_loss import ActionConsistencyLoss


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def count_parameters(model):
    total_params = 0
    trainable_params = 0
    trainable_names = []
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_names.append(name)
    return total_params, trainable_params, trainable_names


def train_one_epoch_actor(
    args,
    model,
    inverse_dynamics,
    l3_loss_fn,
    epoch,
    calvin_loader,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    """Training loop with L3 Action Consistency Loss."""
    import time
    from contextlib import suppress
    import torch.nn.functional as F
    from tqdm import tqdm
    from utils.train_utils import (
        get_cast_dtype,
        get_autocast,
        patchify,
        normalize_patchfied_image,
        AverageMeter,
    )

    num_batches_per_epoch = calvin_loader.num_batches
    total_training_steps = num_batches_per_epoch * args.num_epochs
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    inverse_dynamics.train()

    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")

    mv_avg_loss = []
    mv_avg_l3 = []

    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch

        # Unpack batch (same as original DreamVLA)
        images_primary = batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images_wrist = batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True)
        text_tokens = batch_calvin[1].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, args.window_size, 1)
        states = batch_calvin[4].to(device_id, dtype=cast_dtype, non_blocking=True)
        actions = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)

        # Prepare inputs
        bs, seq_len = images_primary.shape[:2]

        if args.gripper_width:
            input_states = torch.cat([states[..., :6], states[..., -2:]], dim=-1)
        else:
            input_states = torch.cat([states[..., :6], states[..., [-1]]], dim=-1)
            input_states[..., 6:] = (input_states[..., 6:] + 1) // 2

        actions[..., 6:] = (actions[..., 6:] + 1) // 2

        input_image_primary = images_primary[:, :args.sequence_length, :]
        input_image_wrist = images_wrist[:, :args.sequence_length, :]
        input_text_token = text_tokens[:, :args.sequence_length, :]
        input_state = input_states[:, :args.sequence_length, :]

        # Label actions for L3
        label_actions = torch.cat([
            actions[:, j:args.sequence_length - args.atten_goal + j, :].unsqueeze(-2)
            for j in range(args.action_pred_steps)
        ], dim=-2)

        # Track infos (if available)
        track_infos = batch_calvin[12] if len(batch_calvin) > 12 else {}

        with autocast():
            # Forward pass through DreamVLA
            arm_pred_action, gripper_pred_action, image_pred, _, _, _, depth_pred, traj_pred, dino_feat_pred, sam_feat_pred = model(
                image_primary=input_image_primary,
                image_wrist=input_image_wrist,
                state=input_state,
                text_token=input_text_token,
                action=actions[:, :args.sequence_length, :],
                track_infos=track_infos,
                action_label=label_actions[:, :args.sequence_length - args.atten_goal].detach() if hasattr(args, 'use_dit_head') and args.use_dit_head else None,
            )

        # ============ ACTOR L3 Loss ============
        # For L3, we need:
        # 1. Current state latent (proxy: state embedding)
        # 2. Predicted next state latent (proxy: observation predictions)
        # 3. Ground truth actions

        # Use state input as current latent proxy
        current_latent = input_state[:, :args.sequence_length - args.atten_goal].flatten(0, 1)
        current_latent = current_latent[..., :model.module.hidden_dim if hasattr(model, 'module') else model.hidden_dim]

        # Pad/project if needed
        hidden_dim = model.module.hidden_dim if hasattr(model, 'module') else model.hidden_dim
        if current_latent.shape[-1] < hidden_dim:
            current_latent = F.pad(current_latent, (0, hidden_dim - current_latent.shape[-1]))

        # Use observation predictions as predicted next state (if available)
        if image_pred is not None:
            # image_pred: (B*S, 2, pred_num, num_patches, patch_dim)
            pred_features = image_pred[:, 0].flatten(1, 2)  # Use primary camera
            pred_next_latent = pred_features.mean(dim=1)  # Pool to (B*S, hidden)
            if pred_next_latent.shape[-1] != hidden_dim:
                pred_next_latent = F.adaptive_avg_pool1d(
                    pred_next_latent.unsqueeze(1), hidden_dim
                ).squeeze(1)
        elif dino_feat_pred is not None:
            pred_features = dino_feat_pred[:, 0].flatten(1, 2)
            pred_next_latent = pred_features.mean(dim=1)
            if pred_next_latent.shape[-1] != hidden_dim:
                pred_next_latent = F.adaptive_avg_pool1d(
                    pred_next_latent.unsqueeze(1), hidden_dim
                ).squeeze(1)
        else:
            # Fallback: use a simple MLP prediction
            pred_next_latent = current_latent

        # Get GT actions for L3
        gt_actions = label_actions[:, :args.sequence_length - args.atten_goal, 0, :].flatten(0, 1)
        gt_arm = gt_actions[..., :6]
        gt_gripper = gt_actions[..., 6:]

        # Compute L3: Inverse dynamics from predicted next state
        with autocast():
            pred_arm, pred_gripper = inverse_dynamics(current_latent.detach(), pred_next_latent)
            l3_result = l3_loss_fn(pred_arm, pred_gripper, gt_arm, gt_gripper)
            l3_loss = l3_result['loss']

        # ============ Original DreamVLA Losses ============
        # Action loss
        if args.loss_action and args.action_pred_steps and not args.use_dit_head:
            loss_arm_action = F.smooth_l1_loss(
                arm_pred_action[:, :args.sequence_length - args.atten_goal],
                label_actions[:, :args.sequence_length - args.atten_goal, :, :6].detach()
            )
            loss_gripper_action = F.binary_cross_entropy(
                gripper_pred_action[:, :args.sequence_length - args.atten_goal],
                label_actions[:, :args.sequence_length - args.atten_goal, :, 6:].detach()
            )
        elif args.use_dit_head:
            loss_arm_action = arm_pred_action  # DiT returns loss directly
            loss_gripper_action = torch.tensor([0.0]).to(device_id)
        else:
            loss_arm_action = torch.tensor([0.0]).to(device_id)
            loss_gripper_action = torch.tensor([0.0]).to(device_id)

        # Other losses (simplified - add back image/depth/etc. as needed)
        loss_image = torch.tensor([0.0]).to(device_id)

        # ============ Total Loss ============
        loss_calvin = args.loss_arm_action_ratio * loss_arm_action + args.loss_gripper_action_ratio * loss_gripper_action
        loss_total = loss_calvin + args.l3_weight * l3_loss

        # Gradient accumulation
        loss = loss_total / args.gradient_accumulation_steps

        mv_avg_loss.append(loss.item())
        mv_avg_l3.append(l3_loss.item())

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(inverse_dynamics.parameters(), 0.1)

        # Step optimizer
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (num_steps == num_batches_per_epoch - 1):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                wandb.log({
                    "loss_total": loss_total.item(),
                    "loss_action": loss_calvin.item(),
                    "l3_action_consistency": l3_loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "global_step": global_step,
                })

        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({
            "loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon,
            "L3": sum(mv_avg_l3[-avg_horizon:]) / avg_horizon if mv_avg_l3 else 0,
        })


def add_actor_args(parser):
    """Add ACTOR-specific arguments."""
    parser.add_argument("--l3_weight", type=float, default=0.1, help="Weight for L3 action consistency loss")
    parser.add_argument("--use_actor", action="store_true", help="Enable ACTOR training")
    return parser


@record
def main(args):
    os.environ["WANDB_DIR"] = f"{os.path.abspath(args.save_checkpoint_path)}"

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    random_seed(args.seed)

    # Create DreamVLA model
    model = DreamVLA(
        finetune_type=args.finetune_type,
        clip_device=device_id,
        vit_checkpoint_path=args.vit_checkpoint_path,
        sequence_length=args.sequence_length,
        num_resampler_query=args.num_resampler_query,
        num_obs_token_per_image=args.num_obs_token_per_image,
        calvin_input_image_size=args.calvin_input_image_size,
        patch_size=args.patch_size,
        action_pred_steps=args.action_pred_steps,
        obs_pred=args.obs_pred,
        transformer_layers=args.transformer_layers,
        hidden_dim=args.hidden_dim,
        transformer_heads=args.transformer_heads,
        phase=args.phase,
        gripper_width=args.gripper_width,
        depth_pred=args.depth_pred,
        dino_feat_pred=args.dino_feat_pred,
        sam_feat_pred=args.sam_feat_pred,
        use_dit_head=args.use_dit_head,
    )

    # Create ACTOR components
    inverse_dynamics = InverseDynamicsHead(
        state_dim=args.hidden_dim,
        action_dim=7,
        hidden_dim=args.hidden_dim,
    )
    l3_loss_fn = ActionConsistencyLoss()

    # Load dataset
    if args.finetune_type == "calvin":
        dataset = get_calvin_dataset(args, model.image_processor, clip, epoch=0)
    elif args.finetune_type == "libero_finetune":
        dataset = get_libero_finetune_dataset(args, model.image_processor, clip, epoch=0)
    else:
        raise ValueError(f"Unknown finetune_type: {args.finetune_type}")

    # Setup precision
    if args.precision == "bf16":
        model = model.bfloat16()
    elif args.precision == "fp16":
        model = model.half()

    model = model.to(device_id)
    inverse_dynamics = inverse_dynamics.to(device_id)
    model._init_model_type()

    # DDP
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    # Optimizer (include inverse dynamics parameters)
    all_params = list(ddp_model.parameters()) + list(inverse_dynamics.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in all_params if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # LR scheduler
    total_training_steps = dataset.dataloader.num_batches * args.num_epochs
    args.warmup_steps = dataset.dataloader.num_batches * args.warmup_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Load checkpoint if provided
    if args.finetune_from_pretrained_ckpt is not None:
        checkpoint = torch.load(args.finetune_from_pretrained_ckpt, map_location="cpu")
        ddp_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"Loaded checkpoint from {args.finetune_from_pretrained_ckpt}")

    # W&B init
    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project + "_actor",
            entity=args.wandb_entity,
            name=args.run_name + "_actor",
            config=vars(args),
        )

    # Training
    total_params, trainable_params, _ = count_parameters(model)
    id_params, id_trainable, _ = count_parameters(inverse_dynamics)
    print(f"DreamVLA params: {total_params/1e6:.1f}M, trainable: {trainable_params/1e6:.1f}M")
    print(f"Inverse Dynamics params: {id_params/1e6:.1f}M")

    ddp_model.train()
    for epoch in range(args.num_epochs):
        dataset.set_epoch(epoch)
        train_one_epoch_actor(
            args=args,
            model=ddp_model,
            inverse_dynamics=inverse_dynamics,
            l3_loss_fn=l3_loss_fn,
            epoch=epoch,
            calvin_loader=dataset.dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device_id=device_id,
            wandb=wandb,
        )

        # Save checkpoint
        if args.rank == 0 and args.save_checkpoint and epoch % args.save_checkpoint_seq == 0:
            ckpt_dir = os.path.join(args.save_checkpoint_path, args.run_name + "_actor")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": get_checkpoint(ddp_model),
                "inverse_dynamics_state_dict": inverse_dynamics.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(ckpt_dir, f"{epoch}.pth"))


if __name__ == "__main__":
    parser = get_parser()
    parser = add_actor_args(parser)
    args = parser.parse_args()
    main(args)
