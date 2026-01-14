#!/bin/bash
# ACTOR-Enhanced DreamVLA Training on LIBERO-Spatial
# This adds L3 Action Consistency Loss to DreamVLA

export WANDB_MODE=offline

### NEED TO CHANGE ###
save_checkpoint_path="checkpoints/finetune_DreamVLA_ACTOR_libero/"
root_dir="."
vit_checkpoint_path="checkpoints/vit_mae/mae_pretrain_vit_base.pth"
finetune_from_pretrained_ckpt="checkpoints/dreamvla/libero_pretrain.pth"
libero_path="libero_spatial_converted"
### NEED TO CHANGE ###

node=1
node_num=1  # Single GPU for initial testing

python train_actor.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 4 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --workers 4 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs 40 \
    --seed 42 \
    --batch_size 4 \
    --precision fp32 \
    --learning_rate 2e-4 \
    --save_checkpoint \
    --finetune_type libero_finetune \
    --root_dir ${root_dir} \
    --wandb_project dreamvla_actor \
    --weight_decay 1e-4 \
    --num_resampler_query 16 \
    --run_name libero_finetune_spatial_actor \
    --save_checkpoint_path ${save_checkpoint_path} \
    --transformer_layers 24 \
    --hidden_dim 1024 \
    --transformer_heads 16 \
    --phase "finetune" \
    --obs_pred \
    --action_pred_steps 3 \
    --sequence_length 7 \
    --future_steps 3 \
    --window_size 10 \
    --loss_image \
    --loss_action \
    --reset_action_token \
    --reset_obs_token \
    --save_checkpoint_seq 1 \
    --start_save_checkpoint 25 \
    --gripper_width \
    --warmup_epochs 5 \
    --libero_path ${libero_path} \
    --finetune_from_pretrained_ckpt ${finetune_from_pretrained_ckpt} \
    --report_to_wandb \
    --use_dit_head \
    --l3_weight 0.5 \
    --attn_implementation "sdpa"
