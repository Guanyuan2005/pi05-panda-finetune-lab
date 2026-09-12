from pathlib import Path
p=Path("/root/shared-nvme/workspace/openpi/src/openpi/training/config.py")
s=p.read_text()
name="pi05_panda_h50_overfit_1ep"
if f'name="{name}"' not in s:
    block="""    TrainConfig(
        name=\"pi05_panda_h50_overfit_1ep\",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            max_token_len=64,
            paligemma_variant=\"gemma_2b_lora\",
            action_expert_variant=\"gemma_300m_lora\",
        ),
        data=LeRobotPandaDataConfig(
            repo_id=\"local/panda_overfit_1ep\",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            \"gs://openpi-assets/checkpoints/pi05_base/params\"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            max_token_len=64,
            paligemma_variant=\"gemma_2b_lora\",
            action_expert_variant=\"gemma_300m_lora\",
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size=1,
        num_train_steps=5000,
        log_interval=100,
        save_interval=5000,
        keep_period=5000,
        num_workers=2,
        wandb_enabled=False,
        assets_base_dir=\"/root/shared-nvme/openpi-assets\",
        checkpoint_base_dir=\"/root/shared-nvme/checkpoints\",
    ),
"""
    s=s.replace("_CONFIGS = [\n", "_CONFIGS = [\n"+block, 1)
    p.write_text(s)
    print("added",name)
else: print("exists",name)
