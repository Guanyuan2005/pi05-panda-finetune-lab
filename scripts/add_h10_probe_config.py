from pathlib import Path
p=Path("/root/shared-nvme/workspace/openpi/src/openpi/training/config.py")
s=p.read_text()
name="pi05_panda_h10_mem_probe"
if f'name="{name}"' not in s:
    block='    TrainConfig(\n        name="pi05_panda_h10_mem_probe",\n        model=pi0_config.Pi0Config(\n            pi05=True,\n            action_horizon=10,\n            max_token_len=64,\n            paligemma_variant="gemma_2b_lora",\n            action_expert_variant="gemma_300m_lora",\n        ),\n        data=LeRobotPandaDataConfig(\n            repo_id="local/panda_multi_object_box_v2_train_compat",\n            base_config=DataConfig(prompt_from_task=True),\n        ),\n        weight_loader=weight_loaders.CheckpointWeightLoader(\n            "gs://openpi-assets/checkpoints/pi05_base/params"\n        ),\n        freeze_filter=pi0_config.Pi0Config(\n            pi05=True,\n            action_horizon=10,\n            max_token_len=64,\n            paligemma_variant="gemma_2b_lora",\n            action_expert_variant="gemma_300m_lora",\n        ).get_freeze_filter(),\n        ema_decay=None,\n        batch_size=1,\n        num_train_steps=100,\n        log_interval=10,\n        save_interval=100,\n        keep_period=100,\n        num_workers=2,\n        wandb_enabled=False,\n        assets_base_dir="/root/shared-nvme/openpi-assets",\n        checkpoint_base_dir="/root/shared-nvme/checkpoints",\n    ),\n'
    s=s.replace("_CONFIGS = [\n", "_CONFIGS = [\n"+block, 1)
    p.write_text(s)
    print("added",name)
else:
    print("exists",name)
