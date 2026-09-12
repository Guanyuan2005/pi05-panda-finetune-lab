#!/usr/bin/env python3
"""Register Panda transforms and training configs in OpenPI."""

import argparse
from pathlib import Path


PANDA_FACTORY = '''

@dataclasses.dataclass(frozen=True)
class LeRobotPandaDataConfig(DataConfigFactory):
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.image",
                        "observation/wrist_image": "observation.images.wrist_image",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[panda_policy.PandaInputs(model_type=model_config.model_type)],
            outputs=[panda_policy.PandaOutputs()],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=self.action_sequence_keys,
        )
'''


def panda_train_config(name: str, steps: int) -> str:
    return f'''    TrainConfig(
        name="{name}",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotPandaDataConfig(
            repo_id="local/panda_multi_object_box_v2_train_compat",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size=1,
        num_train_steps={steps},
        log_interval=10,
        save_interval={steps},
        keep_period={steps},
        num_workers=2,
        wandb_enabled=False,
        assets_base_dir="/root/shared-nvme/openpi-assets",
        checkpoint_base_dir="/root/shared-nvme/checkpoints",
    ),
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path)
    args = parser.parse_args()
    path = args.config_path
    text = path.read_text()

    import_line = "import openpi.policies.panda_policy as panda_policy\n"
    if import_line not in text:
        anchor = "import openpi.policies.libero_policy as libero_policy\n"
        text = text.replace(anchor, anchor + import_line)

    if "class LeRobotPandaDataConfig" not in text:
        anchor = "\n\n@dataclasses.dataclass(frozen=True)\nclass RLDSDroidDataConfig"
        text = text.replace(anchor, PANDA_FACTORY + anchor)

    if 'name="pi05_panda_multi_object_lora_smoke"' not in text:
        configs = (
            panda_train_config("pi05_panda_multi_object_lora_smoke", 100)
            + panda_train_config("pi05_panda_multi_object_lora", 20_000)
        )
        text = text.replace("_CONFIGS = [\n", "_CONFIGS = [\n" + configs)

    path.write_text(text)
    print(f"registered Panda configs in {path}")


if __name__ == "__main__":
    main()
