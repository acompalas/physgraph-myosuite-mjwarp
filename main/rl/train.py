"""Training entry point, ported from PhysGraph's main/rl/train.py
(launch_rlg_hydra). Structure kept as close to the original as possible
-- same Hydra setup, same rl_games Runner/model/network registration,
same config dump / experiment-dir logic. The ONLY genuinely IsaacGym-
coupled pieces in the original were `import isaacgym`, `import
physgraph_envs`, and the body of create_isaacgym_env() -- all replaced
below with our own mujoco-warp env factory (physgraph_envs.lib.make,
now backed by MyoHandPourEnv instead of an IsaacGym task).

Uses PhysGraph's own (non-BiH) model classes -- ModelA2CContinuousLogStd
/SepModelA2CContinuousLogStd -- rather than the bimanual transformer
network builder, since our task is single-hand with a simple flat
observation vector, not the multi-modal dict observation the BiH
network was built for."""
import warnings
warnings.filterwarnings("ignore")

import hydra
from omegaconf import DictConfig, OmegaConf

import lib


def preprocess_train_config(cfg, config_dict):
    """Adding common configuration parameters to the rl_games train config.
    Unchanged from PhysGraph's own version -- no IsaacGym dependency here."""
    train_cfg = config_dict["params"]["config"]
    train_cfg["device"] = cfg.rl_device
    train_cfg["population_based_training"] = False
    train_cfg["pbt_idx"] = None
    train_cfg["full_experiment_name"] = cfg.get("full_experiment_name")

    print(f"Using rl_device: {cfg.rl_device}")
    print(f"Using sim_device: {cfg.sim_device}")

    try:
        model_size_multiplier = config_dict["params"]["network"]["mlp"]["model_size_multiplier"]
        if model_size_multiplier != 1:
            units = config_dict["params"]["network"]["mlp"]["units"]
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(f"Modified MLP units by x{model_size_multiplier} to "
                  f"{config_dict['params']['network']['mlp']['units']}")
    except KeyError:
        pass

    return config_dict


@hydra.main(version_base="1.1", config_name="config", config_path="../cfg")
def launch_rlg_hydra(cfg: DictConfig):
    import os
    from datetime import datetime

    from hydra.utils import to_absolute_path

    if cfg.display:
        import cv2
        import numpy as np

        cv2.imshow("dummy", np.zeros((1, 1, 3), dtype=np.uint8))
        cv2.waitKey(1)

    import physgraph_envs
    from lib.utils.reformat import omegaconf_to_dict, print_dict
    from lib.utils.utils import set_np_formatting, set_seed
    from lib.utils.rlgames_utils import (
        RLGPUAlgoObserver,
        MultiObserver,
        ComplexObsRLGPUEnv,
    )
    from lib.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from lib.rl.runner import Runner
    from lib.rl.models import ModelA2CContinuousLogStd, SepModelA2CContinuousLogStd
    from rl_games.algos_torch.model_builder import register_network, register_model

    register_model("my_continuous_a2c_logstd", ModelA2CContinuousLogStd)
    register_model("sep_my_continuous_a2c_logstd", SepModelA2CContinuousLogStd)

    if cfg.checkpoint:
        if type(cfg.checkpoint) == str:
            cfg.checkpoint = to_absolute_path(cfg.checkpoint)
        elif type(cfg.checkpoint) == list:
            cfg.checkpoint = [to_absolute_path(cp) for cp in cfg.checkpoint]

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    set_np_formatting()

    global_rank = int(os.getenv("RANK", "0"))
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    def create_mjwarp_env():
        """Replaces PhysGraph's create_isaacgym_env(). Our env factory
        (physgraph_envs.lib.make) ignores the IsaacGym-specific kwargs
        that don't apply to mujoco-warp."""
        kwargs = dict(
            sim_device=cfg.sim_device,
            rl_device=cfg.rl_device,
            cfg=cfg.task,
            headless=cfg.headless,
        )
        envs = physgraph_envs.lib.make(**kwargs)
        return envs

    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "RLGPU",
            "env_creator": create_mjwarp_env,
        },
    )

    obs_spec = {}
    if "central_value_config" in cfg.rl_train.params.config:
        critic_net_cfg = cfg.rl_train.params.config.central_value_config.network
        obs_spec["states"] = {
            "names": list(critic_net_cfg.inputs.keys()),
            "concat": not critic_net_cfg.name == "complex_net",
            "space_name": "state_space",
        }

    vecenv.register("RLGPU", lambda config_name, num_actors: ComplexObsRLGPUEnv(config_name))

    rlg_config_dict = omegaconf_to_dict(cfg.rl_train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    observers = [RLGPUAlgoObserver()]

    if cfg.wandb_activate:
        cfg.seed += global_rank
        if global_rank == 0:
            wandb_observer = WandbAlgoObserver(cfg)
            observers.append(wandb_observer)

    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        return runner

    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    if cfg.test:
        prefix = "dump_" if cfg.save_rollouts else "test_"
        experiment_dir = os.path.join(
            "dumps",
            prefix + cfg.rl_train.params.config.name + "__{date:%m-%d-%H-%M-%S}".format(date=datetime.now()),
        )
    else:
        experiment_dir = os.path.join(
            "runs",
            cfg.rl_train.params.config.name + "__" + "{date:%m-%d-%H-%M-%S}".format(date=datetime.now()),
        )
        cfg.rl_train.params.config.full_experiment_name = experiment_dir.replace("runs/", "")
        runner.params["config"]["full_experiment_name"] = experiment_dir.replace("runs/", "")
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    runner.run(
        {
            "train": not cfg.test,
            "play": cfg.test,
            "checkpoint": cfg.checkpoint,
            "from_ckpt_epoch": cfg.from_ckpt_epoch,
            "sigma": cfg.sigma if cfg.sigma != "" else None,
            "save_rollouts": {
                "save_rollouts": cfg.save_rollouts,
                "rollout_saving_fpath": os.path.join(experiment_dir, "rollouts.hdf5"),
                "save_successful_rollouts_only": cfg.save_successful_rollouts_only,
                "num_rollouts_to_save": cfg.num_rollouts_to_save,
                "num_rollouts_to_run": cfg.num_rollouts_to_run,
                "min_episode_length": cfg.min_episode_length,
            },
        }
    )


if __name__ == "__main__":
    launch_rlg_hydra()
