import argparse, json, os
from pathlib import Path
import numpy as np
from stable_baselines3 import DQN
from envs.graph_flow_env import FlowEnv
from envs.generators import FlowGenConfig
import yaml

def make_env(env_cfg):
    cfg = FlowGenConfig(**env_cfg)
    return FlowEnv(cfg, max_steps=env_cfg.get("max_steps", 60))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/dqn")
    ap.add_argument("--total_timesteps", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env_config", type=str, default="configs/env_easy.yaml")
    args = ap.parse_args()

    env_cfg = yaml.safe_load(open(args.env_config))
    env = make_env(env_cfg)
    model = DQN("MlpPolicy", env, verbose=0, seed=args.seed, learning_rate=3e-4, gamma=0.99)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.learn(total_timesteps=args.total_timesteps)
    model.save(os.path.join(args.out, f"dqn_seed{args.seed}.zip"))
    with open(os.path.join(args.out, f"meta_seed{args.seed}.json"), "w") as f:
        json.dump({"env_cfg": env_cfg, "seed": args.seed}, f)

if __name__ == "__main__":
    main()
