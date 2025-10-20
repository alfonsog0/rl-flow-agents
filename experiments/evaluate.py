import argparse, glob, json, os
import numpy as np
from stable_baselines3 import PPO, DQN
from sb3_contrib import RecurrentPPO
from envs.graph_flow_env import FlowEnv
from envs.generators import FlowGenConfig
import yaml

def load_model(path):
    if "ppo_lstm" in path:
        return RecurrentPPO.load(path)
    elif "ppo" in os.path.basename(path):
        return PPO.load(path)
    else:
        return DQN.load(path)

def evaluate(model, env_cfg, episodes=50, seed=0):
    gen_keys = set(FlowGenConfig.__dataclass_fields__.keys())
    gen_kwargs = {k: v for k, v in env_cfg.items() if k in gen_keys}
    env = FlowEnv(FlowGenConfig(**gen_kwargs), max_steps=env_cfg.get("max_steps", 60), seed=seed)
    succ, steps = 0, []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed+ep)
        done = False
        st = 0
        lstm_state = None
        while not done:
            if hasattr(model, "predict") and "Recurrent" in model.__class__.__name__:
                action, lstm_state = model.predict(obs, state=lstm_state, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))
            done = term or trunc
            st += 1
        if info.get("node_id") == env.flow.goal:
            succ += 1
        steps.append(st)
    return {"success_rate": succ/episodes, "avg_steps": float(np.mean(steps))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_glob", required=True)
    ap.add_argument("--env_config", default="configs/env_easy.yaml")
    args = ap.parse_args()

    env_cfg = yaml.safe_load(open(args.env_config))
    out = []
    for path in glob.glob(args.models_glob):
        model = load_model(path)
        stats = evaluate(model, env_cfg, episodes=50, seed=0)
        out.append({"model": path, **stats})
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
