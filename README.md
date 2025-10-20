# RL Flow Agents (MVP)

**Objective.** Train agents to explore and validate product-like flows (directed graphs) under sparse/noisy feedback. Show learning vs strong baselines and transfer to unseen graphs with limited tuning.

## Environment (why this design)
- **States**: screens as nodes in a random DAG.
- **Actions**: abstract menu mapped to outgoing edges.
- **Partial observability**: observation = one-hot current node; hidden dependency flag is invisible (POMDP).
- **Failure modes**:
  1) Dead-ends (absorbing failure)  
  2) Stochastic pop-ups masking actions  
  3) Hidden dependency (earlier action toggles later transition)
- **Rewards**: step=-0.01, success=+1.0, failure=-0.5.

## Algorithms
- **DQN** (value-based), **PPO** (policy-gradient), **RecurrentPPO** (LSTM for memory).

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# train 5 seeds (example)
for s in 0 1 2 3 4; do python agents/run_ppo.py --seed $s --env_config configs/env_easy.yaml --total_timesteps 150000; done
for s in 0 1 2 3 4; do python agents/run_recurrent_ppo.py --seed $s --env_config configs/env_easy.yaml --total_timesteps 150000; done
for s in 0 1 2 3 4; do python agents/run_dqn.py --seed $s --env_config configs/env_easy.yaml --total_timesteps 150000; done

# evaluate on base and distribution shifts
python experiments/evaluate.py --models_glob 'results/ppo/*.zip' --env_config configs/env_easy.yaml > results/ppo_eval.json
python experiments/evaluate.py --models_glob 'results/ppo_lstm/*.zip' --env_config configs/shift_popups.yaml > results/ppo_lstm_shift_popups.json
python experiments/evaluate.py --models_glob 'results/dqn/*.zip' --env_config configs/shift_deeper.yaml > results/dqn_shift_deeper.json

# plot (MVP)
python experiments/plot.py --jsonl results/ppo_eval.json results/ppo_lstm_shift_popups.json results/dqn_shift_deeper.json --out results/plots
