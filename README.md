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
- **Observation dimension**: Fixed observation size. We set obs_dim=32 and pad one-hot node observations to this length. This allows training a single policy and evaluating zero-shot on graphs with different sizes (n_nodes ≤ 32) without changing the network architecture. This keeps evaluations under distribution shift (deeper graphs, more popups) apples-to-apples.
- **Memory & generalisation**: Expectation. PPO-LSTM (recurrent) > PPO (feedforward) on popup-heavy settings (partial observability), PPO ≈/≥ DQN on base; both degrade under heavier popups, but LSTM degrades less. Zero-shot on deeper graphs shows a drop but remains above random, demonstrating transfer.

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


DQN   (base)        mean=0.780  sd=0.076  95%CI±0.095  n=5
PPO   (base)        mean=0.832  sd=0.048  95%CI±0.060  n=5
PPO-LSTM (base)     mean=0.808  sd=0.023  95%CI±0.028  n=5
PPO   (popups↑)     mean=0.812  sd=0.054  95%CI±0.067  n=5
PPO-LSTM (popups↑)  mean=0.752  sd=0.066  95%CI±0.082  n=5
DQN   (popups↑)     mean=0.824  sd=0.062  95%CI±0.077  n=5
PPO   (deeper)      mean=0.728  sd=0.073  95%CI±0.091  n=5