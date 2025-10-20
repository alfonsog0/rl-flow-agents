import argparse, json, math
import matplotlib.pyplot as plt

def load_and_agg(paths_with_labels):
    agg = []
    for path, label in paths_with_labels:
        rows = json.load(open(path))
        srs = [r["success_rate"] for r in rows]
        n = len(srs)
        mean = sum(srs)/n
        var = sum((x-mean)**2 for x in srs)/(n-1) if n>1 else 0.0
        sd = var**0.5
        t = 2.776 if 2<=n<=30 else 1.96
        ci = t*sd/math.sqrt(n) if n>1 else 0.0
        agg.append((label, mean, ci))
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/plots")
    ap.add_argument("--base_ppo", default="results/ppo_eval.json")
    ap.add_argument("--base_dqn", default="results/dqn_eval.json")
    ap.add_argument("--base_lstm", default="results/ppo_lstm_eval.json")
    ap.add_argument("--shift_pop_ppo", default=None)
    ap.add_argument("--shift_pop_lstm", default=None)
    ap.add_argument("--shift_deeper_ppo", default=None)
    args = ap.parse_args()

    items = []
    items.append((args.base_dqn, "DQN (base)"))
    items.append((args.base_ppo, "PPO (base)"))
    items.append((args.base_lstm, "PPO-LSTM (base)"))
    if args.shift_pop_ppo:    items.append((args.shift_pop_ppo, "PPO (popups↑)"))
    if args.shift_pop_lstm:   items.append((args.shift_pop_lstm, "PPO-LSTM (popups↑)"))
    if args.shift_deeper_ppo: items.append((args.shift_deeper_ppo, "PPO (deeper)"))

    data = load_and_agg(items)
    labels = [l for l,_,_ in data]
    means  = [m for _,m,_ in data]
    cis    = [c for _,_,c in data]

    plt.figure()
    plt.bar(labels, means, yerr=cis, capsize=5)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Success rate (mean ± 95% CI)")
    plt.tight_layout()
    import os; os.makedirs(args.out, exist_ok=True)
    plt.savefig(f"{args.out}/success_rate_agg.png")

if __name__ == "__main__":
    main()
