import argparse, json, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", help="one or more JSON files with [{model, success_rate, avg_steps}]")
    ap.add_argument("--out", default="results/plots")
    args = ap.parse_args()

    import os; os.makedirs(args.out, exist_ok=True)
    rows = []
    for j in args.jsonl:
        rows += json.load(open(j))

    labels = [r["model"].split("/")[-1] for r in rows]
    sr = [r["success_rate"] for r in rows]
    plt.figure()
    plt.bar(labels, sr)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Success rate")
    plt.tight_layout()
    plt.savefig(f"{args.out}/success_rate.png")

if __name__ == "__main__":
    main()
