"""
TinyMind experiments — the ablation study that makes this deep.

An ablation means: change ONE thing, keep everything else the same,
measure what happens. If you change 5 things at once, you learn nothing.

Experiment matrix for the full project:
  E1  decay=0.00, echoic=5, curriculum=off   (base transformer)
  E2  decay=0.02, echoic=5, curriculum=off   (forgetting only)
  E3  decay=0.05, echoic=5, curriculum=off   (strong forgetting)
  E4  decay=0.02, echoic=0, curriculum=off   (forgetting without echoic — should be worse, per prior work)
  E5  decay=0.02, echoic=5, curriculum=on    (life-cycle: childhood -> adolescence)
  E6  E5 + adult specialization, one field (full life-cycle)

Each experiment: 2 seeds if you have time, 1 seed if you are in a hurry
for CaCTuS, and say so honestly. Report mean loss, grammar accuracy,
and 3 good + 3 bad generations per experiment.

This script does NOT train for you in this skeleton version — it
prints the exact commands to run, in order, so you can copy them into
Colab and keep a log. The deep training script is train.py with
--decay / --echoic / --stage flags. Use this file as your lab notebook
template: fill in results/results.md as you go, do not back-fill numbers
you did not run.
"""
import itertools

EXPERIMENTS = [
    ("E1_base", {"decay": 0.0, "echoic": 5, "curriculum": False}),
    ("E2_forgetting", {"decay": 0.02, "echoic": 5, "curriculum": False}),
    ("E3_strong_forgetting", {"decay": 0.05, "echoic": 5, "curriculum": False}),
    ("E4_no_echoic", {"decay": 0.02, "echoic": 0, "curriculum": False}),
    ("E5_lifecycle", {"decay": 0.02, "echoic": 5, "curriculum": True}),
    ("E6_full_adult", {"decay": 0.02, "echoic": 5, "curriculum": True}),
]


def main():
    print("TinyMind experiment matrix — run in order, log every result:")
    print()
    for name, cfg in EXPERIMENTS:
        print(f"{name}: decay={cfg['decay']} echoic={cfg['echoic']} curriculum={cfg['curriculum']}")
        print(f"  python train.py --stage childhood --epochs 3 --decay {cfg['decay']} --echoic {cfg['echoic']} --out checkpoints/{name}_child.pt")
        if cfg["curriculum"]:
            print(f"  python train.py --stage adolescence --init checkpoints/{name}_child.pt --epochs 3 --decay {cfg['decay']} --out checkpoints/{name}_teen.pt")
            print(f"  python benchmarks.py --checkpoint checkpoints/{name}_teen.pt --decay {cfg['decay']}")
        print()
    print("Results template — copy to results/results.md and fill in ONLY after running:")
    print("| Exp | Decay | Echoic | Curriculum | Loss | Grammar % | Notes |")
    print("|-----|-------|--------|------------|------|-----------|-------|")


if __name__ == "__main__":
    main()
