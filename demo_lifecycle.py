"""
Life-cycle demo: same prompt, three ages, side by side.

Needs checkpoints/child.pt, checkpoints/teen.pt, and one adult checkpoint.
If a checkpoint is missing, it prints what to run to make it, instead
of crashing or faking output. Never fake a generation sample.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

PROMPT = "Sam has a"


def run(checkpoint, config):
    if not Path(checkpoint).exists():
        return f"[missing {checkpoint} — train that stage first]"
    out = subprocess.run(
        [sys.executable, "generate.py", "--checkpoint", checkpoint,
         "--config", config,
         "--prompt", PROMPT, "--max_new_tokens", "25", "--temperature", "0.8"],
        capture_output=True, text=True)
    return out.stdout.strip() or "[no output]"


def main():
    os.chdir(Path(__file__).resolve().parent)
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="skeleton", choices=["skeleton", "deep"],
                   help="Must match the config used to train the checkpoints.")
    args = p.parse_args()
    print(f"Prompt: '{PROMPT}'")
    print()
    print("AGE 5 — childhood:")
    print(" ", run("checkpoints/child.pt", args.config))
    print()
    print("AGE 15 — adolescence:")
    print(" ", run("checkpoints/teen.pt", args.config))
    print()
    for field in ["science", "stories", "code"]:
        ckpt = f"checkpoints/adult_{field}.pt"
        if Path(ckpt).exists():
            print(f"ADULT — {field} specialist:")
            print(" ", run(ckpt, args.config))
            print()


if __name__ == "__main__":
    main()
