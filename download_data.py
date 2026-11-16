"""
Download real data for the deep version, with a safe fallback.

Deep version data plan, in order of preference:
  1. TinyStories (GPT-4 generated simple stories, public) — ideal for
     childhood / adolescence stages. Download via HuggingFace datasets
     if you have internet and the datasets library in Colab.
  2. BabyLM 10M strict-small — ideal for the child-sized claim, but
     larger to download and clean. Use it if you have time for v2.
  3. Synthetic data from data.py — what ships in this repo, so the
     project always runs offline. Use it for testing, not for final
     results claims about real language.

This script tries (1), and if it fails for any reason, it prints
exactly why and falls back to (3). It never silently pretends
synthetic data is real TinyStories data.
"""
import subprocess
import sys


def main():
    print("Trying TinyStories download via HuggingFace datasets...")
    print("If this fails in your environment, that is normal offline.")
    print("Fallback: python data.py  (synthetic, for testing only)")
    try:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train[:10000]")
        print(f"Downloaded TinyStories subset: {len(ds)} stories")
        with open("data/tinystories.txt", "w") as f:
            for ex in ds:
                text = ex.get("text", "").replace("\n", " ").strip()
                if text:
                    f.write(text + "\n")
        print("Saved: data/tinystories.txt")
        print("Next: adapt tokenizer.py / train.py to use data/tinystories.txt")
        print("as childhood data, and keep adolescence as longer stories.")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Fallback: run 'python data.py' to build synthetic data.")
        print("Do NOT claim TinyStories results if you used synthetic data.")


if __name__ == "__main__":
    main()
