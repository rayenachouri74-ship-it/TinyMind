"""
TinyMind benchmarks — how you prove a model works, not just that it runs.

1. Perplexity / loss on a held-out set: lower is better, but it is not
   the whole story. A model can have low loss and still write nonsense.
2. Grammar minimal pairs (BLiMP-style, tiny version):
   Correct:   "The cat is on the mat."
   Incorrect: "The cat are on the mat."
   The model should assign lower loss to the correct sentence.
   Accuracy = % of pairs where correct wins. Chance = 50%.
3. Long-range memory:
   Short gap vs. long gap, same name at the start and end.
   Does forgetting help or hurt the long gap? Report both.
4. Field perplexity:
   Loss of the teen model vs. the adult specialist on science/stories/code.
   Specialization should lower loss on its own field, and may raise
   loss on childhood data — that is catastrophic forgetting, measure it.

Run: python benchmarks.py --checkpoint checkpoints/teen.pt
Fill the table in README.md with your real numbers, never invented ones.
"""
import argparse
import os
from pathlib import Path
import torch
from data import load_stage
from tokenizer import WordTokenizer
from model import TinyMind
from config import get_config
from train import get_batch


def sentence_loss(model, tok, sentence, device):
    ids = tok.encode(sentence)
    if len(ids) < 3:
        return float("inf")
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([ids[1:]], device=device)
    with torch.no_grad():
        _, loss = model(x, y)
    return loss.item()


def main():
    os.chdir(Path(__file__).resolve().parent)
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--decay", type=float, default=0.02)
    p.add_argument("--config", default="skeleton", choices=["skeleton", "deep"],
                   help="Must match the config used to train --checkpoint.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = WordTokenizer.load("vocab.json")
    cfg = get_config(args.config)
    model = TinyMind(tok.vocab_size, n_embd=cfg.n_embd, n_head=cfg.n_head,
                     n_layer=cfg.n_layer, block_size=cfg.block_size,
                     decay=args.decay).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    pairs = [
        ("The cat is on the mat.", "The cat are on the mat."),
        ("Sam has a dog.", "Sam have a dog."),
        ("Lily likes the park.", "Lily like the park."),
        ("The bird flies high.", "The bird fly high."),
    ]
    correct = 0
    print("Grammar minimal pairs (correct should have lower loss):")
    for good, bad in pairs:
        lg = sentence_loss(model, tok, good, device)
        lb = sentence_loss(model, tok, bad, device)
        win = lg < lb
        correct += int(win)
        print(f"  {'WIN' if win else 'LOSS'}  good={lg:.3f} bad={lb:.3f}  | {good}")
    print(f"Grammar accuracy: {correct}/{len(pairs)} = {correct/len(pairs)*100:.0f}% (chance 50%)")
    print()
    # Held-out loss
    texts = load_stage("adolescence")
    total = 0.0
    with torch.no_grad():
        for _ in range(30):
            xb, yb = get_batch(texts, tok, 64, 16, device)
            _, loss = model(xb, yb)
            total += loss.item()
    print(f"Held-out adolescence loss: {total/30:.4f}")


if __name__ == "__main__":
    main()
