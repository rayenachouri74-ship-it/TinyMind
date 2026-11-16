"""
Adulthood: specialization and field choice.

Two modes:
  1. --choose : test the teen model on a small trial set from each field,
     pick the field with the LOWEST loss. That is the model's aptitude:
     it learns that field fastest, like a human choosing a career
     they are good at. No magic, just loss.
  2. --field X : fine-tune the teen model on field X for a few steps,
     save checkpoints/adult_X.pt. The base is kept, so you can branch
     the same teen into science, stories, and code, and compare them.
     Switching fields = loading a different adult checkpoint.

Catastrophic forgetting test (do this in evaluate.py after specializing):
  does the adult science model still remember childhood stories?
  If its childhood loss went up a lot, it forgot — say so, do not hide it.
"""
import argparse
import os
from pathlib import Path
import torch
from data import load_stage
from tokenizer_loader import load_tokenizer
from model import TinyMind
from config import get_config
from train import get_batch


def trial_loss(model, texts, tok, device, batches=20, block_size=64):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(batches):
            xb, yb = get_batch(texts, tok, block_size, 16, device)
            _, loss = model(xb, yb)
            total += loss.item()
    return total / batches


def main():
    os.chdir(Path(__file__).resolve().parent)
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="checkpoints/teen.pt")
    p.add_argument("--choose", action="store_true")
    p.add_argument("--field", default="", choices=["", "science", "stories", "code"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--steps_per_epoch", type=int, default=100)
    p.add_argument("--decay", type=float, default=0.02)
    p.add_argument("--out", default="")
    p.add_argument("--config", default="skeleton", choices=["skeleton", "deep"],
                   help="Must match the config used to train --base.")
    p.add_argument("--tokenizer", default="word", choices=["word", "bpe"])
    p.add_argument("--vocab", default="vocab.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(args.tokenizer, args.vocab)
    cfg = get_config(args.config)
    model = TinyMind(tok.vocab_size, n_embd=cfg.n_embd, n_head=cfg.n_head,
                     n_layer=cfg.n_layer, block_size=cfg.block_size,
                     decay=args.decay).to(device)
    model.load_state_dict(torch.load(args.base, map_location=device))
    print(f"base: {args.base}")

    fields = ["science", "stories", "code"]
    if args.choose:
        print("Field choice trial (lower loss = better aptitude):")
        scores = {}
        for field in fields:
            texts = load_stage(f"field_{field}")
            loss = trial_loss(model, texts, tok, device, block_size=cfg.block_size)
            scores[field] = loss
            print(f"  {field}: {loss:.4f}")
        chosen = min(scores, key=scores.get)
        print(f"Chosen field: {chosen}")
        print(f"To specialize into it, run:")
        print(f"  python specialize.py --field {chosen} --base {args.base} --out checkpoints/adult_{chosen}.pt")
        return

    if not args.field:
        print("Use --choose, or --field science|stories|code")
        return

    texts = load_stage(f"field_{args.field}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)  # smaller LR for fine-tuning
    model.train()
    step = 0
    for epoch in range(args.epochs):
        total = 0.0
        for _ in range(args.steps_per_epoch):
            xb, yb = get_batch(texts, tok, cfg.block_size, 32, device)
            _, loss = model(xb, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            step += 1
        print(f"adult {args.field} epoch {epoch+1} avg loss: {total/args.steps_per_epoch:.4f}")
    out = args.out or f"checkpoints/adult_{args.field}.pt"
    torch.save(model.state_dict(), out)
    print(f"saved adult checkpoint: {out}")


if __name__ == "__main__":
    main()
