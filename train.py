"""
Train one life stage of TinyMind.

  childhood:    start from scratch, data/childhood.txt
  adolescence:  start from childhood checkpoint, data/adolescence.txt
  adult:        use specialize.py instead, it calls the same idea with a field

Loss = cross-entropy: how surprised is the model by the true next word?
Lower loss = less surprised = better. That is the only training signal.

Checkpoints are saved per stage, so demo_lifecycle.py can load
child.pt, teen.pt and adult_*.pt and show the same prompt at each age.
"""
import argparse
import os
import csv
import random
import torch
from pathlib import Path

from data import load_stage
from tokenizer_loader import load_tokenizer
from model import TinyMind
from config import get_config


def get_batch(texts, tok, block_size, batch_size, device):
    # Pick random texts, encode them, cut a block, and make targets
    # = the same block shifted by one word. That is next-word prediction.
    xs, ys = [], []
    for _ in range(batch_size):
        text = random.choice(texts)
        ids = tok.encode(text)
        if len(ids) < block_size + 1:
            ids = ids + [tok.word_to_id["<pad>"]] * (block_size + 1 - len(ids))
        start = random.randint(0, max(0, len(ids) - block_size - 1))
        chunk = ids[start:start + block_size + 1]
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    return (torch.tensor(xs, device=device),
            torch.tensor(ys, device=device))


def main():
    os.chdir(Path(__file__).resolve().parent)
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="childhood",
                   choices=["childhood", "adolescence", "tinystories"])
    p.add_argument("--init", default="", help="checkpoint to start from, or empty for scratch")
    p.add_argument("--decay", type=float, default=0.02)
    p.add_argument("--echoic", type=int, default=5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--steps_per_epoch", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", default="checkpoints/child.pt")
    p.add_argument("--tokenizer", default="word", choices=["word", "bpe"])
    p.add_argument("--vocab", default="vocab.json")
    p.add_argument("--config", default="skeleton", choices=["skeleton", "deep"],
                   help="skeleton = current 2-layer/128-dim behaviour (default); "
                        "deep = config.py 6-layer/256-dim model. Use the same flag "
                        "when loading the checkpoint, or shapes will not match.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, stage: {args.stage}, decay: {args.decay}")

    tok = load_tokenizer(args.tokenizer, args.vocab)
    texts = load_stage(args.stage)
    print(f"training texts: {len(texts)}")

    cfg = get_config(args.config)
    print(f"config: {args.config} (layers={cfg.n_layer}, dim={cfg.n_embd}, "
          f"heads={cfg.n_head}, context={cfg.block_size})")
    model = TinyMind(tok.vocab_size, n_embd=cfg.n_embd, n_head=cfg.n_head,
                     n_layer=cfg.n_layer, block_size=cfg.block_size,
                     decay=args.decay, echoic=args.echoic).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
        print(f"loaded init checkpoint: {args.init}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    log_path = f"results/loss_{args.stage}.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        step = 0
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for _ in range(args.steps_per_epoch):
                xb, yb = get_batch(texts, tok, cfg.block_size, args.batch_size, device)
                _, loss = model(xb, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
                step += 1
                if step % 50 == 0:
                    writer.writerow([step, round(loss.item(), 4)])
            print(f"epoch {epoch+1}/{args.epochs}  avg loss: {total/args.steps_per_epoch:.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
