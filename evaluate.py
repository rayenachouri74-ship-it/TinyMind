"""
Evaluate a checkpoint: loss + a simple long-range memory test.

Memory test idea, in plain words:
  Take a story where a name appears at the start, then many words later
  the story refers back to that name. A model with good long memory
  should predict the name better than a model that forgot it.
  We measure loss on the final name word only, for short vs. long gaps.
  If forgetting hurts the long gap a lot, report that — it is a finding,
  not a failure to hide.
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


def main():
    os.chdir(Path(__file__).resolve().parent)
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stage", default="adolescence")
    p.add_argument("--decay", type=float, default=0.02)
    p.add_argument("--config", default="skeleton", choices=["skeleton", "deep"],
                   help="Must match the config used to train --checkpoint.")
    p.add_argument("--tokenizer", default="word", choices=["word", "bpe"])
    p.add_argument("--vocab", default="vocab.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(args.tokenizer, args.vocab)
    cfg = get_config(args.config)
    model = TinyMind(tok.vocab_size, n_embd=cfg.n_embd, n_head=cfg.n_head,
                     n_layer=cfg.n_layer, block_size=cfg.block_size,
                     decay=args.decay).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    texts = load_stage(args.stage)
    total = 0.0
    with torch.no_grad():
        for _ in range(50):
            xb, yb = get_batch(texts, tok, cfg.block_size, 16, device)
            _, loss = model(xb, yb)
            total += loss.item()
    print(f"checkpoint: {args.checkpoint}")
    print(f"stage: {args.stage}  avg loss (50 batches): {total/50:.4f}")
    print()
    print("Memory test (manual, read this in an interview):")
    print("  Short gap: 'Sam has a cat. Sam likes the cat.'")
    print("  Long gap:  'Sam has a cat. ' + 40 filler words + ' Sam likes the cat.'")
    print("  Compare the loss on the final 'cat' word for each gap.")
    print("  A forgetting model should do worse on the long gap — that is expected.")


if __name__ == "__main__":
    main()
