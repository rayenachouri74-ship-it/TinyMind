"""
Generate text from a checkpoint.

Example:
  python generate.py --checkpoint checkpoints/child.pt --prompt "Sam has a"
  python generate.py --checkpoint checkpoints/adult_science.pt --prompt "The moon is"
"""
import argparse
import os
from pathlib import Path
import torch
from tokenizer_loader import load_tokenizer
from model import TinyMind
from config import get_config


def main():
    os.chdir(Path(__file__).resolve().parent)
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="Sam has a")
    p.add_argument("--max_new_tokens", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.8)
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

    ids = torch.tensor([tok.encode(args.prompt, add_bos_eos=False)], device=device)
    out = model.generate(ids, args.max_new_tokens, temperature=args.temperature)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
