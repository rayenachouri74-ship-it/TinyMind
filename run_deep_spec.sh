#!/bin/bash
set -x
python3 specialize.py --field code --base checkpoints/deep_teen.pt --config deep --tokenizer bpe --vocab bpe_vocab.json --epochs 1 --steps_per_epoch 100 --out checkpoints/deep_adult_code.pt
python3 specialize.py --field science --base checkpoints/deep_teen.pt --config deep --tokenizer bpe --vocab bpe_vocab.json --epochs 1 --steps_per_epoch 100 --out checkpoints/deep_adult_science.pt
python3 specialize.py --field stories --base checkpoints/deep_teen.pt --config deep --tokenizer bpe --vocab bpe_vocab.json --epochs 1 --steps_per_epoch 100 --out checkpoints/deep_adult_stories.pt
echo "=== DEEP SPECIALIZATION DONE ==="
ls -lh checkpoints/deep_adult_*.pt
