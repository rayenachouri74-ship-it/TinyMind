#!/bin/bash
set -x
echo "=== DEEP CHILDHOOD ===" 
python3 train.py --stage tinystories --config deep --tokenizer bpe --vocab bpe_vocab.json --epochs 2 --steps_per_epoch 150 --batch_size 16 --decay 0.02 --out checkpoints/deep_child.pt
echo "=== DEEP ADOLESCENCE ==="
python3 train.py --stage adolescence --config deep --tokenizer bpe --vocab bpe_vocab.json --init checkpoints/deep_child.pt --epochs 2 --steps_per_epoch 150 --batch_size 16 --decay 0.02 --out checkpoints/deep_teen.pt
echo "=== DEEP BASE ==="
python3 train.py --stage tinystories --config deep --tokenizer bpe --vocab bpe_vocab.json --epochs 2 --steps_per_epoch 150 --batch_size 16 --decay 0.0 --out checkpoints/deep_child_base.pt
echo "=== DEEP SPECIALIZATION ==="
python3 specialize.py --field code --base checkpoints/deep_teen.pt --config deep --tokenizer bpe --vocab bpe_vocab.json --epochs 1 --steps_per_epoch 100 --out checkpoints/deep_adult_code.pt
python3 specialize.py --field science --base checkpoints/deep_teen.pt --config deep --tokenizer bpe --vocab bpe_vocab.json --epochs 1 --steps_per_epoch 100 --out checkpoints/deep_adult_science.pt
python3 specialize.py --field stories --base checkpoints/deep_teen.pt --config deep --tokenizer bpe --vocab bpe_vocab.json --epochs 1 --steps_per_epoch 100 --out checkpoints/deep_adult_stories.pt
echo "=== DEEP TRAINING DONE ==="
ls -lh checkpoints/deep_*.pt
