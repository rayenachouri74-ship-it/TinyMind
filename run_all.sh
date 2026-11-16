#!/bin/bash
# TinyMind full life-cycle, in order. Run in Colab or on a GPU machine.
# Start with epochs=1 to test, then use the defaults in each script.
set -e
pip install -r requirements.txt
python data.py
python tokenizer.py
python train.py --stage childhood --epochs 3 --decay 0.02 --out checkpoints/child.pt
python train.py --stage adolescence --init checkpoints/child.pt --epochs 3 --decay 0.02 --out checkpoints/teen.pt
python train.py --stage childhood --epochs 3 --decay 0.0 --out checkpoints/child_base.pt
python specialize.py --choose --base checkpoints/teen.pt
python specialize.py --field science --base checkpoints/teen.pt --epochs 2 --out checkpoints/adult_science.pt
python specialize.py --field stories --base checkpoints/teen.pt --epochs 2 --out checkpoints/adult_stories.pt
python specialize.py --field code --base checkpoints/teen.pt --epochs 2 --out checkpoints/adult_code.pt
python evaluate.py --checkpoint checkpoints/adult_science.pt --stage adolescence
python demo_lifecycle.py
