"""
TinyMind deep config — one place for all hyperparameters.

Why a config file?
  In a deep project, you do NOT hard-code numbers inside train.py.
  You change them here, rerun, and you can say in an interview:
  "I tried decay 0, 0.02 and 0.05, and here is what changed."
  That is an ablation study, and it is what makes a project deep.

Deep defaults (for Colab GPU):
  layers 6, dim 256, heads 8, context 128, vocab from BPE (8000)
  This is still tiny compared to real LLMs, but much deeper than
  the 2-layer / 128-dim skeleton. Training it takes hours, not minutes.

Skeleton defaults (for a first CPU test):
  layers 2, dim 128, heads 4, context 64, word vocab ~140
  Use --config skeleton to test the pipeline in minutes.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # Model
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 8
    block_size: int = 128
    vocab_size: int = 8000  # overridden by tokenizer vocab size at runtime
    dropout: float = 0.0
    # Forgetting
    decay: float = 0.02
    echoic: int = 5
    # Training
    batch_size: int = 64
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    max_steps: int = 5000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    # Life cycle
    childhood_epochs: int = 3
    adolescence_epochs: int = 3
    adult_epochs: int = 2


SKELETON = Config(
    n_layer=2, n_embd=128, n_head=4, block_size=64,
    vocab_size=5000, max_steps=600,
)

DEEP = Config()  # defaults above


def get_config(name="deep"):
    return SKELETON if name == "skeleton" else DEEP
