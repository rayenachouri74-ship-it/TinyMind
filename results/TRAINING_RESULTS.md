# TinyMind — skeleton training results, 2026-10-23

Trained on this VM, CPU only, PyTorch 2.14.1+cpu, Python 3.12.3.
Config: skeleton — 2 layers, 128 dim, 4 heads, context 64, vocab 140 (word-level).
Data: SYNTHETIC only, from data.py — 2,000 childhood, 1,500 adolescence,
600 each science/stories/code. No TinyStories, no real data, no pretrained weights.
AdamW, cross-entropy loss. This is a pipeline proof, not general language ability.

## Training losses (avg per epoch, from train.py output)

| Stage | Decay | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---|---|---|---|
| Childhood (forgetting) | 0.02 | 0.8561 | 0.1740 | 0.1329 |
| Childhood (base, no forgetting) | 0.0 | 0.8441 | 0.1626 | 0.1320 |
| Adolescence, from child.pt | 0.02 | 0.8109 | 0.1476 | 0.1288 |

Ablation finding: base vs forgetting final loss differs by 0.0009
(0.1320 vs 0.1329). That is noise at this scale, single seed, templated data.
Do NOT claim forgetting helped. It did not show a benefit here.

## Held-out evaluation (evaluate.py, 50 batches)

- child.pt on childhood: 0.1275
- child_base.pt on childhood, decay 0: 0.1271
- teen.pt on adolescence: 0.1255
- teen.pt on childhood: 1.0557 — continued training largely overwrote
  childhood templates. This is catastrophic forgetting / distribution shift,
  a real limit to report, not hide. No rehearsal was used.
- adult_code.pt on field_code: 0.0792
- adult_science.pt on field_science: 0.3021
- adult_stories.pt on field_stories: 0.2746

## Specialization choice (specialize.py --choose, teen model, trial loss)

- science: 3.2771
- stories: 5.1194
- code: 1.1439 — chosen by lowest loss

Caveat: code trial set is only 4 snippets repeated, so low loss likely
reflects repetition, not true aptitude. Treat "chosen field" as a demo
mechanism, not a proven career aptitude.

Adult fine-tuning, 2 epochs each, from teen.pt:
- adult_code: 0.3512 -> 0.1200
- adult_science: 1.2659 -> 0.4835
- adult_stories: 1.4862 -> 0.4318

## Generations (generate.py, verbatim)

Good:
1. Teen, prompt "When Sam went to the":
   "when sam went to the garden, the fox followed because it wanted to play,
   although it was late and the sky was dark. nina found a kite near the river and"
2. Child, prompt "Sam has a":
   "sam has a a rabbit. the rabbit likes the school.. plays with a book in the school."
   Mostly correct, with errors: double "a a", double period, missing subject.
3. Teen, prompt "Sam has a cat", temperature 0.5:
   "sam has a cat to the forest, the cat followed because it wanted to play,
   although it was late and the sky"

Bad:
1. Adult science, prompt "The moon is":
   "the moon is be a, it. once night. it world not fish to make food. this process is gas."
2. Adult code, prompt "def add":
   "def add a, a, b" — no valid code. Word-level tokenizer is unsuitable for code.
3. Adult stories, prompt "Sam found a":
   "sam found a sound, zoe heard a not the forest. it was to heard has a dog and it was not the sound."

## Checkpoints in this run

child.pt, child_base.pt, teen.pt, adult_code.pt, adult_science.pt, adult_stories.pt
Each ~1.8 MB, skeleton config. Deep config (6 layers, 256 dim, BPE 8000, 10M words)
was NOT trained. Not posted to GitHub. Not mastered by Elia until he can explain
method, data, and these results in his own words.
