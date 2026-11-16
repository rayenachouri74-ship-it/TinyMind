# TinyMind Study Guide — how to actually own this project

You asked for the full project to study it. Studying it is now the work.
Do not submit it, put it on GitHub as yours, or talk about it in an
interview until you can pass the checks below out loud, without notes.

## 7-day study plan (about 2 hours a day, adjust to your classes)

**Day 1 — Tokenizer**
Run `python data.py` and `python tokenizer.py`. Then explain, out loud:
- What is a token? Why do we turn words into numbers?
- What are <pad>, <unk>, <bos>, <eos> for?
- Why must all life stages share one tokenizer?

**Day 2 — Data**
Read `data.py`. Explain:
- What is childhood data vs. adolescence data, in one sentence each?
- Why is synthetic data acceptable for a demo, and what is its limit?
- What would you change if you used real TinyStories data?

**Day 3 — Model, part 1: attention**
Read `model.py`, class `ForgettingAttention`. Explain:
- What are Q, K, V in plain words?
- What does the causal mask do, and why?
- What is decay, what is echoic, and why did the fleeting-memory paper need both?

**Day 4 — Model, part 2: transformer block**
Explain:
- What does a block do? Attention + MLP + residual, in your own words.
- What is loss (cross-entropy)? What does lower loss mean, and what does it NOT mean?
- Why is the model tiny (2 layers, 128 dim) on purpose?

**Day 5 — Training**
Run training with `--epochs 1` first, then a real run in Colab. Explain:
- What happens in one training step? Forward, loss, backward, optimizer step.
- Why do we save a checkpoint per life stage?
- Why is the fine-tuning learning rate smaller in `specialize.py`?

**Day 6 — Specialization and field choice**
Explain:
- How does `--choose` pick a field? (Lowest trial loss = best aptitude, no magic.)
- How do you switch fields? (Load a different adult checkpoint.)
- What is catastrophic forgetting, and how would you test for it?

**Day 7 — Full interview drill**
Do the 2-minute pitch below, then answer the 10 questions below, out loud,
in simple English, no notes, no phone. Record yourself once. If you
stumble, go back to that day, do not just re-read.

## 2-minute pitch (memorize the structure, not the words)

"I built TinyMind, a tiny language model trained from scratch that grows
up. In childhood it learns short simple stories, in adolescence longer
complex language, and as an adult it specializes into a field.
I added human-like forgetting to attention, with a small echoic buffer,
based on fleeting-memory research, and I trained it on child-sized data
inspired by BabyLM. I tested base vs. forgetting, and I tested whether
specializing makes it forget childhood. Here are my results [fill in
your real numbers], and here are 3 generations that failed — I show
those too."

## 10 interview questions you must be able to answer

1. Why a tiny model, not a big one?
2. What is a tokenizer and why word-level here?
3. Explain forgetting attention with one example.
4. What is echoic memory and why 3–7 words?
5. What is BabyLM and why did you cite it?
6. Did curriculum learning help in BabyLM? (Answer: not as expected — you tested it.)
7. What is cross-entropy loss?
8. How does field choice work? How is it not magic?
9. What is catastrophic forgetting and did you see it?
10. What are the limits of your model? (Small data, synthetic data, word tokenizer, no real-world performance.)

## What NOT to say

- Do NOT say you invented BabyLM, forgetting attention, or curriculum learning.
- Do NOT say the model writes like ChatGPT. It does not, and saying so loses trust.
- Do NOT say you trained on 10 million words if your run used 1 million. Report your real run.
- Do NOT hide bad generations. Show 3 good and 3 bad.

## If you run out of time before 16 Nov (CaCTüS deadline)

Submit the scoped version: childhood + adolescence + one forgetting ablation
+ README with honest results. That is a finished, explainable project.
The full life-cycle with 3 adult fields is your v2 for EPFL (29 Nov) and
ETH (16 Dec) and for master's applications, not a reason to miss CaCTüS.
