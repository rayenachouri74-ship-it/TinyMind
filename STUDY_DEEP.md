# TinyMind Deep Study — 14-day plan to actually own the deep version

This is the deep version, not the skeleton. Studying it in a day is
not owning it. Use this plan, out loud, in simple English, no notes
for the final drill.

## Days 1-2: Tokenizers, both kinds
- Word tokenizer: why <unk> happens, why vocab 140 is too small for real language.
- BPE: characters -> merges -> subwords. Train it, decode a new word
  like "unhappiness" and show it does not become <unk>.
- Interview question: "Why did you switch from word to BPE for the deep version?"

## Days 3-4: Model and forgetting, in depth
- Draw attention, Q/K/V, causal mask, softmax, on paper.
- Explain decay formula, echoic buffer, and why decay without echoic
  fails (per fleeting-memory paper).
- Explain residual connections, LayerNorm, MLP, in one sentence each.

## Days 5-6: Training, deep config
- Read config.py: layers, dim, heads, context, LR schedule, warmup, grad clip.
- Explain cosine LR decay and why fine-tuning LR is smaller.
- Run a skeleton training run, watch loss, save a checkpoint, load it in generate.py.

## Days 7-8: Data, real vs. synthetic
- Explain TinyStories vs. BabyLM vs. synthetic templates, with limits of each.
- Explain why you must NOT claim TinyStories results if you used synthetic data.
- Clean one real dataset yourself in Colab, even a small subset, and note what broke.

## Days 9-10: Experiments and benchmarks
- Run experiments.py matrix, at least E1 vs. E2, with real numbers.
- Run benchmarks.py: grammar minimal pairs, held-out loss, memory gap test.
- Write results/results.md with your real table, plus 3 good and 3 bad generations.

## Days 11-12: Life cycle and specialization
- Explain childhood -> adolescence as continued training, not 3 separate models.
- Explain field choice by lowest trial loss, and how to switch fields.
- Test catastrophic forgetting: childhood loss before vs. after specialization.

## Days 13-14: Interview drill, full
- 2-minute pitch, structure only, not memorized words.
- 20 questions: add to the 10 in STUDY_GUIDE.md:
  11. BPE vs. word tokenizer, tradeoffs?
  12. What is an ablation and why one variable at a time?
  13. What is perplexity and why is it not enough?
  14. What is a minimal pair and why 50% is chance?
  15. What is warmup and cosine decay?
  16. What happens if decay is too high?
  17. What is a checkpoint and why per life stage?
  18. How would you scale this to 10M words? What breaks first?
  19. What did BabyLM find about curriculum learning?
  20. What are your model's 3 biggest limits, honestly?

If you cannot answer 17/20 out loud, you are not ready to post this
as a project you fully understand. Post the code, keep studying,
and update the README results only when you have run them.
