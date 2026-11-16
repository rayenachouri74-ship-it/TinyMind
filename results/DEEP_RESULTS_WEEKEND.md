# TinyMind deep — verified training results (draft, 2026-10-23)

Status: TRAINING + EVALUATION COMPLETE (evaluation finished
2026-10-23, after a torch reinstall following a VM replacement).
Do NOT post this to GitHub as a finished project, and do NOT claim
interview mastery, until Elia can explain the numbers below out loud.

## Verified deep training losses (from the run logs, not estimates)

Config: 6 layers, 256 dim, 8 heads, context 128, BPE vocab 1,027,
5,000 real TinyStories (810,809 words), CPU, PyTorch 2.14.1+cpu.
Checkpoints are ~22 MB each in `checkpoints/`.

- Deep childhood (TinyStories, forgetting decay 0.02):
  epoch 1: 4.9378 -> epoch 2: 3.8729 — `deep_child.pt`
- Deep adolescence (continued from childhood, synthetic templated texts):
  epoch 1: 0.2451 -> epoch 2: 0.0664 — `deep_teen.pt`
  WARNING: this drop is much steeper than childhood because adolescence
  is still templated synthetic text, not real complex language.
  Do not read it as general ability.
- Deep base ablation (TinyStories, decay 0, no forgetting):
  epoch 1: 4.7343 -> epoch 2: 3.7973 — `deep_child_base.pt`
  Ablation, honestly: forgetting (3.8729) vs base (3.7973) differ by
  only 0.0756 at the end. Again, no clear forgetting benefit at this
  scale, on this data, single seed. Say that plainly in an interview.
- Deep adult specializations (from deep_teen, 1 epoch x 100 steps each):
  - code: 0.0836 — `deep_adult_code.pt` (saved 17:27 UTC 2026-10-23)
  - science: 0.4816 — `deep_adult_science.pt` (saved 17:35 UTC 2026-10-23)
  - stories: 0.3912 — `deep_adult_stories.pt` (saved 17:43 UTC 2026-10-23)
  Training is therefore COMPLETE: all three adult branches are saved.
  WARNING, carried over from the skeleton run: the code set was only
  4 repeated snippets, so a low code loss is repetition, not proven
  aptitude, unless the deep code set is shown to be larger.

## Held-out evaluation (evaluate.py, 50 batches each, from results/deep_evaluation.log)

- deep_child.pt on TinyStories: 3.6833 (training final 3.8729)
- deep_child_base.pt on TinyStories, decay 0: 3.6518
  Held-out ablation gap: 3.6833 vs 3.6518 = 0.0315 — again no clear
  forgetting benefit, same conclusion as training loss.
- deep_teen.pt on adolescence: 0.0637 (templated data, not general ability)
- deep_teen.pt on TinyStories: 6.3255 — continued adolescence training
  largely overwrote childhood, exactly as in the skeleton run
  (skeleton teen-on-childhood 1.0557 vs child 0.1275). Report this as
  a limit, not a success: no rehearsal was used.
- deep_adult_code.pt on field_code: 0.0169 — extremely low, but the
  code set is 4 snippets repeated, so this is repetition, not aptitude.
- deep_adult_science.pt on field_science: 0.0625
- deep_adult_stories.pt on field_stories: 0.0561
  (from results/deep_evaluation2.log; the first eval run was terminated
  mid-eval and a second run failed instantly because a VM replacement
  had wiped the system torch install — checkpoints in the home folder
  survived, torch did not, exactly as at 17:58 UTC. Torch 2.14.1+cpu
  was reinstalled from cache in ~68s and the rerun completed.)
- deep_adult_science.pt on TinyStories: 7.0287
- deep_adult_code.pt on TinyStories: 6.8930
- deep_adult_stories.pt on TinyStories: 6.3726
  All three adult models are worse on childhood TinyStories than the
  child model itself (3.6833) and than the teen (6.3255 for science,
  though stories at 6.3726 is close to teen). Specialization, like
  adolescence, overwrote childhood — no rehearsal was used.

## Generations (generate.py, temperature 0.8, 30 tokens, verbatim)

- deep_child, "Sam has a":
  "sam has a cootetee and fish and a night the dan leant seek it was
  a bird and jourwket" — garbled. The deep child model, trained on
  real TinyStories, does NOT generate clean templated sentences the
  way the skeleton teen did. Say so plainly.
- deep_teen, "When Sam went to the":
  "when sam went to the park the bird followed because it wanted to
  play although it was late and the sky was" — templated, coherent,
  cut off at the token limit.
- deep_adult_science, "The moon is":
  "the moon is es around the earth once every month it does need play
  alled thinschool" — the science fact is partly there, with garbling.
- deep_adult_code, "def add":
  "def add a b return a b" — close in shape, not valid code: no
  parentheses, colon, or newline structure.
- deep_adult_stories, "One quiet night,":
  "one quiet night leo heard a strange sound in the forest it was not
  a bird and it was not" — templated, coherent, cut off at the limit.

## What this proves, and what it does not

Proves: the full life-cycle pipeline trains end to end at deep scale
on real TinyStories, branches into three fields, and the forgetting
ablation is measurable.
Does NOT prove: general language ability, that forgetting helps,
interview mastery, or a GitHub-ready finished project. Evaluation is
now complete (above), and its main finding is a limit: continued
training overwrote childhood (teen 6.3255, adults 6.37–7.03 on
TinyStories vs child 3.6833), and deep child generation is garbled.

## Weekend study drill (30 minutes, no new training needed)

Answer out loud, in simple English, no notes for the last pass:
1. Why is deep childhood loss (~3.9) so much higher than skeleton
   childhood loss (~0.13)? (Real TinyStories vs templated synthetic
   data; BPE vocab 1,027 vs word vocab 140.)
2. Why did adolescence loss collapse to 0.0664? (Templated data
   again — same limitation as the skeleton, just on a deeper model.)
3. What does the base-vs-forgetting gap of 0.0756 actually show?
   (No clear benefit at this scale; one seed; do not overclaim.)
4. Why is code loss (0.0836) far below science (0.4816)?
   (Check the code set size/repetition before claiming aptitude.)
5. What is the one evaluation result you most need before posting?
   (Teen/adult on childhood data — now known for deep: teen on
   TinyStories is 6.3255 vs child 3.6833, i.e. continued training
   overwrote childhood, just as the skeleton showed at 1.0557 vs
   0.1275. Report it as a limit, not a success.)

6. Why is deep child generation garbled while teen/stories
   generations are templated and clean? (Real TinyStories vs templated
   synthetic data — the deep child never learned clean templates.)
