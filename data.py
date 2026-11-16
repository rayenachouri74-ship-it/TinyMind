"""
TinyMind data — small, synthetic, fully explainable datasets.

We do NOT download TinyStories here, because a first project should run
offline and you should know exactly what is in the data.
The templates below are simple, but they teach the same lesson:
short simple language -> longer complex language -> specialised language.

Life stages:
  childhood:   8-15 words, simple present, one idea per sentence.
  adolescence: 25-50 words, because / when / although, two ideas linked.
  adult fields: science, stories, code — each a different future.

If you later want real TinyStories data, replace load_all_texts() with
a TinyStories loader. The rest of the project does not change, because
it only needs a list of strings.
"""
import random
from pathlib import Path

random.seed(42)
BASE = Path(__file__).resolve().parent

NAMES = ["sam", "lily", "max", "nina", "leo", "mia", "ben", "zoe"]
ANIMALS = ["cat", "dog", "bird", "fish", "rabbit", "fox"]
PLACES = ["park", "school", "garden", "river", "house", "forest"]
OBJECTS = ["ball", "book", "kite", "cake", "hat", "star"]


def childhood_stories(n=2000):
    out = []
    for _ in range(n):
        name = random.choice(NAMES)
        animal = random.choice(ANIMALS)
        place = random.choice(PLACES)
        obj = random.choice(OBJECTS)
        out.append(
            f"{name} has a {animal}. "
            f"The {animal} likes the {place}. "
            f"{name} plays with a {obj} in the {place}."
        )
    return out


def adolescence_stories(n=1500):
    out = []
    for _ in range(n):
        name = random.choice(NAMES)
        animal = random.choice(ANIMALS)
        place = random.choice(PLACES)
        obj = random.choice(OBJECTS)
        out.append(
            f"When {name} went to the {place}, the {animal} followed because "
            f"it wanted to play, although it was late and the sky was dark. "
            f"{name} found a {obj} near the river and decided to keep it, "
            f"because it reminded {name} of a happy day at school."
        )
    return out


def science_texts(n=600):
    facts = [
        "The moon goes around the earth once every month. It does not make its own light. We see it because the sun lights it.",
        "Water can be a liquid, a solid, or a gas. When water is cold it becomes ice. When water is hot it becomes steam.",
        "Plants need light, water, and air to grow. Leaves use light to make food. This process is called photosynthesis.",
        "The earth spins once every day. That spin makes day and night. The earth also goes around the sun once every year.",
    ]
    return [random.choice(facts) for _ in range(n)]


def story_texts(n=600):
    out = []
    for _ in range(n):
        name = random.choice(NAMES)
        out.append(
            f"One quiet night, {name} heard a strange sound in the forest. "
            f"It was not a bird and it was not the wind. It was a small door, "
            f"hidden in an old tree, and behind it was a world that {name} had never seen."
        )
    return out


def code_texts(n=600):
    snippets = [
        "def add(a, b):\n    return a + b",
        "def is_even(n):\n    return n % 2 == 0",
        "def greet(name):\n    return 'hello ' + name",
        "for i in range(5):\n    print(i)",
    ]
    # Code is kept as words for the word tokenizer: it will learn code
    # tokens like def, return, print as ordinary words. That is a limitation
    # to mention in an interview, not hide.
    return [random.choice(snippets).replace("\n", " ") for _ in range(n)]


def write_data():
    Path(BASE / "data").mkdir(exist_ok=True)
    for name, texts in [
        ("childhood.txt", childhood_stories()),
        ("adolescence.txt", adolescence_stories()),
        ("field_science.txt", science_texts()),
        ("field_stories.txt", story_texts()),
        ("field_code.txt", code_texts()),
    ]:
        with open(BASE / "data" / name, "w") as f:
            f.write("\n".join(texts))
        print(f"{name}: {len(texts)} texts")


def load_all_texts():
    # All stages share one vocabulary, see tokenizer.py.
    texts = []
    for name in ["childhood", "adolescence", "field_science", "field_stories", "field_code"]:
        p = BASE / f"data/{name}.txt"
        if p.exists():
            texts.extend(p.read_text().splitlines())
    if not texts:
        # Fallback: generate in memory if data/ was not written yet.
        texts = childhood_stories(500) + adolescence_stories(300)
    return texts


def load_stage(stage):
    if stage == "tinystories":
        p = BASE / "data/tinystories.txt"
    else:
        p = BASE / f"data/{stage}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Run python data.py first. Missing {p}")
    return p.read_text().splitlines()


if __name__ == "__main__":
    write_data()
