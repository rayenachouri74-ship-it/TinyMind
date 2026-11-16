"""
TinyMind tokenizer — words to numbers, numbers to words.

Why word-level, not BPE?
  For a first LM project, word-level is explainable in 30 seconds:
  every word gets one number. BPE is better, but harder to explain
  and debug. You can say that in an interview.

Special tokens:
  <pad> = filler, <unk> = unknown word, <bos> = start, <eos> = end.
"""
import json
import re
from pathlib import Path


def split_text(text):
    # Lowercase, keep words and simple punctuation as separate tokens.
    # Example: "Hello, Sam!" -> ["hello", ",", "sam", "!"]
    text = text.lower()
    return re.findall(r"[a-z]+|[.,!?;]", text)


class WordTokenizer:
    def __init__(self, vocab=None):
        # vocab: list of words, index = number. Index 0..3 are special.
        self.special = ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.vocab = list(self.special)
        self.word_to_id = {w: i for i, w in enumerate(self.vocab)}
        if vocab:
            for w in vocab:
                self.add_word(w)

    def add_word(self, word):
        if word not in self.word_to_id:
            self.word_to_id[word] = len(self.vocab)
            self.vocab.append(word)

    def build_from_texts(self, texts, max_vocab=5000):
        # Count words, keep the most common ones. Rare words become <unk>.
        from collections import Counter
        counts = Counter()
        for t in texts:
            counts.update(split_text(t))
        for word, _ in counts.most_common(max_vocab):
            self.add_word(word)
        return self

    def encode(self, text, add_bos_eos=True):
        ids = []
        if add_bos_eos:
            ids.append(self.word_to_id["<bos>"])
        for w in split_text(text):
            ids.append(self.word_to_id.get(w, self.word_to_id["<unk>"]))
        if add_bos_eos:
            ids.append(self.word_to_id["<eos>"])
        return ids

    def decode(self, ids):
        words = []
        for i in ids:
            w = self.vocab[i] if 0 <= i < len(self.vocab) else "<unk>"
            if w in ("<pad>", "<bos>", "<eos>"):
                continue
            words.append(w)
        # Put punctuation back without a space before it.
        out = ""
        for w in words:
            if w in ".,!?;":
                out += w
            else:
                out += (" " if out else "") + w
        return out

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.vocab, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            vocab = json.load(f)
        tok = cls()
        tok.vocab = vocab
        tok.word_to_id = {w: i for i, w in enumerate(vocab)}
        return tok

    @property
    def vocab_size(self):
        return len(self.vocab)


if __name__ == "__main__":
    # Build vocab from all data files, so every stage shares one tokenizer.
    # This is important: childhood, adolescence and adulthood must speak
    # the same number-language, or checkpoints cannot be reused.
    from data import load_all_texts
    texts = load_all_texts()
    tok = WordTokenizer().build_from_texts(texts, max_vocab=5000)
    tok.save(str(Path(__file__).resolve().parent / "vocab.json"))
    print(f"vocab size: {tok.vocab_size}")
    # Use a sentence actually in the data, so it round-trips cleanly.
    # Words outside the data become <unk> — that is expected, and a
    # limitation to mention in an interview, not a bug to hide.
    test = "Sam has a cat."
    print(f"example: {tok.encode(test)}")
    print(f"decoded: {tok.decode(tok.encode(test))}")
    print(f"outside-vocab example: {tok.decode(tok.encode('The cat sat on the mat.'))}  <- <unk> is expected here")
