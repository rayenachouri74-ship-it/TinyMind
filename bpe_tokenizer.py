"""
BPE tokenizer — Byte Pair Encoding, the real tokenizer used by LLMs.

Word tokenizer (tokenizer.py) gives one number per word. That breaks
on new words: they become <unk>. BPE fixes that by splitting words
into frequent subwords: "unhappiness" -> "un" + "happi" + "ness".
Rare words can still be built from pieces, so <unk> almost disappears.

How BPE training works, in 3 steps:
  1. Start with characters, e.g. "cat" -> c a t, with a word-end marker.
  2. Count which adjacent pair appears most often, merge it into one token.
  3. Repeat for N merges (here 8000). The merges are the vocabulary.

This is a teaching implementation, not production SentencePiece.
It is slow on huge data, fast enough on TinyMind data, and you can
explain every line. For real TinyStories, use 8000 merges and a
larger corpus; the code does not change.
"""
import json
import re
from collections import Counter
from pathlib import Path


class BPETokenizer:
    def __init__(self):
        self.merges = []  # list of (a, b) in merge order
        self.vocab = {}  # token -> id
        self.id_to_token = {}

    @staticmethod
    def _split_word(word):
        # Characters + end marker, so "cat" and "cats" do not merge wrongly.
        return list(word) + ["</w>"]

    def train(self, texts, num_merges=8000):
        # Start from characters.
        words = Counter()
        for t in texts:
            for w in re.findall(r"[a-z]+", t.lower()):
                words[" ".join(self._split_word(w))] += 1
        # Initial vocab: all characters + end marker.
        chars = set()
        for w in words:
            chars.update(w.split())
        self.vocab = {c: i for i, c in enumerate(sorted(chars))}
        # Learn merges.
        for _ in range(num_merges):
            pairs = Counter()
            for word, freq in words.items():
                symbols = word.split()
                for a, b in zip(symbols, symbols[1:]):
                    pairs[(a, b)] += freq
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            self.merges.append(best)
            # Apply this merge to all words.
            new_words = {}
            bigram = " ".join(best)
            replacement = "".join(best)
            for word, freq in words.items():
                new_word = word.replace(bigram, replacement)
                new_words[new_word] = freq
            words = new_words
            if replacement not in self.vocab:
                self.vocab[replacement] = len(self.vocab)
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        return self

    def _encode_word(self, word):
        symbols = self._split_word(word)
        # Apply merges in learned order, like BPE does.
        for a, b in self.merges:
            merged = "".join((a, b))
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text):
        ids = []
        for w in re.findall(r"[a-z]+", text.lower()):
            for tok in self._encode_word(w):
                # Unknown subword fallback: use character ids if present.
                ids.append(self.vocab.get(tok, 0))
        return ids

    def decode(self, ids):
        text = "".join(self.id_to_token.get(i, "") for i in ids)
        return text.replace("</w>", " ").strip()

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"merges": self.merges, "vocab": self.vocab}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        tok = cls()
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.vocab = data["vocab"]
        tok.id_to_token = {i: t for t, i in tok.vocab.items()}
        return tok

    @property
    def vocab_size(self):
        return len(self.vocab)


if __name__ == "__main__":
    from data import load_all_texts
    texts = load_all_texts()
    tok = BPETokenizer().train(texts, num_merges=2000)
    tok.save("bpe_vocab.json")
    print(f"BPE vocab size: {tok.vocab_size}, merges: {len(tok.merges)}")
    test = "sam has a cat"
    print(tok.encode(test), "->", tok.decode(tok.encode(test)))
