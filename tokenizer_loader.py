"""Load a word or BPE tokenizer with a common interface.

Skeleton uses word tokens (vocab.json). Deep uses BPE (bpe_vocab.json)
trained on real TinyStories. BPE has no <pad> token in its learned vocab,
so the wrapper maps <pad> to id 0 for batch padding. TinyStories are
long (>128 subword tokens), so padding is rare in deep training, and
this mapping is documented, not hidden.
"""
from tokenizer import WordTokenizer
from bpe_tokenizer import BPETokenizer


class BPEWrapper:
    def __init__(self, tok):
        self._tok = tok
        self.word_to_id = {"<pad>": 0}

    @property
    def vocab_size(self):
        return self._tok.vocab_size

    def encode(self, text, add_bos_eos=False):
        return self._tok.encode(text)

    def decode(self, ids):
        return self._tok.decode(ids)


def load_tokenizer(kind, path):
    if kind == "bpe":
        return BPEWrapper(BPETokenizer.load(path))
    return WordTokenizer.load(path)
