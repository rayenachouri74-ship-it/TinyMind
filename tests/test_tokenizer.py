import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tokenizer import WordTokenizer


def test_roundtrip():
    tok = WordTokenizer().build_from_texts(["Sam has a cat.", "Lily likes the park."])
    assert tok.decode(tok.encode("Sam has a cat.")) == "sam has a cat."


def test_unknown_becomes_unk():
    tok = WordTokenizer().build_from_texts(["Sam has a cat."])
    ids = tok.encode("Sam has a zzzword.")
    assert tok.word_to_id["<unk>"] in ids


def test_vocab_save_load(tmp_path):
    tok = WordTokenizer().build_from_texts(["Sam has a cat."])
    p = tmp_path / "vocab.json"
    tok.save(str(p))
    tok2 = WordTokenizer.load(str(p))
    assert tok2.vocab == tok.vocab
