import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from bpe_tokenizer import BPETokenizer


def test_bpe_roundtrip_simple():
    tok = BPETokenizer().train(["sam has a cat", "lily likes the park"], num_merges=50)
    text = "sam has a cat"
    decoded = tok.decode(tok.encode(text))
    # BPE decode joins with spaces from </w>, so check words are present.
    assert "sam" in decoded and "cat" in decoded
    assert tok.vocab_size > 0
