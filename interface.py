"""
TinyMind — easy web interface (from the full-prompt spec, section 7).

Run:
    pip install -r requirements.txt
    python interface.py
Then open http://localhost:8000 in a browser.

No flags for the user to remember: deep checkpoints always use
--config deep / --tokenizer bpe / --vocab bpe_vocab.json internally,
exactly as generate.py requires. Checkpoints load lazily, only when
first selected, and stay in memory afterwards. A missing checkpoint
or a missing torch install gives a clear on-page error, never fake
output. Only the Python standard library plus torch are used.
"""
import html
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

MODELS = [
    ("child", "Child", "checkpoints/deep_child.pt"),
    ("teen", "Teen", "checkpoints/deep_teen.pt"),
    ("adult_science", "Adult — Science", "checkpoints/deep_adult_science.pt"),
    ("adult_stories", "Adult — Stories", "checkpoints/deep_adult_stories.pt"),
    ("adult_code", "Adult — Code", "checkpoints/deep_adult_code.pt"),
    ("base", "Base, no forgetting", "checkpoints/deep_child_base.pt"),
]
MODEL_BY_KEY = {key: (label, path) for key, label, path in MODELS}
COMPARE_KEYS = ["child", "teen", "adult_stories"]

_cache = {}  # key -> (model, tokenizer)


def load_model(key):
    """Load (and cache) one deep checkpoint. Raises a readable error."""
    if key in _cache:
        return _cache[key]
    label, rel = MODEL_BY_KEY[key]
    path = HERE / rel
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint missing: {rel} — download the split parts "
            "(code zip first, then this checkpoint into checkpoints/) "
            "before generating with {label}."
        )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is not installed in this environment. Run "
            "'pip install -r requirements.txt' (or "
            "'pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu' then 'pip install numpy') "
            "and restart interface.py."
        ) from exc
    from config import get_config
    from model import TinyMind
    from tokenizer_loader import load_tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer("bpe", "bpe_vocab.json")
    cfg = get_config("deep")
    model = TinyMind(tok.vocab_size, n_embd=cfg.n_embd, n_head=cfg.n_head,
                     n_layer=cfg.n_layer, block_size=cfg.block_size,
                     decay=0.02).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    _cache[key] = (model, tok, device)
    return _cache[key]


def generate(key, prompt, temperature, max_new_tokens):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is not installed in this environment. Run "
            "'pip install -r requirements.txt' and restart interface.py."
        ) from exc

    model, tok, device = load_model(key)
    ids = torch.tensor([tok.encode(prompt, add_bos_eos=False)], device=device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens, temperature=temperature)
    return tok.decode(out[0].tolist())


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TinyMind — a model that grows up</title>
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 1em;line-height:1.5}
button,.model-btn{padding:.6em 1em;margin:.2em;cursor:pointer}
.model-btn.active{background:#222;color:#fff}
textarea,input[type=text]{width:100%;padding:.6em;font-size:1em}
.cols{display:grid;grid-template-columns:repeat(3,1fr);gap:1em}
.cols>div{border:1px solid #ccc;padding:1em;white-space:pre-wrap;overflow-wrap:anywhere}
table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.4em .7em;text-align:left}
.limits{background:#fff8e1;border:1px solid #e6c200;padding:1em}
.error{color:#a00;font-weight:bold}
</style></head><body>
<h1>TinyMind — a model that grows up</h1>
<p>Childhood &rarr; adolescence &rarr; adulthood. Same base, different ages.</p>
<form method="post" action="/generate">
<p><strong>Model:</strong><br>
__MODEL_BUTTONS__
<input type="hidden" name="model" id="model-field" value="teen"></p>
<p><strong>Prompt:</strong><br>
<input type="text" name="prompt" id="prompt-field" value="When Sam went to the">
<br>
<button type="button" onclick="setPrompt('Sam has a')">Sam has a</button>
<button type="button" onclick="setPrompt('The moon is')">The moon is</button>
<button type="button" onclick="setPrompt('One quiet night,')">One quiet night,</button></p>
<p>Temperature (0.2–1.2): <input type="range" name="temperature" min="0.2" max="1.2" step="0.1" value="0.8" oninput="this.nextElementSibling.textContent=this.value"> <span>0.8</span><br>
Length (10–50 tokens): <input type="range" name="length" min="10" max="50" step="1" value="30" oninput="this.nextElementSibling.textContent=this.value"> <span>30</span></p>
<p><label><input type="checkbox" name="compare" value="1"> Compare all ages with the same prompt</label></p>
<p><button type="submit" style="font-size:1.1em">Generate</button></p>
</form>
<script>
function setPrompt(t){document.getElementById('prompt-field').value=t;}
function pick(k,btn){document.getElementById('model-field').value=k;
document.querySelectorAll('.model-btn').forEach(b=>b.classList.remove('active'));
btn.classList.add('active');}
document.querySelectorAll('.model-btn').forEach(b=>{if(b.dataset.key==='teen')b.classList.add('active');});
</script>
__RESULT__
<h2>Verified deep results (23 October 2026)</h2>
<table><tr><th>Stage</th><th>Training loss</th><th>Held-out</th></tr>
<tr><td>Childhood (TinyStories, decay 0.02)</td><td>4.9378 &rarr; 3.8729</td><td>3.6833</td></tr>
<tr><td>Base, no forgetting (decay 0)</td><td>4.7343 &rarr; 3.7973</td><td>3.6518</td></tr>
<tr><td>Adolescence (templated)</td><td>0.2451 &rarr; 0.0664</td><td>teen on TinyStories: 6.3255</td></tr>
<tr><td>Adults</td><td>code 0.0836 / science 0.4816 / stories 0.3912</td><td>on TinyStories: 6.37–7.03</td></tr></table>
<div class="limits"><h2>Limits — read these first</h2>
<ul>
<li>Forgetting shows <strong>no clear benefit</strong> at this scale: training gap 0.0756, held-out gap 0.0315, single seed.</li>
<li>Continued training <strong>overwrote childhood</strong> — no rehearsal was used (teen 6.3255 vs child 3.6833 on TinyStories).</li>
<li>Adolescence and adult field data are <strong>templated and heavily repeated</strong> (code: only 4 snippets). Low adult losses are repetition, not proven aptitude.</li>
<li>Deep child generation is <strong>garbled</strong>: real TinyStories are far more varied than templated data.</li>
<li>Low loss on templated data does <strong>not</strong> mean general language ability.</li>
</ul>
<p>Not posted to GitHub as finished, and not mastered, until the author can explain it out loud without notes.</p></div>
</body></html>"""

RESULT_TEMPLATE = "<h2>Output</h2><p>Model, checkpoint, and settings used for this output: {label}</p>{body}"


def model_buttons():
    return "".join(
        f'<button type="button" class="model-btn" data-key="{k}" '
        f'onclick="pick(\'{k}\',this)">{html.escape(label)}</button>'
        for k, label, _ in MODELS
    )


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if urlparse(self.path).path != "/":
            self.send_error(404)
            return
        body = PAGE.replace("__MODEL_BUTTONS__", model_buttons()).replace("__RESULT__", "")
        self._send(body)

    def do_POST(self):
        if urlparse(self.path).path != "/generate":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        form = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
        prompt = (form.get("prompt", ["When Sam went to the"])[0] or "").strip()
        temperature = float(form.get("temperature", ["0.8"])[0])
        max_new = int(form.get("length", ["30"])[0])
        temperature = min(1.2, max(0.2, temperature))
        max_new = min(50, max(10, max_new))
        compare = bool(form.get("compare"))
        try:
            if compare:
                cols = []
                for key in COMPARE_KEYS:
                    label, rel = MODEL_BY_KEY[key]
                    try:
                        text = generate(key, prompt, temperature, max_new)
                    except Exception as exc:  # show per-column, don't fake it
                        text = f"Error: {exc}"
                    cols.append(f"<div><h3>{html.escape(label)}</h3>"
                                f"<p>{html.escape(text)}</p></div>")
                body = RESULT_TEMPLATE.format(
                    label=html.escape(
                        f"Compare: Child / Teen / Adult Stories, prompt={prompt!r}, "
                        f"temperature={temperature}, length={max_new}"),
                    body=f'<div class="cols">{"".join(cols)}</div>')
            else:
                key = form.get("model", ["teen"])[0]
                if key not in MODEL_BY_KEY:
                    key = "teen"
                label, rel = MODEL_BY_KEY[key]
                text = generate(key, prompt, temperature, max_new)
                body = RESULT_TEMPLATE.format(
                    label=html.escape(
                        f"{label} ({rel}), prompt={prompt!r}, "
                        f"temperature={temperature}, length={max_new}"),
                    body=f"<p style='white-space:pre-wrap'>{html.escape(text)}</p>")
        except Exception as exc:
            body = (f"<h2>Output</h2><p class='error'>Could not generate: "
                    f"{html.escape(str(exc))}</p>"
                    "<p>Nothing was faked — fix the error above (usually "
                    "install torch, or download the missing checkpoint) "
                    "and press Generate again.</p>")
        page = PAGE.replace("__MODEL_BUTTONS__", model_buttons()).replace("__RESULT__", body)
        self._send(page)

    def _send(self, body):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print("TinyMind interface: open http://localhost:8000")
    print("If generation fails with 'torch is not installed', run:")
    print("  pip install -r requirements.txt")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
