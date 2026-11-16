"""
TinyMind model — a tiny transformer with human-like forgetting.

Normal attention: every past word can matter equally.
Forgetting attention: a past word matters less the further back it is.
  weight(distance) = exp(-decay * max(0, distance - echoic))
  echoic = 3-7 words that are kept perfectly, like human echoic memory.
  decay = 0 means normal attention. decay = 0.02 is a gentle forgetting.

This is the one architectural change in TinyMind.
Everything else is a standard tiny transformer:
  embedding -> positional embedding -> 2 transformer blocks -> output head.
  Each block: attention + MLP, with LayerNorm and residual connections.

If you can draw this on a whiteboard and explain decay and echoic,
you can explain the whole model in an interview.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ForgettingAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, decay=0.0, echoic=5):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.decay = decay
        self.echoic = echoic
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        # Causal mask: a word can only look at itself and earlier words.
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size))
        # Distance matrix: distance[i, j] = i - j for j <= i, else 0.
        # Used only to build the forgetting bias, not learned.
        idx = torch.arange(block_size)
        dist = (idx.view(-1, 1) - idx.view(1, -1)).clamp(min=0)
        self.register_buffer("distance", dist)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, heads, T, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Raw attention scores, scaled.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Forgetting bias: older words get a negative bias, so after
        # softmax they get less weight. Words inside echoic keep full weight.
        if self.decay > 0:
            eff_dist = (self.distance[:T, :T] - self.echoic).clamp(min=0)
            att = att + (-self.decay * eff_dist).view(1, 1, T, T)
        # Causal mask: future words get -inf, so they are ignored.
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v  # B, heads, T, head_dim
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, decay, echoic):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = ForgettingAttention(n_embd, n_head, block_size, decay, echoic)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        # Residual: x + attention(x), then x + mlp(x). This lets gradients flow.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyMind(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=2,
                 block_size=64, decay=0.0, echoic=5):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, decay, echoic)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            # Predict the next word: logits for position t predict target t.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        # Greedy if temperature is very low, sampling otherwise.
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            if temperature < 0.2:
                next_id = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
