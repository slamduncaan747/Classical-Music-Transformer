import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
import numpy as np
import pandas as pd
import ast
import pretty_midi
from midi2audio import FluidSynth
import os

params = {
    "batch_size": 32,
    "block_size": 512,
    "max_iters": 1000,
    "eval_interval": 250,
    "learning_rate": 3e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "eval_iters": 200,
    "n_embed": 256,
    "n_layers": 12,
    "n_head": 16,
    "dropout": 0.35,
}


torch.manual_seed(1337)


def create_note_mapping():
    note_to_idx = {}

    for i in range(128):
        note_to_idx[str(i)] = i

    note_to_idx["START"] = 128
    note_to_idx["END"] = 129

    return note_to_idx


def load_music_data(filepath="choral_sequences.csv"):
    df = pd.read_csv(filepath)
    sequences = [ast.literal_eval(s) for s in df.sequence]

    flat_sequence = []
    for song in sequences:
        for chord in song:
            flat_sequence.extend(chord)

    data = torch.tensor(flat_sequence, dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(params["eval_iters"])
        for k in range(params["eval_iters"]):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(params["n_embed"], head_size, bias=False)
        self.query = nn.Linear(params["n_embed"], head_size, bias=False)
        self.value = nn.Linear(params["n_embed"], head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(params["block_size"], params["block_size"]))
        )
        self.dropout = nn.Dropout(params["dropout"])

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        q, k = apply_rotary_pos_emb(q, k)

        wei = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(params["n_embed"], params["n_embed"])
        self.dropout = nn.Dropout(params["dropout"])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(params["n_embed"], 4 * params["n_embed"]),
            nn.ReLU(),
            nn.Linear(4 * params["n_embed"], params["n_embed"]),
            nn.Dropout(params["dropout"]),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embed, n_head, head_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


def apply_rotary_pos_emb(q, k):
    B, T, head_size = q.shape
    dim = head_size // 2

    pos = torch.arange(T, device=q.device, dtype=q.dtype)
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, dim, device=q.device, dtype=q.dtype) / dim)
    )
    sinusoid_inp = torch.einsum("t,d->td", pos, inv_freq)
    cos = sinusoid_inp.cos()[None, :, :]
    sin = sinusoid_inp.sin()[None, :, :]

    q1, q2 = q[..., :dim], q[..., dim:]
    k1, k2 = k[..., :dim], k[..., dim:]

    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

    return q_rot, k_rot


class ChordTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.note_embed = nn.Embedding(vocab_size, params["n_embed"])
        self.pos_embed = nn.Embedding(4, params["n_embed"])  # Position in chord
        self.chord_embed = nn.Embedding(params["block_size"], params["n_embed"])

        self.blocks = nn.Sequential(
            *[
                Block(
                    params["n_embed"],
                    params["n_head"],
                    params["n_embed"] // params["n_head"],
                    params["dropout"],
                )
                for _ in range(params["n_layers"])
            ]
        )

        self.ln_f = nn.LayerNorm(params["n_embed"])
        self.head = nn.Linear(params["n_embed"], vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        seq_len = T
        note_pos = torch.arange(4, device=idx.device).repeat(seq_len // 4 + 1)[:seq_len]
        chord_pos = torch.arange(seq_len, device=idx.device) // 4

        tok_emb = self.note_embed(idx)
        pos_emb = self.pos_embed(note_pos) + self.chord_embed(chord_pos)
        x = tok_emb + pos_emb.unsqueeze(0)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx, max_new_chords):
        batch_size = idx.size(0)

        for _ in range(max_new_chords * 4):
            if idx.size(1) > params["block_size"]:
                idx_cond = idx[:, -params["block_size"] :]
            else:
                idx_cond = idx

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def process_generated_sequence(sequence):
    valid_notes = [n for n in sequence if 0 <= n < 128]

    chords = [valid_notes[i : i + 4] for i in range(0, len(valid_notes), 4)]

    processed_chords = []
    for chord in chords:
        if len(chord) < 4:
            chord += [0] * (4 - len(chord))
        processed_chords.append(chord[:4])

    return processed_chords


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - params["block_size"], (params["batch_size"],))
    x = torch.stack([data[i : i + params["block_size"]] for i in ix])
    y = torch.stack([data[i + 1 : i + params["block_size"] + 1] for i in ix])
    return x.to(params["device"]), y.to(params["device"])


def initialize_model():
    note_to_idx = create_note_mapping()
    vocab_size = len(note_to_idx)
    print(vocab_size)

    return ChordTransformer(vocab_size=vocab_size).to(params["device"]), vocab_size


if __name__ == "__main__":
    train_data, val_data = load_music_data()
    model, vocab_size = initialize_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"])

    for iter in range(params["max_iters"]):
        if iter % params["eval_interval"] == 0:
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(loss.item())
    torch.save(model.state_dict(), "music_transformer_model.pt")

    context = train_data[:4].unsqueeze(0).to(params["device"])
    model.eval()
    with torch.no_grad():
        generated = model.generate(context, max_new_chords=100)

    generated_sequence = generated[0].tolist()

    processed_chords = process_generated_sequence(generated_sequence)

    print("Successfully created generated_chorale.mid")
