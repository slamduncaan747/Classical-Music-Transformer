import torch
import argparse
import pretty_midi
import os
from midi2audio import FluidSynth
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Use the same parameters as your training script
params = {
    "block_size": 128,
    "n_embed": 256,
    "n_layers": 12,
    "n_head": 16,
    "dropout": 0.2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


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
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(params["n_embed"], params["n_embed"])
        self.dropout = nn.Dropout(params["dropout"])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


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
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention(
            params["n_head"], params["n_embed"] // params["n_head"]
        )
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(params["n_embed"])
        self.ln2 = nn.LayerNorm(params["n_embed"])

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class ChordTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.note_embed = nn.Embedding(vocab_size, params["n_embed"])
        self.pos_embed = nn.Embedding(4, params["n_embed"])
        self.chord_embed = nn.Embedding(params["block_size"], params["n_embed"])
        self.blocks = nn.Sequential(*[Block() for _ in range(params["n_layers"])])
        self.ln_f = nn.LayerNorm(params["n_embed"])
        self.head = nn.Linear(params["n_embed"], vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        note_pos = torch.arange(4, device=idx.device).repeat(T // 4 + 1)[:T]
        chord_pos = torch.arange(T, device=idx.device) // 4
        tok_emb = self.note_embed(idx)
        pos_emb = self.pos_embed(note_pos) + self.chord_embed(chord_pos)
        x = tok_emb + pos_emb.unsqueeze(0)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, None

    def generate(self, idx, max_new_chords, temperature=0.8, top_k=5):
        for _ in range(max_new_chords * 4):
            idx_cond = (
                idx
                if idx.size(1) <= params["block_size"]
                else idx[:, -params["block_size"] :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def load_model(model_path, vocab_size=130):
    model = ChordTransformer(vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=params["device"]))
    model = model.to(params["device"])
    model.eval()
    return model


def generate_midi(sequence, output_file="generated"):
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(0)
    chord_duration = 0.5
    current_time = 0.0

    for i in range(0, len(sequence), 4):
        chord = sequence[i : i + 4]
        for note in chord:
            if 0 <= note < 128:
                midi_note = pretty_midi.Note(
                    velocity=100,
                    pitch=note,
                    start=current_time,
                    end=current_time + chord_duration,
                )
                piano.notes.append(midi_note)
        current_time += chord_duration

    pm.instruments.append(piano)
    pm.write(f"{output_file}.mid")
    return f"{output_file}.mid"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate music with trained transformer"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="music_transformer_model.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Optional starting context as comma-separated integers",
    )
    parser.add_argument(
        "--length", type=int, default=100, help="Number of chords to generate"
    )
    parser.add_argument(
        "--output", type=str, default="generated", help="Base name for output files"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature (0.1-2.0)"
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Top-k sampling (0 for no limit)"
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Prepare context
    if args.context:
        context = torch.tensor(
            [int(x) for x in args.context.split(",")],
            dtype=torch.long,
            device=params["device"],
        ).unsqueeze(0)
    else:
        # Random start from valid notes (0-127)
        context = torch.randint(0, 128, (1, 4), device=params["device"])

    # Generate sequence
    with torch.no_grad():
        generated = model.generate(context, args.length, args.temperature, args.top_k)

    # Process and save
    sequence = generated[0].cpu().tolist()
    midi_file = generate_midi(sequence, args.output)

    print(f"Successfully generated {midi_file}")
    print("First 16 generated notes:", sequence[:16])
