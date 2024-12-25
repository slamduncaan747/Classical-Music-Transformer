import torch
import torch.nn as nn
import math

class MelodyTransformer(nn.Module):
    def __init__(self, 
                 vocab_size=141,    # 0-127 for pitch, 128 for rest, 129-140 for durations
                 d_model=256,       # embedding dimension
                 nhead=4,           # number of attention heads
                 num_layers=4,      # number of transformer layers
                 max_seq_length=512):  # maximum sequence length
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        seq_length = x.size(1)
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pe[:seq_length]
        x = self.transformer_encoder(x)
        output = self.fc_out(x)
        
        return output