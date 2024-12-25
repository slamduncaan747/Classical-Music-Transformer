import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
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

class MelodyDataset(Dataset):
    def __init__(self, json_file, sequence_length=64):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.sequence_length = sequence_length
        self.tokens = []
        for item in data:
            self.tokens.extend(item['tokens'])
        
        # Print token statistics
        print("Token statistics:")
        print(f"Min token value: {min(self.tokens)}")
        print(f"Max token value: {max(self.tokens)}")
        print(f"Total tokens: {len(self.tokens)}")
        
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
    
    def __len__(self):
        return len(self.tokens) - self.sequence_length - 1
    
    def __getitem__(self, idx):
        sequence = self.tokens[idx:idx + self.sequence_length]
        target = self.tokens[idx + 1:idx + self.sequence_length + 1]
        return sequence, target

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (sequences, targets) in enumerate(dataloader):
        # Print shape and value information for debugging
        if batch_idx == 0:
            print(f"\nSequences shape: {sequences.shape}")
            print(f"Sequences min: {sequences.min()}, max: {sequences.max()}")
            print(f"Targets shape: {targets.shape}")
            print(f"Targets min: {targets.min()}, max: {targets.max()}")
        
        sequences, targets = sequences.to(device), targets.to(device)
        
        optimizer.zero_grad()
        output = model(sequences)  # shape: [batch_size, seq_length, vocab_size]
        
        # Reshape for cross entropy
        output = output.view(-1, model.vocab_size)  # shape: [batch_size * seq_length, vocab_size]
        targets = targets.view(-1)  # shape: [batch_size * seq_length]
        
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader)

def train_model(model, json_file, epochs=10, batch_size=32, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    dataset = MelodyDataset(json_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        avg_loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Average loss: {avg_loss:.4f}")
        
        # Save model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'melody_model_epoch_{epoch+1}.pt')

if __name__ == "__main__":
    # Initialize model with vocab size based on your actual token range
    # We'll print the token range first to determine this
    
    with open('melody_dataset.json', 'r') as f:
        data = json.load(f)
    
    all_tokens = []
    for item in data:
        all_tokens.extend(item['tokens'])
    
    vocab_size = max(all_tokens) + 1
    print(f"\nRequired vocabulary size: {vocab_size}")
    
    model = MelodyTransformer(vocab_size=vocab_size)
    print(f"Model vocabulary size: {model.vocab_size}")
    
    # Train model
    train_model(model, 
                json_file='melody_dataset.json',
                epochs=20,
                batch_size=32,
                lr=0.001)