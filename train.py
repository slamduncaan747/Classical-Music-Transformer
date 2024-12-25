import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle

# Positional Encoding
def positional_encoding(position, d_model):
    angles = torch.arange(d_model)[None, :] / torch.pow(10000, (2 * (torch.arange(d_model) // 2)) / float(d_model))
    pos_encoding = torch.zeros((position, d_model))
    pos_encoding[:, 0::2] = torch.sin(torch.arange(position)[:, None] * angles[:, 0::2])
    pos_encoding[:, 1::2] = torch.cos(torch.arange(position)[:, None] * angles[:, 1::2])
    return pos_encoding.unsqueeze(0)  # Add batch dimension

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        if len(x.shape) == 4:  # If x is incorrectly 4D
            x = x.squeeze(1)   # Remove the extra dimension (likely caused by unsqueeze or view)

        # Transpose for MultiheadAttention if batch_first=False
        if not self.attention.batch_first:
            x = x.transpose(0, 1)  # [seq_len, batch_size, embed_dim]

        # MultiheadAttention expects [seq_len, batch_size, embed_dim]
        attn_output, _ = self.attention(x, x, x)

        # Residual connection
        x = attn_output + x

        # Apply layer norm
        x = self.norm1(x)

        # Feedforward
        ff_output = self.ff(x)
        ff_output = self.dropout(ff_output)

        # Residual connection
        x = ff_output + x
        x = self.norm2(x)

        # If batch_first=False, transpose back for subsequent layers
        if not self.attention.batch_first:
            x = x.transpose(0, 1)  # [batch_size, seq_len, embed_dim]

        return x


# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, num_heads, num_layers, ff_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = x.transpose(0, 1)  # Switch to (seq_len, batch_size, d_model) for MultiheadAttention
        for layer in self.layers:
            x = layer(x)
        x = x[-1, :, :]  # Take the output of the last token (sequence end)
        return self.fc(x)

# Load the prepared training data and encoder
print("Loading token encoder and training data...")
with open('token_encoder.pkl', 'rb') as f:
    token_encoder = pickle.load(f)

X = np.load('X_train.npy')
y = np.load('y_train.npy')

# Convert y to one-hot encoded format
y = torch.tensor(y, dtype=torch.long)

# Split into training and validation sets
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# Create DataLoaders
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), y_train)
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.long), y_val)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Model Configuration
d_model = 128
num_heads = 8
num_layers = 4
ff_dim = 512
max_seq_len = X.shape[1]
vocab_size = len(token_encoder.classes_)

# Initialize Model
model = TransformerModel(vocab_size, max_seq_len, d_model, num_heads, num_layers, ff_dim)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

# Train the Model
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)
