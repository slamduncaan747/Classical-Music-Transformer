import numpy as np
import json
from sklearn.preprocessing import LabelEncoder

# Load the dataset
with open('melody_dataset.json', 'r') as f:
    sequences = json.load(f)

# Flatten the sequences and create a list of all tokens
all_tokens = []
for seq in sequences:
    all_tokens.extend(seq['tokens'])

# Encode the tokens (e.g., using LabelEncoder or one-hot encoding)
token_encoder = LabelEncoder()
all_tokens_encoded = token_encoder.fit_transform(all_tokens)

# Define a sequence length for the LSTM input
sequence_length = 256  # Adjust based on your data and memory constraints
X = []
y = []

# Create input-output pairs
for i in range(len(all_tokens_encoded) - sequence_length):
    X.append(all_tokens_encoded[i:i + sequence_length])
    y.append(all_tokens_encoded[i + sequence_length])

# Convert to numpy arrays for training
X = np.array(X)
y = np.array(y)

# Reshape X for LSTM input (samples, timesteps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Normalize the input (if needed, depending on your token scale)
X = X / float(len(token_encoder.classes_))

print(f"Prepared {len(X)} sequences for training.")

# Save the prepared data and encoder
np.save('X_train.npy', X)
np.save('y_train.npy', y)
import pickle
with open('token_encoder.pkl', 'wb') as f:
    pickle.dump(token_encoder, f)

print("Saved training data and encoder to disk.")
