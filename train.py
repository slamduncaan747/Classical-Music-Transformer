import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical

# Positional Encoding
def positional_encoding(position, d_model):
    angles = np.arange(d_model)[np.newaxis, :] / np.power(10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
    pos_encoding = np.zeros((position, d_model))
    pos_encoding[:, 0::2] = np.sin(np.arange(position)[:, np.newaxis] * angles[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(np.arange(position)[:, np.newaxis] * angles[:, 1::2])
    return tf.constant(pos_encoding[np.newaxis, ...], dtype=tf.float32)

# Transformer Block
def transformer_block(inputs, d_model, num_heads, ff_dim, dropout=0.1):
    attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attention = tf.keras.layers.Dropout(dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + inputs)
    
    ff = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation='relu'),
        tf.keras.layers.Dense(d_model),
    ])
    ff_output = ff(attention)
    ff_output = tf.keras.layers.Dropout(dropout)(ff_output)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(ff_output + attention)

# Transformer Model
def build_transformer(vocab_size, max_seq_len, d_model, num_heads, num_layers, ff_dim):
    inputs = tf.keras.layers.Input(shape=(max_seq_len,))
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    pos_encoding = positional_encoding(max_seq_len, d_model)
    x = embeddings + pos_encoding[:, :max_seq_len, :]

    for _ in range(num_layers):
        x = transformer_block(x, d_model, num_heads, ff_dim)

    x = x[:, -1, :]
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

# Load the prepared training data and encoder
print("Loading token encoder and training data...")
with open('token_encoder.pkl', 'rb') as f:
    token_encoder = pickle.load(f)

X = np.load('X_train.npy')
y = np.load('y_train.npy')

# Convert y to one-hot encoded format
y = to_categorical(y, num_classes=len(token_encoder.classes_))

# Split into training and validation sets
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64).shuffle(1000)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)

# Build and Compile the Model
d_model = 128
num_heads = 8
num_layers = 4
ff_dim = 512
max_seq_len = X.shape[1]
vocab_size = len(token_encoder.classes_)

model = build_transformer(vocab_size, max_seq_len, d_model, num_heads, num_layers, ff_dim)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)
