import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
import pickle
import numpy as np
import matplotlib.pyplot as plt

print("Loading token encoder and training data...")
# Load the encoder that was saved during data preparation
with open('token_encoder.pkl', 'rb') as f:
    token_encoder = pickle.load(f)
print(f"Number of unique tokens: {len(token_encoder.classes_)}")

# Load the prepared training data
X = np.load('X_train.npy')
y = np.load('y_train.npy')
print(f"Training data shape: X={X.shape}, y={y.shape}")

print("\nBuilding LSTM model architecture...")
# Define the improved LSTM model architecture with:
# 1. Bidirectional LSTM layers for better pattern recognition
# 2. Increased dropout and L2 regularization
# 3. Learning rate scheduling
model = Sequential()
model.add(Bidirectional(LSTM(512, return_sequences=True), 
                       input_shape=(X.shape[1], X.shape[2]), 
                       name='lstm_layer_1'))
model.add(Bidirectional(LSTM(256, return_sequences=False),
                       name='lstm_layer_2'))
model.add(Dropout(0.4, name='dropout_1'))
model.add(Dense(256, activation='relu', 
                kernel_regularizer=regularizers.l2(0.01),
                name='dense_1'))
model.add(Dropout(0.4, name='dropout_2'))
model.add(Dense(len(token_encoder.classes_), activation='softmax', 
                name='output_layer'))

print("\nCompiling model...")
# Configure training parameters with learning rate scheduler
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Show detailed model architecture
print("\nModel Architecture:")
model.summary()

# Training configuration
epochs = 5
batch_size = 64
validation_split = 0.2

# Setup callbacks
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.0001,
    verbose=1
)

print(f"\nStarting training for {epochs} epochs with batch size {batch_size}...")
print(f"Using {validation_split*100}% of data for validation")

# Train the model with callbacks
history = model.fit(
    X, y,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    verbose=1,
    callbacks=[tensorboard_callback, lr_scheduler]
)

print("\nSaving trained model...")
model.save('melody_lstm_model.h5')
print("Model saved successfully as 'melody_lstm_model.h5'")

# Print final training results
final_loss = history.history['loss'][-1]
final_accuracy = history.history['accuracy'][-1]
final_val_loss = history.history['val_loss'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]

print(f"\nFinal Training Results:")
print(f"Training Loss: {final_loss:.4f}")
print(f"Training Accuracy: {final_accuracy:.4f}")
print(f"Validation Loss: {final_val_loss:.4f}") 
print(f"Validation Accuracy: {final_val_accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
