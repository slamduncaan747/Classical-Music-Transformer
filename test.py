import numpy as np
import pickle
from tensorflow.keras.models import load_model

def generate_melody(model, token_encoder, sequence_length=64, seed=None, length=200):
    if seed is None:
        seed = np.random.choice(len(token_encoder.classes_), sequence_length)

    # Convert seed to appropriate shape for the model (1, sequence_length, 1)
    seed = np.reshape(seed, (1, len(seed), 1)) / float(len(token_encoder.classes_))

    # Generate tokens
    generated_tokens = []
    for _ in range(length):
        prediction = model.predict(seed, verbose=0)
        predicted_token = np.argmax(prediction)

        generated_tokens.append(predicted_token)

        # Fix: Reshape predicted token to match seed dimensions (1, 1, 1)
        predicted_token_reshaped = np.array([[[predicted_token / float(len(token_encoder.classes_))]]])
        seed = np.append(seed[:, 1:, :], predicted_token_reshaped, axis=1)

    # Decode generated tokens back to the original tokens
    generated_sequence = token_encoder.inverse_transform(generated_tokens)
    return generated_sequence

model = load_model('melody_lstm_model.h5')
token_encoder = pickle.load(open('token_encoder.pkl', 'rb'))

# Generate a melody
generated_melody = generate_melody(model, token_encoder)
print("Generated Melody:")
print(generated_melody)
