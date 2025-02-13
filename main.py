import os
import pandas as pd
import ast

# Special tokens and constants
START_TOKEN = 128
END_TOKEN = 129
CHORD_SIZE = 4


def create_note_mapping():
    """Create mapping between MIDI notes and integer indices"""
    note_to_idx = {}
    # MIDI notes 0-127
    for i in range(128):
        note_to_idx[str(i)] = i
    # Special tokens
    note_to_idx["START"] = START_TOKEN
    note_to_idx["END"] = END_TOKEN
    idx_to_note = {v: k for k, v in note_to_idx.items()}
    return note_to_idx, idx_to_note


def process_chorale(file_path, song_id, note_to_idx):
    """
    Process a chorale into sequence of chords with special tokens
    Returns: (song_id, sequence_string)
    """
    df = pd.read_csv(file_path)
    chords = []

    # Convert chords to indices
    for row in df.itertuples(index=False):
        chord = [note_to_idx[str(n)] for n in row][:CHORD_SIZE]  # Use first 4 notes
        chords.append(chord)

    # Add start/end markers as full chords
    start_chord = [START_TOKEN] * CHORD_SIZE
    end_chord = [END_TOKEN] * CHORD_SIZE
    full_sequence = [start_chord] + chords + [end_chord]

    return (song_id, str(full_sequence))  # Store as string for CSV


def process_dataset():
    note_to_idx, _ = create_note_mapping()
    all_data = []
    song_counter = 0

    for split in ["train", "valid", "test"]:
        split_path = os.path.join("./", split)
        if not os.path.exists(split_path):
            continue

        for fname in sorted(os.listdir(split_path)):
            if fname.endswith(".csv"):
                file_path = os.path.join(split_path, fname)
                song_id, sequence = process_chorale(
                    file_path, song_counter, note_to_idx
                )
                all_data.append({"song_id": song_id, "sequence": sequence})
                song_counter += 1

    # Save to CSV with sequences as strings
    df = pd.DataFrame(all_data)
    df.to_csv("choral_sequences.csv", index=False)
    print(f"Processed {len(df)} songs. Sequence format:")
    print("START_CHORD, CHORD_1, CHORD_2, ..., END_CHORD")
    print("Each chord is: [note1_idx, note2_idx, note3_idx, note4_idx]")


if __name__ == "__main__":
    process_dataset()
