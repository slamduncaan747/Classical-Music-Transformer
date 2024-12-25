import pretty_midi
import numpy as np
import os
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

class MelodyDatasetCreator:
    def __init__(self, timestep=1/128):  # 1/128 of a quarter note
        self.timestep = timestep
    
    def process_monophonic_midi(self, midi_file):
        pm = pretty_midi.PrettyMIDI(midi_file)
        notes = []

        instrument = pm.instruments[0]
        
        # Sort notes by start time and pitch (lower pitch first)
        sorted_notes = sorted(instrument.notes, key=lambda x: (x.start, x.pitch))
        
        # Filter out overlapping notes, keeping the lower pitch
        filtered_notes = []
        current_note = None
        
        for note in sorted_notes:
            if current_note is None:
                current_note = note
                continue
            
            if note.start < current_note.end:
                if note.pitch < current_note.pitch:
                    current_note = note
                continue
            else:
                filtered_notes.append(current_note)
                current_note = note
        
        if current_note:
            filtered_notes.append(current_note)
        
        for i, note in enumerate(filtered_notes):
            start = round(note.start / self.timestep) * self.timestep
            end = round(note.end / self.timestep) * self.timestep
            duration = end - start

            if i > 0:
                prev_end = round(filtered_notes[i-1].end / self.timestep) * self.timestep
                rest_duration = start - prev_end
                if rest_duration > self.timestep / 2:
                    rest_steps = round(rest_duration / self.timestep)
                    notes.append({'type': 'rest', 'duration_steps': rest_steps})

            duration_steps = round(duration / self.timestep)
            notes.append({'type': 'note', 'pitch': note.pitch, 'duration_steps': duration_steps})
        
        return notes

    def create_tokens(self, notes):
        tokens = []
        for note in notes:
            if note['type'] == 'rest':
                tokens.append(200)
                duration_token = min(note['duration_steps'] // 12, 99)
                tokens.append(300 + duration_token)
            else:
                tokens.append(note['pitch'])
                duration_token = min(note['duration_steps'] // 12, 99)
                tokens.append(300 + duration_token)
        return tokens

    def process_directory(self, midi_dir):
        midi_dir = Path(midi_dir)
        all_sequences = []

        for midi_path in midi_dir.glob('**/*.mid*'):
            try:
                notes = self.process_monophonic_midi(str(midi_path))
                if notes:
                    tokens = self.create_tokens(notes)
                    all_sequences.append(tokens)
            except Exception as e:
                print(f"Error processing {midi_path}: {e}")
        
        return all_sequences

class MelodyDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return torch.tensor(sequence, dtype=torch.long)

def main():
    creator = MelodyDatasetCreator()
    midi_dir = "./files"

    # Process MIDI files
    sequences = creator.process_directory(midi_dir)
    
    # Create dataset
    dataset = MelodyDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0))

    # Example usage
    for batch in dataloader:
        print("Batch shape:", batch.shape)
        break

if __name__ == "__main__":
    main()
