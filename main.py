import pretty_midi
import numpy as np
import os
import json
from pathlib import Path

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
                # Notes overlap - keep the one with lower pitch
                if note.pitch < current_note.pitch:
                    current_note = note
                # If current note has lower pitch, just skip this one
                continue
            else:
                filtered_notes.append(current_note)
                current_note = note
        
        # Don't forget to append the last note
        if current_note:
            filtered_notes.append(current_note)
        
        # Process the filtered notes
        for i, note in enumerate(filtered_notes):
            # Quantize start and end times to nearest 12th note
            start = round(note.start / self.timestep) * self.timestep
            end = round(note.end / self.timestep) * self.timestep
            duration = end - start
            
            # Calculate rest before this note
            if i > 0:
                prev_end = round(filtered_notes[i-1].end / self.timestep) * self.timestep
                rest_duration = start - prev_end
                if rest_duration > self.timestep/2:  # threshold for minimal rest
                    rest_steps = round(rest_duration / self.timestep)
                    notes.append({
                        'type': 'rest',
                        'duration_steps': rest_steps
                    })
            
            # Add the note
            duration_steps = round(duration / self.timestep)
            notes.append({
                'type': 'note',
                'pitch': note.pitch,
                'duration_steps': duration_steps
            })
        
        return notes

    def create_tokens(self, notes):
        tokens = []
        
        # New token mapping:
        # Notes: 0-127 (unchanged, MIDI standard)
        # Rest marker: 200
        # Duration tokens: 300-400 (giving us space for up to 100 different duration values)
        
        for note in notes:
            if note['type'] == 'rest':
                tokens.append(200)  # Rest token now uses 200 instead of 128
                # Duration tokens now start at 300
                duration_token = min(note['duration_steps'] // 12, 99)  # Max of 99 to keep under 400
                tokens.append(300 + duration_token)
            else:
                tokens.append(note['pitch'])
                # Duration tokens now start at 300
                duration_token = min(note['duration_steps'] // 12, 99)  # Max of 99 to keep under 400
                tokens.append(300 + duration_token)
        
        return tokens

    def process_directory(self, midi_dir, output_file):
        midi_dir = Path(midi_dir)
        all_sequences = []
        
        # Process each MIDI file
        for midi_path in midi_dir.glob('**/*.mid*'):  # catches both .mid and .midi
            try:
                notes = self.process_monophonic_midi(str(midi_path))
                if notes:  # if not None (no overlapping notes)
                    tokens = self.create_tokens(notes)
                    all_sequences.append({
                        'file': str(midi_path.relative_to(midi_dir)),
                        'tokens': tokens
                    })
            except Exception as e:
                print(f"Error processing {midi_path}: {e}")
        
        # Save all sequences to a file
        with open(output_file, 'w') as f:
            json.dump(all_sequences, f)
        
        print(f"Processed {len(all_sequences)} files successfully")
        return all_sequences

def main():
    # Create dataset
    creator = MelodyDatasetCreator()
    
    # Process all MIDI files in directory
    midi_dir = "./files"
    output_file = "melody_dataset.json"
    
    sequences = creator.process_directory(midi_dir, output_file)
    
    # Print some statistics
    if sequences:
        num_sequences = len(sequences)
        total_tokens = sum(len(seq['tokens']) for seq in sequences)
        avg_length = total_tokens / num_sequences
        
        print(f"\nDataset Statistics:")
        print(f"Number of sequences: {num_sequences}")
        print(f"Average sequence length: {avg_length:.2f} tokens")
        print(f"Dataset saved to: {output_file}")

if __name__ == "__main__":
    main()