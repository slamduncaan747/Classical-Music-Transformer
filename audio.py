import pretty_midi
import json
from pathlib import Path

class MelodyToAudio:
    def __init__(self, timestep=1/12):  # Match the timestep from dataset creation
        self.timestep = timestep
        
    def tokens_to_midi(self, tokens):
        # Create a PrettyMIDI object
        pm = pretty_midi.PrettyMIDI()
        
        # Create an instrument (piano by default)
        instrument = pretty_midi.Instrument(program=0)  # 0 is piano
        
        current_time = 0.0
        
        # Process tokens in pairs (pitch/rest, duration)
        for i in range(0, len(tokens), 2):
            if i + 1 >= len(tokens):
                break
                
            token = tokens[i]
            duration_steps = tokens[i + 1] - 128  # Subtract 128 to get actual duration
            duration = duration_steps * self.timestep
            
            if token == 128:  # Rest
                current_time += duration
            else:  # Note
                # Create a Note instance
                note = pretty_midi.Note(
                    velocity=100,  # Standard velocity
                    pitch=token,
                    start=current_time,
                    end=current_time + duration
                )
                instrument.notes.append(note)
                current_time += duration
        
        # Add the instrument to the PrettyMIDI object
        pm.instruments.append(instrument)
        return pm
    
    def save_as_mp3(self, midi_data, output_path):
        # First save as MIDI
        temp_midi_path = output_path.with_suffix('.mid')
        midi_data.write(str(temp_midi_path))
        
        try:
            # Convert MIDI to MP3 using fluidsynth
            import subprocess
            soundfont_path = "./SoundFont/FluidR3_GM.sf2"  # You'll need to specify this
            mp3_path = output_path.with_suffix('.mp3')
            
            # Convert MIDI to WAV
            subprocess.run([
                'fluidsynth',
                '-ni',
                soundfont_path,
                str(temp_midi_path),
                '-F',
                str(output_path.with_suffix('.wav'))
            ])
            
            # Convert WAV to MP3
            subprocess.run([
                'ffmpeg',
                '-i',
                str(output_path.with_suffix('.wav')),
                '-codec:a',
                'libmp3lame',
                '-qscale:a',
                '2',
                str(mp3_path)
            ])
            
            # Clean up temporary files
            temp_midi_path.unlink()
            output_path.with_suffix('.wav').unlink()
            
        except Exception as e:
            print(f"Error converting to MP3: {e}")
            print("Saving as MIDI file only")
            temp_midi_path.rename(output_path.with_suffix('.mid'))

def main():
    # Load the dataset
    with open("melody_dataset.json", 'r') as f:
        sequences = json.load(f)
    
    converter = MelodyToAudio()
    output_dir = Path("generated_melodies")
    output_dir.mkdir(exist_ok=True)
    
    # Convert the first sequence as an example
    if sequences:
        sequence = sequences[0]
        print(f"Converting sequence from: {sequence['file']}")
        
        midi_data = converter.tokens_to_midi(sequence['tokens'])
        output_path = output_dir / f"generated_melody_0"
        converter.save_as_mp3(midi_data, output_path)
        print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()