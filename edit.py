import mido
from mido import MidiFile, MidiTrack, Message
from itertools import groupby


def process_midi(input_midi, output_midi):
    midi = MidiFile(input_midi)
    new_midi = MidiFile()
    tracks = [MidiTrack() for _ in range(4)]
    new_midi.tracks.extend(tracks)

    # Collect all notes with start/end times
    notes = []
    active_notes = {}  # {(channel, note): (start_time, velocity)}

    # Parse all notes from input MIDI
    for track in midi.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type == "note_on":
                if msg.velocity > 0:
                    key = (msg.channel, msg.note)
                    if key not in active_notes:
                        active_notes[key] = (abs_time, msg.velocity)
                else:  # Handle note_off as velocity=0 note_on
                    key = (msg.channel, msg.note)
                    if key in active_notes:
                        start, velocity = active_notes.pop(key)
                        notes.append(
                            {
                                "start": start,
                                "end": abs_time,
                                "pitch": msg.note,
                                "velocity": velocity,
                            }
                        )
            elif msg.type == "note_off":
                key = (msg.channel, msg.note)
                if key in active_notes:
                    start, velocity = active_notes.pop(key)
                    notes.append(
                        {
                            "start": start,
                            "end": abs_time,
                            "pitch": msg.note,
                            "velocity": velocity,
                        }
                    )

    # Process and distribute notes
    notes.sort(key=lambda x: (x["start"], x["pitch"]))

    # Group notes by start time and sort within each group
    grouped_notes = []
    for start_time, group in groupby(notes, key=lambda x: x["start"]):
        sorted_group = sorted(list(group), key=lambda x: x["pitch"])
        grouped_notes.append((start_time, sorted_group))

    # Distribute notes to tracks and merge sustained notes
    track_notes = [[] for _ in range(4)]
    for start_time, group in grouped_notes:
        for i, note in enumerate(group[:4]):  # Only take first 4 notes per time
            track_idx = i
            if not track_notes[track_idx]:
                track_notes[track_idx].append(note)
            else:
                last_note = track_notes[track_idx][-1]
                if (
                    note["pitch"] == last_note["pitch"]
                    and note["start"] == last_note["end"]
                ):
                    # Merge with previous note
                    track_notes[track_idx][-1]["end"] = note["end"]
                else:
                    track_notes[track_idx].append(note)

    # Convert to MIDI messages with proper timing
    for track_idx in range(4):
        abs_time = 0
        for note in track_notes[track_idx]:
            delta_on = note["start"] - abs_time
            tracks[track_idx].append(
                Message(
                    "note_on",
                    note=note["pitch"],
                    velocity=note["velocity"],
                    time=delta_on,
                )
            )
            delta_off = note["end"] - note["start"]
            tracks[track_idx].append(
                Message("note_off", note=note["pitch"], velocity=0, time=delta_off)
            )
            abs_time = note["end"]
        tracks[track_idx].append(mido.MetaMessage("end_of_track", time=0))

    new_midi.save(output_midi)


# Usage example
process_midi("generated.mid", "processed_chorale_fixed.mid")
