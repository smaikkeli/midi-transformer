import pretty_midi

def note_name_from_midi_number(midi_number):
    """
    Given a MIDI pitch number, return its corresponding note name using pretty_midi.
    
    Args:
    midi_number (int): MIDI pitch number (0-127).
    
    Returns:
    str: Note name (e.g., "C4").
    """
    return pretty_midi.note_number_to_name(midi_number)

def generate_midi_vocab():
    """
    Generates a dictionary mapping MIDI pitch numbers to their corresponding note names.
    
    Returns:
    dict: Mapping of MIDI pitch numbers to note names.
    """
    midi_vocab = {}
    for midi_number in range(128):
        note_name = note_name_from_midi_number(midi_number)
        midi_vocab[midi_number] = note_name
    return midi_vocab

# Generate the MIDI pitch vocabulary
midi_vocab = generate_midi_vocab()

# Print the vocabulary
for midi_number, note_name in midi_vocab.items():
    print(f'{midi_number}: {note_name}')