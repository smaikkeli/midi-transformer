**Midi Transformer**

*Data Representation*

Event-based representation

a single midi event consists of:

Note-on: (0-127)
Note offset: Note duration (32th note multiples; 1-64)
Timegrid: Position (16bins; 1-16)
Tempo changes: Tempo (30-209BPM)
Chord: Chord (60 types)

Tokenization:
Example token:

Note on C4
Note off C4
Time Shift 10ms
COntrol Change sustain pedal on

Vocabulary consists of all possible notes
Time shift quantized
COntrol changes and their possible values??
