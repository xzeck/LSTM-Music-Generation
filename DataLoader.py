from music21 import converter, instrument, note, chord
from glob import glob
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation

class DataLoader:
    def __init__(self, midi_path) -> None:
        self.midi_path = midi_path
        self.notes = []
        self.pitchnames = None
        
    def load(self):
        for file in glob(f"{self.midi_path}/*.mid"):
            midi = converter.parse(file)
            notes_to_parse = None
            
            parts = instrument.partitionByInstrument(midi)
            
            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes
                
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    self.notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    self.notes.append('.'.join(str(n) for n in element.normalOrder))

        self.pitchnames = sorted(set(item for item in self.notes))
        return self.notes
    
    def generate_sequence(self, sequence_length):
        if not self.pitchnames:
            raise ValueError("Please call load() method before generate_sequence()")
            
        n_vocabs = len(self.pitchnames)
        note_to_int = dict((note, number) for number, note in enumerate(self.pitchnames))
        
        network_input = []
        network_output = []
        
        for i in range(0, len(self.notes) - sequence_length):
            sequence_in = self.notes[i: i + sequence_length]
            sequence_out = self.notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
        
        n_patterns = len(network_input)
        
        # Reshape input to be [samples, time steps, features]
        network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
        
        # Normalize input
        network_input = network_input / float(n_vocabs)
        
        # One-hot encode the output
        network_output = to_categorical(network_output, num_classes=n_vocabs)
        
        return network_input, network_output, n_vocabs, self.pitchnames