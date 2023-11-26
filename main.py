import os
import sys
import logging
from datetime import datetime
import numpy as np
from music21 import instrument, note, chord, stream
from typing import List, Optional, Tuple
from config import Config
from DataLoader import DataLoader
from LSTM import Model

class MusicGenerator:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the music generator with configuration.
        
        Args:
            config_path: Optional path to config file
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        try:
            self.config = Config(config_path)
            self.dataloader = DataLoader(
                self.config.get_property("midi_file_location", ".")
            )
            self.model = Model(
                dropout=self.config.get_property("dropout")
            )
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            raise

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, int, List[str]]:
        """
        Load and prepare the training data.
        
        Returns:
            Tuple containing network input, output, vocabulary size, and pitch names
        """
        try:
            self.logger.info("Loading MIDI files...")
            notes = self.dataloader.load()
            
            self.logger.info("Generating sequences...")
            return self.dataloader.generate_sequence(
                self.config.get_property("sequence_length")
            )
        except Exception as e:
            self.logger.error(f"Data preparation failed: {str(e)}")
            raise

    def train_model(self, network_input: np.ndarray, network_output: np.ndarray,
                   n_vocab: int) -> None:
        """
        Train the model with the prepared data.
        """
        try:
            self.logger.info("Building and training model...")
            self.model.build_model(
                n_vocab=n_vocab,
                sequence_length=self.config.get_property("sequence_length")
            )
            
            self.model.train(
                network_input,
                network_output,
                epochs=self.config.get_property("epochs"),
                batch_size=self.config.get_property("batch_size")
            )
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/model_{timestamp}.h5"
            os.makedirs("models", exist_ok=True)
            self.model.save_model(model_path)
            self.logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def generate_music(self, network_input: np.ndarray, n_vocab: int,
                    pitchnames: List[str], num_notes: int = 500,
                    output_file: str = "output.mid", temperature: float = 1.0) -> None:
        """
        Generate music using the trained model with temperature-based sampling.
        
        Args:
            temperature: Controls randomness. Lower values make the output more deterministic.
        """
        try:
            self.logger.info(f"Generating {num_notes} notes...")
            
            # Create note to integer mapping
            int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
            
            # Generate initial pattern
            start = np.random.randint(0, len(network_input) - 1)
            
            # Extract the initial pattern and flatten it to 1D
            pattern = network_input[start].flatten()
            sequence_length = len(pattern)
            
            prediction_output = []
            
            # Generate notes
            for note_index in range(num_notes):
                # Reshape for prediction
                prediction_input = pattern.reshape((1, sequence_length, 1))
                prediction_input = prediction_input / float(n_vocab)
                
                # Get prediction
                prediction = self.model.predict(prediction_input)[0]
                
                # Apply temperature scaling
                prediction = np.log(prediction) / temperature
                exp_prediction = np.exp(prediction)
                prediction = exp_prediction / np.sum(exp_prediction)
                
                # Sample from the probability distribution
                index = np.random.choice(len(prediction), p=prediction)
                result = int_to_note[index]
                prediction_output.append(result)
                
                # Update pattern
                new_value = np.array([index / float(n_vocab)])
                pattern = np.append(pattern[1:], new_value)
                
                if (note_index + 1) % 50 == 0:
                    self.logger.info(f"Generated {note_index + 1} notes")
            
            self._create_midi(prediction_output, output_file)
            
        except Exception as e:
            self.logger.error(f"Music generation failed: {str(e)}")
            raise

    def _create_midi(self, prediction_output: List[str], output_file: str,
                    note_offset: float = 0.5) -> None:
        """
        Create a MIDI file from the predicted output.
        """
        try:
            self.logger.info(f"Creating MIDI file with {len(prediction_output)} notes")
            offset = 0
            output_notes = []
            
            for pattern in prediction_output:
                if ('.' in str(pattern)) or str(pattern).isdigit():
                    # Handle chord
                    notes_in_chord = str(pattern).split('.')
                    notes = []
                    for current_note in notes_in_chord:
                        try:
                            new_note = note.Note(int(float(current_note)))
                            new_note.storedInstrument = instrument.Piano()
                            notes.append(new_note)
                        except ValueError:
                            self.logger.warning(f"Skipping invalid note in chord: {current_note}")
                            continue
                    
                    if notes:
                        new_chord = chord.Chord(notes)
                        new_chord.offset = offset
                        output_notes.append(new_chord)
                else:
                    # Handle single note
                    try:
                        new_note = note.Note(str(pattern))
                        new_note.offset = offset
                        new_note.storedInstrument = instrument.Piano()
                        output_notes.append(new_note)
                    except ValueError:
                        self.logger.warning(f"Skipping invalid note: {pattern}")
                        continue
                
                offset += note_offset
            
            if not output_notes:
                raise ValueError("No valid notes generated")
            
            midi_stream = stream.Stream(output_notes)
            midi_stream.write('midi', fp=output_file)
            self.logger.info(f"MIDI file saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"MIDI file creation failed: {str(e)}")
            raise

def main():
    """Main execution function."""
    
    try:
        # Initialize generator
        generator = MusicGenerator()
        
        # Prepare data
        network_input, network_output, n_vocab, pitchnames = generator.prepare_data()
        
        print("network_input shape:", network_input.shape)
        print("network_input dtype:", network_input.dtype)
        print("Sample of network_input:", network_input[0][:5])
        # Uncomment to train a new model
        # generator.train_model(network_input, network_output, n_vocab)
        
        # # Load existing model
        generator.model.load_model("models/model_20241126_030613.h5")
        
        # Generate music
        generator.generate_music(
            network_input=network_input,
            n_vocab=n_vocab,
            pitchnames=pitchnames,
            num_notes=500,
            output_file="test_output.mid",
            temperature=1.5
        )
        
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()