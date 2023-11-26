from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
from config import Config

config = Config()

class Model:
    def __init__(self, dropout: float = 0.2):
        self.dropout = dropout
        self.model = Sequential()
        self.history = None
        
    def build_model(self, n_vocab: int, sequence_length: int):
        """
        Build the model architecture dynamically from config.
        
        Args:
            n_vocab: Size of vocabulary (number of unique notes)
            sequence_length: Length of input sequences
        """
        # Reset the model to ensure a clean build
        self.model = Sequential()
        
        # Set the input dimension for the first LSTM layer
        first_layer = True
        
        for layer in config.get_property("layers"):
            layer_type = layer.get('type')
            
            if layer_type == 'lstm':
                lstm_args = {
                    'units': layer.get('neurons'),
                    'return_sequences': layer.get('return_seq', False)
                }
                
                # Add recurrent dropout if specified
                if layer.get('recurrent_dropout') is not None:
                    lstm_args['recurrent_dropout'] = layer.get('recurrent_dropout')
                
                if first_layer:
                    # First LSTM layer needs input_shape
                    lstm_args['input_shape'] = (sequence_length, 1)
                    self.model.add(LSTM(**lstm_args))
                    first_layer = False
                else:
                    self.model.add(LSTM(**lstm_args))
            
            elif layer_type == 'dense':
                # For the final layer, use n_vocab as number of neurons
                neurons = layer.get('neurons')
                if neurons == 0:
                    neurons = n_vocab
                
                activation = layer.get('activation', 'linear')
                self.model.add(Dense(
                    neurons, 
                    activation=activation
                ))
            
            elif layer_type == 'dropout':
                self.model.add(Dropout(layer.get('rate')))
            
            elif layer_type == 'batch_normalization':
                self.model.add(BatchNormalization())
            
            elif layer_type == 'activation':
                self.model.add(Activation(layer.get('activation')))
        
        # Ensure final layer is appropriate for classification
        if self.model.layers[-1].__class__.__name__ != 'Dense' or \
        self.model.layers[-1].activation.__name__ != 'softmax':
            self.model.add(Dense(n_vocab, activation='softmax'))
        
        # Compile the model
        optimizer = Adam(learning_rate=config.get_property("learning_rate", 0.001))
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        # Print model summary for debugging
        self.model.summary()
        
        return self.model

    def train(self, 
              x: np.ndarray, 
              y: np.ndarray, 
              epochs: int = 200, 
              batch_size: int = 64,
              model_dir: str = "models") -> None:
        """
        Train the model with validation data and callbacks.
        """
        print("Training started")
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate model checkpoint path
        checkpoint_path = os.path.join(
            model_dir,
            f"checkpoint_{self._get_timestamp()}.keras"
        )
        
        callbacks = [
            EarlyStopping(
                monitor='loss',
                patience=5,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='loss',
                save_best_only=True,
                save_weights_only=False
            ),
            ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=2,
                min_lr=0.000001
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the final model
        model_name = os.path.join(
            model_dir,
            f"final_model_{self._get_timestamp()}.h5"
        )
        self.save_model(model_name)

    def _get_timestamp(self) -> str:
        """Get current timestamp for file naming"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not built")
        
        return self.model.predict(X_test, verbose=0)
    
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to: {filepath}")
        else:
            raise ValueError("Model not built or trained")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        self.model = load_model(filepath)
        print(f"Model loaded from: {filepath}")
        
    def get_training_history(self) -> Optional[Dict[str, Any]]:
        """
        Get the training history if available.
        
        Returns:
            Dictionary containing training metrics history
        """
        return self.history.history if self.history else None