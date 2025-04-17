from pathlib import Path
from typing import List
from miditok import REMI
from utils.config.config import Config

class TokenizerManager:
    """Manages MIDI tokenization."""

    def __init__(self, config: Config):
        self.config = config

    @classmethod
    def from_config_path(cls, path: Path) -> "TokenizerManager":
        config = Config.load_from_file(path)
        return cls(config)

    def create_tokenizer(self) -> REMI:
        return REMI(tokenizer_config=self.config.tokenizer_config)

    def load_tokenizer(self) -> REMI:
        if not self.config.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {self.config.tokenizer_path}")
        
        return REMI(params=self.config.tokenizer_path)

    def get_or_create_tokenizer(self, midi_paths: List[Path]) -> REMI:
        if self.config.tokenizer_path.exists():
            print(f"Loading existing tokenizer from {self.config.tokenizer_path}")
            return self.load_tokenizer()

        print("Creating and training new tokenizer...")
        tokenizer = self.create_tokenizer()
        tokenizer.train(vocab_size=self.config.vocab_size, files_paths=midi_paths)
        
        tokenizer.save(self.config.tokenizer_path)
        print(f"Tokenizer saved to {self.config.tokenizer_path}")

        return tokenizer