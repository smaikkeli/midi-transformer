from pathlib import Path
from typing import List
from miditok import REMI, MusicTokenizer, TokenizerConfig
from utils.config.config import Config
from datetime import datetime

class TokenizerManager:
    """Manages MIDI tokenization."""

    def __init__(self, config: Config):
        self.config = config

    @classmethod
    def from_config_path(cls, path: Path) -> "TokenizerManager":
        """Create a TokenizerManager from a config file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at {path}")
        
        config = Config.load_from_file(path)
        return cls(config)

    def create_tokenizer(self) -> MusicTokenizer:
        """Creates a new tokenizer based on the configuration."""
        t_config = self.config.tokenizer_config

        #Conver beat resolution from string to tuple
        beat_res_raw = t_config.pop("beat_res", {})
        beat_res = {
            tuple(map(int, k.split("_"))): v
            for k, v in beat_res_raw.items()
        }
        t_config["beat_res"] = beat_res

        t_config = TokenizerConfig(**t_config)
        return REMI(tokenizer_config=t_config)

    def load_tokenizer(self) -> MusicTokenizer:
        """Loads an existing tokenizer from the specified path."""
        if not self.config.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {self.config.tokenizer_path}")
        
        return REMI(params=self.config.tokenizer_path)

    def get_or_create_tokenizer(self, midi_paths: List[Path]) -> MusicTokenizer:
        if self.config.tokenizer_path.exists():
            print(f"Loading existing tokenizer from {self.config.tokenizer_path}")
            return self.load_tokenizer()


        print("Creating and training new tokenizer...")
        tokenizer = self.create_tokenizer()
        tokenizer.train(vocab_size=self.config.vocab_size, files_paths=midi_paths)


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = self.config.tokenizer_path.parts
        self.config.tokenizer_path = Path(timestamp + "_"+ parts[0] + "/"+parts[1])
        
        tokenizer.save(self.config.tokenizer_path)
        print(f"Tokenizer saved to {self.config.tokenizer_path}")

        return tokenizer