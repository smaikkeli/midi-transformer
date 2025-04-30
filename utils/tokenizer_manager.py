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

    def load_tokenizer(self, path) -> MusicTokenizer:
        """Loads an existing tokenizer from the specified path."""
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {path}")
        
        return REMI(params=path)

    def get_or_create_tokenizer(self, midi_paths: List[Path]) -> MusicTokenizer:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        parts = self.config.chunked_dir.parts
        folder_name = timestamp +"_" + parts[0]
        self.config.chunked_dir = Path(folder_name, *parts[1:])
        self.config.chunked_dir = self.config.outputs_dir/"tokenizers"/self.config.chunked_dir

        parts = self.config.tokenizer_dir.parts
        folder_name = timestamp +"_" + parts[0]
        self.config.tokenizer_dir = Path(folder_name, *parts[1:])
        self.config.tokenizer_dir = self.config.outputs_dir/"tokenizers"/self.config.tokenizer_dir


        if (self.config.tokenizer_dir/"trained_tokenizer.json").exists():
            print(f"Loading existing tokenizer from {self.config.tokenizer_dir}")
            return self.load_tokenizer(self.config.tokenizer_dir/"trained_tokenizer.json")


        print("Creating and training new tokenizer...")
        tokenizer = self.create_tokenizer()
        tokenizer.train(vocab_size=self.config.vocab_size, files_paths=midi_paths)
        

        print(self.config.chunked_dir)
        self.config.chunked_dir.mkdir(parents=True,exist_ok=True)   
        tokenizer.save(self.config.tokenizer_dir/"trained_tokenizer.json")
        print(f"Tokenizer saved to {self.config.tokenizer_dir}")

        return tokenizer