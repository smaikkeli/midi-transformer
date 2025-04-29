import json
from pathlib import Path

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class Config:
    data_dir: Path
    chunked_dir: Path
    tokenizer_path: Path
    max_seq_len: int
    min_seq_len: int
    vocab_size: int
    tokenizer_config: Dict
    datasets: Dict


    @staticmethod
    def load_from_file(config_path: Path) -> "Config":
        with open(config_path, 'r') as f:
            raw = json.load(f)
            

        return Config(
            data_dir=Path(raw["DATA_DIR"]),
            chunked_dir=Path(raw["CHUNKED_DIR"]),
            tokenizer_path=Path(raw["TOKENIZER_PATH"]),
            max_seq_len=raw["MAX_SEQ_LEN"],
            min_seq_len=raw["MIN_SEQ_LEN"],
            vocab_size=raw["VOCAB_SIZE"],
            tokenizer_config=raw["TOKENIZER_CONFIG"],
            datasets=raw["DATASETS"]
        )

    def create_directories(self):
        self.data_dir.mkdir(exist_ok=True)
        self.chunked_dir.mkdir(exist_ok=True)