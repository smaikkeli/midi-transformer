import json
from pathlib import Path

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime


@dataclass
class Config:
    outputs_dir: Path
    data_dir: Path
    chunked_dir: Path
    tokenizer_dir: Path
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
            outputs_dir=Path(raw["OUTPUTS_DIR"]),
            data_dir=Path(raw["DATA_DIR"]),
            chunked_dir=Path(raw["CHUNKED_DIR"]),
            tokenizer_dir=Path(raw["TOKENIZER_DIR"]),
            max_seq_len=raw["MAX_SEQ_LEN"],
            min_seq_len=raw["MIN_SEQ_LEN"],
            vocab_size=raw["VOCAB_SIZE"],
            tokenizer_config=raw["TOKENIZER_CONFIG"],
            datasets=raw["DATASETS"]
        )

    def create_directories(self):
        self.data_dir.mkdir(parents=True,exist_ok=True)