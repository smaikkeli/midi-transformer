import json
from pathlib import Path

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

@dataclass
class TokenizerConfig:
    beat_res: Dict[Tuple[int, int], int]
    additional_params: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TokenizerConfig":
        beat_res_raw = data.pop("beat_res", {})
        beat_res = {
            tuple(map(int, k.split("_"))): v
            for k, v in beat_res_raw.items()
        }
        return TokenizerConfig(beat_res=beat_res, additional_params=data)

@dataclass
class Config:
    data_dir: Path
    tokenized_dir: Path
    max_seq_len: int
    vocab_size: int
    tokenizer_config: TokenizerConfig
    datasets: Dict  # You can make this more specific depending on structure

    @property
    def tokenizer_path(self) -> Path:
        return self.data_dir / "combined_tokenizer.json"

    @staticmethod
    def load_from_file(config_path: Path) -> "Config":
        with open(config_path, 'r') as f:
            raw = json.load(f)

        return Config(
            data_dir=Path(raw["DATA_DIR"]),
            tokenized_dir=Path(raw["TOKENIZED_DIR"]),
            max_seq_len=raw["MAX_SEQ_LEN"],
            vocab_size=raw["VOCAB_SIZE"],
            tokenizer_config=TokenizerConfig.from_dict(raw["TOKENIZER_CONFIG"]),
            datasets=raw["DATASETS"]
        )

    def create_directories(self):
        self.data_dir.mkdir(exist_ok=True)
        self.tokenized_dir.mkdir(exist_ok=True)