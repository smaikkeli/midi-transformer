import argparse
import json
import os
import zipfile
import tarfile
import shutil
import requests
import pandas as pd
from pathlib import Path

from miditok import REMI, TokenizerConfig
from miditok.utils import split_files_for_training

from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader, random_split
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
    datasets: Any  # You can make this more specific depending on structure

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

class DatasetManager:
    """Manages downloading and extraction of datasets."""
    
    @staticmethod
    def download_file(url: str, dest: Path) -> None:
        if not dest.exists():
            print(f"Downloading {url} to {dest}...")
            response = requests.get(url, stream=True)
            with open(dest, "wb") as f:
                shutil.copyfileobj(response.raw, f)
            print("Download complete.")
        else:
            print(f"{dest} already exists. Skipping download.")
    
    @staticmethod
    def extract_zip(zip_path: Path, extract_to: Path) -> None:
        print(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
    
    @staticmethod
    def extract_tar(tar_path: Path, extract_to: Path) -> None:
        print(f"Extracting {tar_path} to {extract_to}...")
        with tarfile.open(tar_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
        print("Extraction complete.")
    
    @classmethod
    def get_dataset(cls, dataset_name: str, cleanup: bool = True) -> None:
        if dataset_name not in Config.DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' is not configured")
        
        dataset = Config.DATASETS[dataset_name]
        extract_path = Path(dataset["extract_path"])
        download_path = Path(dataset["download_path"])
        
        if not extract_path.exists():
            cls.download_file(dataset["url"], download_path)
            
            if dataset["file_type"] == "zip":
                cls.extract_zip(download_path, extract_path)
            elif dataset["file_type"] == "tar.gz":
                cls.extract_tar(download_path, extract_path)
            else:
                raise ValueError(f"Unsupported file type: {dataset['file_type']}")
            
            if cleanup and download_path.exists():
                download_path.unlink()
                print(f"Removed archive file {dataset['download_path']}")
        else:
            print(f"Dataset {dataset_name} already exists at {extract_path}")
    
    @classmethod
    def get_all_datasets(cls, cleanup: bool = True) -> None:
        for dataset_name in Config.DATASETS:
            cls.get_dataset(dataset_name, cleanup)


class MIDIProcessor:
    """Processing MIDI files from various datasets."""
    
    @staticmethod
    def get_maestro_midi_paths() -> List[Path]:
        maestro_dir = (Config.DATA_DIR / "maestro" / "maestro-v3.0.0").resolve()
        metadata_path = maestro_dir / "maestro-v3.0.0.csv"
        metadata = pd.read_csv(metadata_path)
        return [maestro_dir / x for x in metadata["midi_filename"]]
    
    @staticmethod
    def get_lakh_midi_paths(limit: Optional[int] = None) -> List[Path]:
        lakh_root = (Config.DATA_DIR / "lakh" / "lmd_full").resolve()
        midi_paths = list(lakh_root.rglob("*.mid"))
        return midi_paths[:limit] if limit else midi_paths
    
    @classmethod
    def get_all_midi_paths(cls, lakh_limit: Optional[int] = 2000) -> List[Path]:
        paths = []
        
        try:
            paths.extend(cls.get_maestro_midi_paths())
        except Exception as e:
            print(f"Error getting MAESTRO paths: {e}")
        
        try:
            paths.extend(cls.get_lakh_midi_paths(limit=lakh_limit))
        except Exception as e:
            print(f"Error getting Lakh paths: {e}")
        
        print(f"Found {len(paths)} MIDI files across all datasets")
        return paths


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

class DatasetPreprocessor:
    """Prepare datasets for training."""

    def __init__(self, config: Config):
        self.config = config

    def tokenize_and_chunk_dataset(self, midi_paths: List[Path], tokenizer: REMI) -> None:
        if self.config.tokenized_dir.exists() and list(self.config.tokenized_dir.glob("**/*.midi")):
            print(f"Tokenized chunks already exist in {self.config.tokenized_dir}")
            return
        
        print(f"Tokenizing and splitting {len(midi_paths)} files into chunks...")
        split_files_for_training(
            files_paths=midi_paths,
            tokenizer=tokenizer,
            save_dir=self.config.tokenized_dir,
            max_seq_len=self.config.max_seq_len,
        )
        print(f"Tokenization complete. Files saved to {self.config.tokenized_dir}")

    def load_tokenized_dataset(self, tokenizer: REMI) -> DatasetMIDI:
        files = list(self.config.tokenized_dir.glob("**/*.midi"))
        print(f"Loading {len(files)} MIDI chunks...")
        
        dataset = DatasetMIDI(
            files_paths=files,
            tokenizer=tokenizer,
            max_seq_len=self.config.max_seq_len,
            bos_token_id=tokenizer["BOS_None"],
            eos_token_id=tokenizer["EOS_None"],
        )
        
        return dataset

    def create_data_loaders(
        self,
        dataset: DatasetMIDI,
        tokenizer: REMI,
        batch_size: int = 1,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:

        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        collator = DataCollator(
            pad_token_id=tokenizer.pad_token_id,
            copy_inputs_as_labels=True,
            shift_labels=False
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collator)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator)

        return train_loader, val_loader, test_loader
    
class Pipeline:
    """Main pipeline for dataset preparation."""
    
    @staticmethod
    def setup_environment(config_path):
        """Set up environment variables and directories."""
        Config.load_config(config_path)
        Config.create_directories()
    
    @classmethod
    def prepare_datasets(cls, datasets=None, cleanup=True):
        """Download and extract specified datasets."""
        if datasets is None:
            DatasetManager.get_all_datasets(cleanup=cleanup)
        else:
            for dataset_name in datasets:
                DatasetManager.get_dataset(dataset_name, cleanup=cleanup)
    
    @classmethod
    def prepare_tokenized_data(cls, lakh_limit=2000):
        """Prepare tokenized data from all datasets."""
        midi_paths = MIDIProcessor.get_all_midi_paths(lakh_limit=lakh_limit)
        
        tokenizer = TokenizerManager.get_or_create_tokenizer(midi_paths)
        
        DatasetPreprocessor.tokenize_and_chunk_dataset(midi_paths, tokenizer)
        
        return tokenizer
    
    @classmethod
    def setup_env_and_prepare_data(cls, config_path, datasets=None, cleanup=True):
        cls.setup_environment(config_path)
        cls.prepare_datasets(datasets, cleanup=cleanup)

    @classmethod
    def tokenize_and_chunk(cls, lakh_limit = 2000):
        cls.prepare_tokenized_data(lakh_limit=lakh_limit)
    
    @classmethod
    def run_full_pipeline(cls, config_path, datasets=None, lakh_limit=2000, cleanup=True):
        """Run the full pipeline from download to data loader creation."""
        cls.setup_env_and_prepare_data(config_path, datasets, cleanup=cleanup)
        
        return cls.tokenize_and_chunk(lakh_limit=lakh_limit)

    
def main():
    argparser = argparse.ArgumentParser(description="MIDI Dataset Preparation Pipeline")
    
    argparser.add_argument('--config', type=str, default="token_config.json", help="Path to configuration file")
    argparser.add_argument('--download', action='store_true', help="Download all datasets specified in the config file")
    argparser.add_argument('--tokenize', action='store_true', help="Tokenize and chunk the dataset after download")
    argparser.add_argument('--lakh_limit', type=int, default=None, help="Limit the number of Lakh MIDI files to process")

    args = argparser.parse_args()

    # Setup environment with specified config
    Pipeline.setup_environment(args.config)

    # Download datasets if --download is provided
    if args.download:
        Pipeline.prepare_datasets(cleanup=True)
        print("All datasets have been downloaded and extracted.")

    # Tokenize datasets if --tokenize is provided
    if args.tokenize:
        Pipeline.prepare_tokenized_data(lakh_limit=args.lakh_limit)
        print(f"Tokenizer created and saved to {Config.TOKENIZER_PATH}")

    # Handle case where neither argument is provided
    if not args.download and not args.tokenize:
        print("No action specified. Use --download to download datasets or --tokenize to tokenize them.")


if __name__ == "__main__":
    main()
