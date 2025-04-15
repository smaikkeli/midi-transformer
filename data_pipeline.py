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
from typing import List, Tuple, Optional, Dict


class Config:
    """Configuration manager for the MIDI tokenization pipeline."""
    
    config = {}  # Default empty config
    
    @classmethod
    def load_config(cls, config_path):
        """Load configuration from a JSON file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        # Parse 'beat_res' to convert the string keys to tuples
        if "TOKENIZER_CONFIG" in config_data and "beat_res" in config_data["TOKENIZER_CONFIG"]:
            beat_res = config_data["TOKENIZER_CONFIG"]["beat_res"]
            parsed_beat_res = {}
            for key, value in beat_res.items():
                # Convert string key like "0_4" to a tuple (0, 4)
                beat_range = tuple(map(int, key.split('_')))
                parsed_beat_res[beat_range] = value
            
            config_data["TOKENIZER_CONFIG"]["beat_res"] = parsed_beat_res
        
        cls.config = config_data
        
        # Set class attributes from config
        cls.DATA_DIR = Path(config_data["DATA_DIR"])
        cls.TOKENIZED_DIR = Path(config_data["TOKENIZED_DIR"])
        cls.MAX_SEQ_LEN = config_data["MAX_SEQ_LEN"]
        cls.VOCAB_SIZE = config_data["VOCAB_SIZE"]
        cls.TOKENIZER_CONFIG = TokenizerConfig(**config_data["TOKENIZER_CONFIG"])
        cls.TOKENIZER_PATH = cls.DATA_DIR / "combined_tokenizer.json"
        cls.DATASETS = config_data["DATASETS"]
        
        return cls.config

    @classmethod
    def create_directories(cls):
        """Create necessary directories for data processing."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.TOKENIZED_DIR.mkdir(exist_ok=True)


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
    
    @staticmethod
    def create_tokenizer() -> REMI:
        return REMI(tokenizer_config=Config.TOKENIZER_CONFIG)
    
    @staticmethod
    def load_tokenizer(config_path: str) -> REMI:
        #Load the data_config, not the tokenizer itself
        Config.load_config(config_path)

        if not Config.TOKENIZER_PATH.exists():
            raise FileNotFoundError(f"Tokenizer not found at {Config.TOKENIZER_PATH}")
        
        return REMI(params=Config.TOKENIZER_PATH)
    
    @classmethod
    def get_or_create_tokenizer(cls, midi_paths: List[Path]) -> REMI:
        if Config.TOKENIZER_PATH.exists():
            print(f"Loading existing tokenizer from {Config.TOKENIZER_PATH}")
            return cls.load_tokenizer()
        
        print("Creating and training new tokenizer...")
        tokenizer = cls.create_tokenizer()
        tokenizer.train(vocab_size=Config.VOCAB_SIZE, files_paths=midi_paths)
        
        tokenizer.save(Config.TOKENIZER_PATH)
        print(f"Tokenizer saved to {Config.TOKENIZER_PATH}")
        
        return tokenizer


class DatasetPreprocessor:
    """Prepare datasets for training."""
    
    @staticmethod
    def tokenize_and_chunk_dataset(midi_paths: List[Path], tokenizer: REMI) -> None:
        if Config.TOKENIZED_DIR.exists() and list(Config.TOKENIZED_DIR.glob("**/*.midi")):
            print(f"Tokenized chunks already exist in {Config.TOKENIZED_DIR}")
            return
        
        print(f"Tokenizing and splitting {len(midi_paths)} files into chunks...")
        split_files_for_training(
            files_paths=midi_paths,
            tokenizer=tokenizer,
            save_dir=Config.TOKENIZED_DIR,
            max_seq_len=Config.MAX_SEQ_LEN,
        )
        print(f"Tokenization complete. Files saved to {Config.TOKENIZED_DIR}")

    @staticmethod
    def load_tokenized_dataset(tokenizer: REMI) -> DatasetMIDI:
        files = list(Config.TOKENIZED_DIR.glob("**/*.midi"))
        print(f"Loading {len(files)} MIDI chunks...")
        
        dataset = DatasetMIDI(
            files_paths=files,
            tokenizer=tokenizer,
            max_seq_len=Config.MAX_SEQ_LEN,
            bos_token_id=tokenizer["BOS_None"],
            eos_token_id=tokenizer["EOS_None"],
        )
        
        return dataset

    @staticmethod
    def create_data_loaders(dataset: DatasetMIDI,
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
