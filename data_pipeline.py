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

# Load config file
def load_config(config_file: str):
    with open(config_file, 'r') as f:
        return json.load(f)

# Global configuration settings for the pipeline (loaded from config file)
config = load_config('config.json')

class Config:
    DATA_DIR = Path(config["DATA_DIR"])
    TOKENIZED_DIR = Path(config["TOKENIZED_DIR"])
    MAX_SEQ_LEN = config["MAX_SEQ_LEN"]
    VOCAB_SIZE = config["VOCAB_SIZE"]
    TOKENIZER_CONFIG = TokenizerConfig(**config["TOKENIZER_CONFIG"])
    TOKENIZER_PATH = DATA_DIR / "combined_tokenizer.json"
    DATASETS = config["DATASETS"]

    @classmethod
    def create_directories(cls):
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.TOKENIZED_DIR.mkdir(exist_ok=True)


class DatasetManager:
    """Manages downloading and extraction of datasets."""
    
    @staticmethod
    def download_file(url: str, dest: Path) -> None:
        """Download a file from URL to destination path."""
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
        """Extract a ZIP archive."""
        print(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
    
    @staticmethod
    def extract_tar(tar_path: Path, extract_to: Path) -> None:
        """Extract a TAR.GZ archive."""
        print(f"Extracting {tar_path} to {extract_to}...")
        with tarfile.open(tar_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
        print("Extraction complete.")
    
    @classmethod
    def get_dataset(cls, dataset_name: str, cleanup: bool = True) -> None:
        """Download and extract a specific dataset."""
        if dataset_name not in Config.DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' is not configured")
        
        dataset = Config.DATASETS[dataset_name]
        extract_path = dataset["extract_path"]
        download_path = dataset["download_path"]
        
        if not extract_path.exists():
            # Download file
            cls.download_file(dataset["url"], download_path)
            
            # Extract based on file type
            if dataset["file_type"] == "zip":
                cls.extract_zip(download_path, extract_path)
            elif dataset["file_type"] == "tar.gz":
                cls.extract_tar(download_path, extract_path)
            else:
                raise ValueError(f"Unsupported file type: {dataset['file_type']}")
            
            # Clean up if requested
            if cleanup and download_path.exists():
                download_path.unlink()
                print(f"Removed archive file {dataset['download_path']}")
        else:
            print(f"Dataset {dataset_name} already exists at {extract_path}")
    
    @classmethod
    def get_all_datasets(cls, cleanup: bool = True) -> None:
        """Download and extract all configured datasets."""
        for dataset_name in Config.DATASETS:
            cls.get_dataset(dataset_name, cleanup)


class MIDIProcessor:
    """Processing MIDI files from various datasets."""
    
    @staticmethod
    def get_maestro_midi_paths() -> List[Path]:
        """Get paths to MIDI files in the MAESTRO dataset."""
        maestro_dir = (Config.DATA_DIR / "maestro" / "maestro-v3.0.0").resolve()
        metadata_path = maestro_dir / "maestro-v3.0.0.csv"
        metadata = pd.read_csv(metadata_path)
        return [maestro_dir / x for x in metadata["midi_filename"]]
    
    @staticmethod
    def get_lakh_midi_paths(limit: Optional[int] = None) -> List[Path]:
        """Get paths to MIDI files in the Lakh dataset with optional limit."""
        lakh_root = (Config.DATA_DIR / "lakh" / "lmd_full").resolve()
        midi_paths = list(lakh_root.rglob("*.mid"))
        return midi_paths[:limit] if limit else midi_paths
    
    @classmethod
    def get_all_midi_paths(cls, lakh_limit: Optional[int] = None) -> List[Path]:
        """Get all MIDI paths from configured datasets."""
        paths = []
        
        # Add MAESTRO paths
        try:
            paths.extend(cls.get_maestro_midi_paths())
        except Exception as e:
            print(f"Error getting MAESTRO paths: {e}")
        
        # Add Lakh paths
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
        """Create a new tokenizer with the configured settings."""
        return REMI(tokenizer_config=Config.TOKENIZER_CONFIG)
    
    @staticmethod
    def load_tokenizer() -> REMI:
        """Load an existing tokenizer from disk."""
        if not Config.TOKENIZER_PATH.exists():
            raise FileNotFoundError(f"Tokenizer not found at {Config.TOKENIZER_PATH}")
        
        return REMI(params=Config.TOKENIZER_PATH)
    
    @classmethod
    def get_or_create_tokenizer(cls, midi_paths: List[Path]) -> REMI:
        """Get existing tokenizer or create and train a new one."""
        if Config.TOKENIZER_PATH.exists():
            print(f"Loading existing tokenizer from {Config.TOKENIZER_PATH}")
            return cls.load_tokenizer()
        
        print("Creating and training new tokenizer...")
        tokenizer = cls.create_tokenizer()
        tokenizer.train(vocab_size=Config.VOCAB_SIZE, files_paths=midi_paths)
        
        # Save the tokenizer
        tokenizer.save(Config.TOKENIZER_PATH)
        print(f"Tokenizer saved to {Config.TOKENIZER_PATH}")
        
        return tokenizer


class DatasetPreprocessor:
    """Prepare datasets for training."""
    
    @staticmethod
    def tokenize_and_chunk_dataset(midi_paths: List[Path], tokenizer: REMI) -> None:
        """Tokenize MIDI files and split them into chunks."""
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
        """Load the tokenized dataset."""
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
    def create_data_loaders(tokenizer: REMI, batch_size: int = 1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load tokenized dataset and create data loaders."""
        files = list(Config.TOKENIZED_DIR.glob("**/*.midi"))
        dataset = DatasetMIDI(
            files_paths=files,
            tokenizer=tokenizer,
            max_seq_len=Config.MAX_SEQ_LEN,
            bos_token_id=tokenizer["BOS_None"],
            eos_token_id=tokenizer["EOS_None"]
        )

        total_size = len(dataset)
        train_size = int(total_size * 0.8)
        val_size = int(total_size * 0.1)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        collator = DataCollator(pad_token_id=tokenizer.pad_token_id)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collator)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator)

        return train_loader, val_loader, test_loader
    
class Pipeline:
    """Main pipeline for dataset preparation."""
    
    @staticmethod
    def setup_environment():
        """Set up environment variables and directories."""
        Config.create_directories()
    
    @classmethod
    def prepare_datasets(cls, datasets=None, lakh_limit=None, cleanup=True):
        """Download and extract specified datasets."""
        if datasets is None:
            # Default to all datasets
            DatasetManager.get_all_datasets(cleanup=cleanup)
        else:
            for dataset_name in datasets:
                DatasetManager.get_dataset(dataset_name, cleanup=cleanup)
    
    @classmethod
    def prepare_tokenized_data(cls, lakh_limit=None):
        """Prepare tokenized data from all datasets."""
        midi_paths = MIDIProcessor.get_all_midi_paths(lakh_limit=lakh_limit)
        
        tokenizer = TokenizerManager.get_or_create_tokenizer(midi_paths)
        
        DatasetPreprocessor.tokenize_and_chunk_dataset(midi_paths, tokenizer)
        
        return tokenizer
    
    
    @classmethod
    def setup_env_and_prepare_data(cls, datasets=None, cleanup=True):
        cls.setup_environment()
        cls.prepare_datasets(datasets, cleanup=cleanup)

    @classmethod
    def tokenize_and_chunk(cls, lakh_limit: Optional[int] = 2000):
        cls.prepare_tokenized_data(lakh_limit=lakh_limit)

    
    @classmethod
    def run_full_pipeline(cls, datasets=None, lakh_limit=2000, cleanup=True):
        """Run the full pipeline from download to data loader creation."""
        cls.setup_env_and_prepare_data(datasets, cleanup=cleanup)
        
        return cls.tokenize_and_chunk(lakh_limit = lakh_limit)

    
def main():
    argparser = argparse.ArgumentParser(description="MIDI Dataset Preparation Pipeline")

    # Argument to control dataset download
    argparser.add_argument('--download', action='store_true', help="Download all datasets specified in the config file.")
    
    # Tokenize and chunk the dataset (optional flag)
    argparser.add_argument('--tokenize', action='store_true', help="Tokenize and chunk the dataset after download.")

    args = argparser.parse_args()

    # Download datasets if --download is provided
    if args.download:
        # Automatically use all datasets from the config
        Pipeline.prepare_datasets(cleanup=True)
        print("All datasets have been downloaded and extracted.")

    # Tokenize datasets if --tokenize is provided
    if args.tokenize:
        Pipeline.prepare_tokenized_data(lakh_limit=None)
        print(f"Tokenizer created and saved to {Config.TOKENIZER_PATH}")

    # Handle case where neither argument is provided
    if not args.download and not args.tokenize:
        print("No action specified. Use --download to download datasets or --tokenize to tokenize them.")



# Example usage
if __name__ == "__main__":
    # Run full pipeline
    main()
    
