from pathlib import Path
from typing import List, Tuple
from miditok import REMI
from miditok.utils import split_files_for_training
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader, random_split

from utils.config.config import Config


class DatasetPreprocessor:
    """Prepare datasets for training."""

    def __init__(self, config: Config):
        self.tokenized_dir = config.tokenized_dir
        self.max_seq_len = config.max_seq_len
        self.vocab_size = config.vocab_size


    def tokenize_and_chunk_dataset(self, midi_paths: List[Path], tokenizer: REMI) -> None:
        if self.tokenized_dir.exists() and list(self.tokenized_dir.glob("**/*.midi")):
            print(f"Tokenized chunks already exist in {self.tokenized_dir}")
            return
        
        print(f"Tokenizing and splitting {len(midi_paths)} files into chunks...")
        split_files_for_training(
            files_paths=midi_paths,
            tokenizer=tokenizer,
            save_dir=self.tokenized_dir,
            max_seq_len=self.max_seq_len,
        )
        print(f"Tokenization complete. Files saved to {self.tokenized_dir}")

    def load_tokenized_dataset(self, tokenizer: REMI) -> DatasetMIDI:
        files = list(self.tokenized_dir.glob("**/*.midi"))
        print(f"Loading {len(files)} MIDI chunks...")
        
        dataset = DatasetMIDI(
            files_paths=files,
            tokenizer=tokenizer,
            max_seq_len=self.max_seq_len,
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