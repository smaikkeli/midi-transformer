from pathlib import Path
from typing import List, Tuple
from miditok import MusicTokenizer
from miditok.utils import split_files_for_training
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader, random_split

from utils.config.config import Config


class DatasetPreprocessor:
    """Prepare datasets for training."""

    def __init__(self, config: Config):
        self.chunked_dir = config.chunked_dir
        self.max_seq_len = config.max_seq_len
        self.min_seq_len = config.min_seq_len
        self.vocab_size = config.vocab_size

    def chunk_dataset(self, midi_paths: List[Path], tokenizer: MusicTokenizer) -> None:
        '''
        Split MIDI files into chunks
        Args:
            midi_paths (List[Path]): List of MIDI file paths to tokenize and chunk.
            tokenizer : Tokenizer to use for tokenization.
        '''

        if (self.chunked_dir.exists()) and (list(self.chunked_dir.rglob("*.mid*"))):
            print(f"Chunked files already exist in {self.chunked_dir} and length is {len(list(self.chunked_dir.glob('**/*.mid*')))}")
            return
        
        print(f"Splitting {len(midi_paths)} files into chunks...")
        chunk_paths = split_files_for_training(
            files_paths=midi_paths,
            tokenizer=tokenizer,
            save_dir=self.chunked_dir,
            max_seq_len=self.max_seq_len,
            min_seq_len=self.min_seq_len
        )
        print(f"Splitting to chunks completed. Files saved to {self.chunked_dir}")
        print(f"In total {len(chunk_paths)} chunks created.")
        return chunk_paths

    def stream_dataset(self, midi_paths: List[Path], tokenizer: MusicTokenizer) -> None:
        '''
        Tokenize MIDI files into a single stream
        Args:
            midi_paths (List[Path]): List of MIDI file paths to tokenize.
            tokenizer : Tokenizer to use for tokenization.
        '''


    def load_chunked_dataset(self, tokenizer: MusicTokenizer) -> DatasetMIDI:
        chunked_files = list(self.chunked_dir.glob("**/*.midi")) + list(self.chunked_dir.glob("**/*.mid"))
        print(f"Loading {len(chunked_files)} MIDI chunks...")
        
        dataset = DatasetMIDI(
            files_paths=chunked_files,
            tokenizer=tokenizer,
            max_seq_len=self.max_seq_len,
            bos_token_id=tokenizer["BOS_None"],
            eos_token_id=tokenizer["EOS_None"],
        )
        
        return dataset

    def create_data_loaders(
        self,
        dataset: DatasetMIDI,
        tokenizer: MusicTokenizer,
        batch_size: int = 1,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
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