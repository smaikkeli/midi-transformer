import argparse

from pathlib import Path


from utils.config.config import Config
from utils.dataset_manager import DatasetManager
from utils.dataset_preprocessor import DatasetPreprocessor
from utils.midi_processor import MIDIProcessor
from utils.tokenizer_manager import TokenizerManager




class Pipeline:
    """Main pipeline for dataset preparation."""
    def __init__(self, config: Config):
        self.config = config

        self.dataset_preprocessor = DatasetPreprocessor(config)
        self.midi_processor = MIDIProcessor(config)
        self.dataset_manager = DatasetManager(config)
        self.tokenizer_manager = TokenizerManager(config)
    
    def setup_environment(self):
        """Set up environment variables and directories."""
        self.config.create_directories()
    
    def prepare_datasets(self, datasets=None, cleanup=True):
        """Download and extract specified datasets."""
        if datasets is None:
            self.dataset_manager.get_all_datasets(cleanup=cleanup)
        else:
            for dataset_name in datasets:
                self.dataset_manager.get_dataset(dataset_name, cleanup=cleanup)
    
    def chunk(self, lakh_limit=None):
        """Prepare tokenized data from all datasets."""
        midi_paths = self.midi_processor.get_all_midi_paths(lakh_limit=lakh_limit)

        print(f"Found {len(midi_paths)} MIDI files across all datasets")

        tokenizer = self.tokenizer_manager.get_or_create_tokenizer(midi_paths)

        chunk_paths = self.dataset_preprocessor.chunk_dataset(midi_paths, tokenizer)

        return chunk_paths
    
    def remove_chunks(self):
        "Removes the current chunked files"
        self.dataset_manager.remove_dir(self.config.chunked_dir)
        print(f"Chunked files removed from {self.config.chunked_dir}")

    
def main():
    argparser = argparse.ArgumentParser(description="MIDI Dataset Preparation Pipeline")
    
    # Common configuration
    argparser.add_argument('--config', type=str, default="data_config.json", help="Path to configuration file")
    
    # Create mutually exclusive group for primary actions
    action_group = argparser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--download', action='store_true', help="Download all datasets specified in the config file")
    action_group.add_argument('--chunk', action='store_true', help="Chunk the dataset after download")
    action_group.add_argument('--remove_chunks', action='store_true', help="Remove chunked files")
    
    # Optional parameters for specific actions
    chunk_options = argparser.add_argument_group('Chunking options')
    chunk_options.add_argument('--lakh_limit', type=int, default=None, help="Limit the number of Lakh MIDI files to process")

    args = argparser.parse_args()

    # Setup environment with specified config
    config = Config.load_from_file(args.config)
    pipeline = Pipeline(config)
    pipeline.setup_environment()

    # Execute primary action based on arguments
    if args.remove_chunks:
        pipeline.remove_chunks()
        print("Chunked files have been removed.")
    
    elif args.download:
        pipeline.prepare_datasets(cleanup=True)
        print("All datasets have been downloaded and extracted.")
    
    elif args.chunk:
        pipeline.chunk(lakh_limit=args.lakh_limit)
        print(f"Dataset has been chunked successfully.")


if __name__ == "__main__":
    main()