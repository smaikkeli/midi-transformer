import argparse

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
    
    def prepare_tokenized_data(self, lakh_limit=None):
        """Prepare tokenized data from all datasets."""
        midi_paths = self.midi_processor.get_all_midi_paths(lakh_limit=lakh_limit)

        print(f"Found {len(midi_paths)} MIDI files across all datasets")
        
        tokenizer = self.tokenizer_manager.get_or_create_tokenizer(midi_paths)
        
        self.dataset_preprocessor.tokenize_and_chunk_dataset(midi_paths, tokenizer)
        
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
    config = Config.load_from_file(args.config)
    pipeline = Pipeline(config)
    pipeline.setup_environment()

    # Download datasets if --download is provided
    if args.download:
        pipeline.prepare_datasets(cleanup=True)
        print("All datasets have been downloaded and extracted.")

    # Tokenize datasets if --tokenize is provided
    if args.tokenize:
        pipeline.prepare_tokenized_data(lakh_limit=args.lakh_limit)
        print(f"Tokenizer created and saved to {config.tokenizer_path}")

    # Handle case where neither argument is provided
    if not args.download and not args.tokenize:
        print("No action specified. Use --download to download datasets or --tokenize to tokenize them.")


if __name__ == "__main__":
    main()
