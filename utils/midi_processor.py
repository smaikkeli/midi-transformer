import pandas as pd
from pathlib import Path

from typing import List, Tuple, Optional, Dict, Any
from utils.config.config import Config

class MIDIProcessor:
    """Processing MIDI files from various datasets."""
    def __init__(self, config: Config):
        self.data_dir = config.data_dir
    

    def get_maestro_midi_paths(self) -> List[Path]:
        maestro_dir = (self.data_dir / "maestro" / "maestro-v3.0.0").resolve()
        metadata_path = maestro_dir / "maestro-v3.0.0.csv"
        metadata = pd.read_csv(metadata_path)
        return [maestro_dir / x for x in metadata["midi_filename"]]
    

    def get_lakh_midi_paths(self, limit: Optional[int] = None) -> List[Path]:
        lakh_root = (self.data_dir / "lakh" / "lmd_full").resolve()
        midi_paths = list(lakh_root.rglob("*.mid"))
        return midi_paths[:limit] if limit else midi_paths
    
    def get_all_midi_paths(self, lakh_limit: Optional[int] = 2000) -> List[Path]:
        paths = []
        try:
            paths.extend(self.get_maestro_midi_paths())
        except Exception as e:
            print(f"Error getting MAESTRO paths: {e}")
        
        try:
            paths.extend(self.get_lakh_midi_paths(limit=lakh_limit))
        except Exception as e:
            print(f"Error getting Lakh paths: {e}")
        
        print(f"Found {len(paths)} MIDI files across all datasets")
        return paths