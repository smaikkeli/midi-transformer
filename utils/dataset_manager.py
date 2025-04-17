import requests
import shutil
import tarfile
import zipfile

from utils.config.config import Config
from pathlib import Path

class DatasetManager:
    """Manages downloading and extraction of datasets."""
    def __init__(self, config: Config):
        self.datasets = config.datasets
    
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
    
    def get_dataset(self, dataset_name: str, cleanup: bool = True) -> None:
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' is not configured")
        
        dataset = self.datasets[dataset_name]
        extract_path = Path(dataset["extract_path"])
        download_path = Path(dataset["download_path"])
        
        if not extract_path.exists():
            self.download_file(dataset["url"], download_path)
            
            if dataset["file_type"] == "zip":
                self.extract_zip(download_path, extract_path)
            elif dataset["file_type"] == "tar.gz":
                self.extract_tar(download_path, extract_path)
            else:
                raise ValueError(f"Unsupported file type: {dataset['file_type']}")
            
            if cleanup and download_path.exists():
                download_path.unlink()
                print(f"Removed archive file {dataset['download_path']}")
        else:
            print(f"Dataset {dataset_name} already exists at {extract_path}")
    
    def get_all_datasets(self, cleanup: bool = True) -> None:
        for dataset_name in self.datasets:
            self.get_dataset(dataset_name, cleanup)