import random
from math import pi 

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import load_from_disk, load_dataset, concatenate_datasets

def filter_fn(example):

    if example["wave"] is None or example["wave"].shape != (2, 2097152):
        return False
    
    if example["info"] is None:
        return False
    
    if example["info"]["title"][0] is None and example["info"]["artist"][0] is None and example["info"]["album"][0] is None and example["info"]["genre"][0] is None and example["info"]["year"][0] is None and example["info"]["crop_id"] is None and example["info"]["num_crops"] is None:
        return False
    
    return True

class Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_train_path: str,
        dataset_valid_path: str,
        batch_size: int,
        *,
        num_workers: int,
        **kwargs: int,
    ) -> None:
        super().__init__()
        data_files = [f"../../classical_music_data_{i}/train_data.json" for i in range(1, 15)]
        data_files.append(f"../../classical_music_data/train_data.json")
        self.dataset_train = load_dataset("json", data_files=data_files, num_proc=32).with_format("torch")["train"]
        data_files = [f"../../classical_music_data_{i}/train_data.json" for i in range(15, 31)]
        dataset_train_2 = load_dataset("json", data_files=data_files, num_proc=32).with_format("torch")["train"]
        self.dataset_train = concatenate_datasets([self.dataset_train, dataset_train_2])

        self.dataset_train = self.dataset_train.train_test_split(test_size=0.01, seed=42)
        self.dataset_valid = self.dataset_train["test"]
        self.dataset_train = self.dataset_train["train"]
        self.num_workers = num_workers
        self.batch_size = batch_size

    def get_dataloader(self, dataset) -> DataLoader:
        return DataLoader(
            dataset=dataset,            
            num_workers=self.num_workers,
            batch_size=self.batch_size, 
            shuffle=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.dataset_train)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.dataset_valid)
