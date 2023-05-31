import random
from math import pi 

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import load_from_disk, load_dataset

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
        self.dataset_train = load_dataset("json", data_files=dataset_train_path).with_format("torch")["train"]
        self.dataset_valid = load_dataset("json", data_files=dataset_valid_path).with_format("torch")["train"]
        self.num_workers = num_workers
        self.batch_size = batch_size

    def get_dataloader(self, dataset) -> DataLoader:
        return DataLoader(
            dataset=dataset,            
            num_workers=self.num_workers,
            batch_size=self.batch_size, 
            shuffle=True,
            pin_memory=True,
            prefetch_factor=2,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.dataset_train)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.dataset_valid)