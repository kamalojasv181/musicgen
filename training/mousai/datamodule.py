import random
from math import pi 

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import load_from_disk

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
        self.dataset_train = load_from_disk(dataset_train_path).with_format("torch")
        self.dataset_valid = load_from_disk(dataset_valid_path).with_format("torch")
        self.num_workers = num_workers
        self.batch_size = batch_size

    def get_dataloader(self, dataset) -> DataLoader:
        return DataLoader(
            dataset=dataset,            
            num_workers=self.num_workers,
            batch_size=self.batch_size, 
            shuffle=False,
            pin_memory=False,
            prefetch_factor=2,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.dataset_train)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.dataset_valid)