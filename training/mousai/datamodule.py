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
        train_files: list,
        valid_files: list,
        test_files: list,
        batch_size: int,
        *,
        num_workers: int,
        num_proc: int = 16,
        **kwargs: int,
    ) -> None:
        super().__init__()

        self.dataset_train = load_dataset("json", data_files=train_files, num_proc=num_proc).with_format("torch")["train"]

        self.dataset_valid = load_dataset("json", data_files=valid_files, num_proc=num_proc).with_format("torch")["train"]

        self.dataset_test = load_dataset("json", data_files=test_files, num_proc=num_proc).with_format("torch")["train"]

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
    
    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.dataset_test)
