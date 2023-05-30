import random
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Type, Tuple 
from math import pi 

import json 
import plotly.graph_objs as go
import pytorch_lightning as pl
import torch
import torchaudio
import wandb
import torch.nn.functional as F 
from einops import rearrange, repeat
from ema_pytorch import EMA
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm 

from archisound import ArchiSound
from transformers import AutoModel 
import os


torch.set_float32_matmul_precision('high')

from audio_diffusion_pytorch import UNetV0

UNetT = lambda: UNetV0

def dropout(proba: float):
    return random.random() < proba

class Module(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_eps: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_weight_decay: float,
        ema_beta: float,
        ema_power: float,
        model: nn.Module,
        embedding_mask_proba: float,
        autoencoder_name: str 
    ):
        super().__init__()
        self.lr = lr
        self.lr_eps = lr_eps
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_weight_decay = lr_weight_decay
        self.model = model
        self.model_ema = EMA(self.model, beta=ema_beta, power=ema_power)
        self.embedding_mask_proba = embedding_mask_proba
        self.autoencoder = AutoModel.from_pretrained(
            f"archinetai/{autoencoder_name}", trust_remote_code=True, use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
        )

    @property
    def device(self):
        return next(self.model.parameters()).device

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()),
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer

    def get_texts(self, infos: Dict, use_dropout: bool = True) -> List[str]:
        texts = []

        batch_size = len(infos['title'][0])
        for i in range(batch_size):
            tags = []
            if infos['title'][0][i] != None and not (dropout(0.1) and use_dropout):
                tags.append(infos['title'][0][i])

            if infos['genre'][0][i] != None and not (dropout(0.1) and use_dropout):
                tags.append(infos['genre'][0][i])

            if infos['artist'][0][i] != None and not (dropout(0.1) and use_dropout):
                tags.append(infos['artist'][0][i])

            if infos['album'][0][i] != None and not (dropout(0.2) and use_dropout):
                tags.append(infos['album'][0][i])

            if infos['year'][0][i] != None and not (dropout(0.2) and use_dropout):
                tags.append(infos['year'][0][i])

            if infos['crop_id'][i].item() != None and infos['num_crops'][i].item() != None and not (dropout(0.2) and use_dropout):
                tags.append(f"{infos['crop_id'][i].item()+1} of {infos['num_crops'][i].item()}")

            random.shuffle(tags)
            text = ' '.join(tags) if random.random() > 0.5 else ', '.join(tags)
            texts.append(text)

        return texts

    @torch.no_grad() 
    def encode(self, wave: Tensor) -> Tensor:
        return self.autoencoder.encode(wave)

    def training_step(self, batch, batch_idx):
        wave, info = batch["wave"], batch["info"]
        latent = self.encode(wave)
        text = self.get_texts(info) 
        loss = self.model(latent, text=text, embedding_mask_proba=self.embedding_mask_proba)
        self.log("train_loss", loss, sync_dist=True)
        self.model_ema.update()
        self.log("ema_decay", self.model_ema.get_current_decay(), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        wave, info = batch["wave"], batch["info"]
        latent = self.encode(wave)
        text = self.get_texts(info) 
        loss = self.model(latent, text=text)
        self.log("valid_loss", loss, sync_dist=True)
        return loss