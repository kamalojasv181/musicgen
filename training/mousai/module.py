import random
from typing import Any, Dict, List, Optional, Tuple 
from math import pi 
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from ema_pytorch import EMA
from torch import Tensor, nn
from transformers import AutoModel 
import os
import numpy as np
import json
from frechet_audio_distance import FrechetAudioDistance
import laion_clap
import sys
sys.path.append("../../")

from utils import save_wav, load_json


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
        autoencoder_name: str,
        validation_path: str, 
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

        self.validation_path = validation_path
        self.frechet = FrechetAudioDistance(
            model_name="pann",
            use_pca=False,
            use_activation=False,
            verbose=False
        )

        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
        self.clap_model.load_ckpt("../../../music_audioset_epoch_15_esc_90.14.pt")

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
            if infos['title'][0][i] != "__None__" and not (dropout(0.1) and use_dropout):
                tags.append(infos['title'][0][i])

            if infos['genre'][0][i] != "__None__" and not (dropout(0.1) and use_dropout):
                tags.append(infos['genre'][0][i])

            if infos['artist'][0][i] != "__None__" and not (dropout(0.1) and use_dropout):
                tags.append(infos['artist'][0][i])

            if infos['album'][0][i] != "__None__" and not (dropout(0.2) and use_dropout):
                tags.append(infos['album'][0][i])

            if infos['year'][0][i] != "__None__" and not (dropout(0.2) and use_dropout):
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
        batch_size=wave.shape[0]
        text = self.get_texts(info) 
        loss = self.model(latent, text=text, embedding_mask_proba=self.embedding_mask_proba)
        self.log("train_loss", loss, sync_dist=True, batch_size=batch_size)
        self.model_ema.update()
        self.log("ema_decay", self.model_ema.get_current_decay(), sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        
        wave, info = batch["wave"], batch["info"]
        
        batch_size=wave.shape[0]
        latent = self.encode(wave)
        text = self.get_texts(info) 
        loss = self.model(latent, text=text)
        self.log("valid_loss", loss, sync_dist=True, batch_size=batch_size)

        samples, info = self.generate_samples(text=text)

        self.save_samples(samples, info, batch_idx)

        return loss
    
    def on_validation_epoch_end(self) -> None:
        # log the fad score and clap score
        fad_score = self.fad(
            background_dir=os.path.join(self.validation_path, "true"),
            eval_dir=os.path.join(self.validation_path, "generated")
        )

        self.log("fad_score", fad_score, sync_dist=True)

        # load the json file prompt_to_path.json
        prompt_to_path = load_json(os.path.join(self.validation_path, "prompt_to_path.json"))

        # compute the clap score
        clap_score = self.clap(prompt_to_path)

        self.log("clap_score", clap_score, sync_dist=True)

    @torch.no_grad()
    def generate_samples(
            self, 
            text: List[str],
            latent_channels: int = 32,
            sampling_steps: int = 100, 
            decoding_steps: int = 100, 
            cfg_scale: int = 5.0, 
            seed: int = 42, 
            length: int = 2048, 
            device: torch.device = None,
            show_progress: bool = True, 
            with_info: bool = True
        ) -> Tuple[Tensor, Dict]:
        
        device = self.device
        generator = torch.Generator(device=device)
        noise_shape = (len(text), latent_channels, length)
        noise = torch.randn(noise_shape, device=device, generator=generator)

        latent = self.model.sample(
            noise, 
            num_steps=sampling_steps,
            text=text,
            embedding_scale=cfg_scale,
            show_progress=show_progress,
        )
        samples = self.autoencoder.decode(
            latent, 
            num_steps=decoding_steps, 
            generator=generator,
            show_progress=show_progress,
        )

        info = dict(
            text=text, 
            sampling_steps=sampling_steps,
            decoding_steps=decoding_steps,
            cfg_scale=cfg_scale,
            seed=seed, 
            length=length, 
            latent=latent,
        )

        return samples, info
    
    def save_samples(self, samples: Tensor, info: Dict, batch_idx: int):
        for i, (sample, prompt) in enumerate(zip(samples, info["text"])):
            sample_cpu = sample.cpu().numpy()
            sample_cpu = np.transpose(sample_cpu)
            sample_cpu = sample_cpu / np.max(np.abs(sample_cpu))

            # save in the self.validation_path/generated folder by the name batch_idx_prompt_idx.wav
            save_wav(sample_cpu, os.path.join(self.validation_path, "generated", f"{batch_idx}_{i}.wav"), sr=48000)

            # load the json file prompt_to_path.json
            prompt_to_path = load_json(os.path.join(self.validation_path, "prompt_to_path.json"))

            # add the prompt and the path to the json file
            prompt_to_path[prompt] = os.path.join(self.validation_path, "generated", f"{batch_idx}_{i}.wav")

            # save the json file
            with open(os.path.join(self.validation_path, "prompt_to_path.json"), "w") as f:
                f.write(json.dumps(prompt_to_path))


    def fad(self, background_dir: str, eval_dir: str) -> float:
        fad_score = self.frechet.score(
            background_dir=background_dir,
            eval_dir=eval_dir
        )

        return fad_score
    
    def clap(self, audio_text_pairs: Dict) -> float:
        
        texts = list(audio_text_pairs.keys())
        audio_paths = list(audio_text_pairs.values())

        audio_embeds = self.clap_model.get_audio_embedding_from_filelist(x = audio_paths, use_tensor=True)
        text_embeds = self.clap_model.get_text_embedding(texts, use_tensor=True)

        dot_product = torch.matmul(audio_embeds, text_embeds.T)
        cosine_similarity = dot_product / (torch.norm(audio_embeds, dim=1) * torch.norm(text_embeds, dim=1))

        avg_cosine_similarity = torch.mean(cosine_similarity)

        return avg_cosine_similarity.item()
