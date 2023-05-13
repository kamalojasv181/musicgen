import torch
from audiolm_pytorch import HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer, CoarseTransformer, CoarseTransformerTrainer, FineTransformer, FineTransformerTrainer, AudioLM
import os
from audiolm_pytorch import SoundStream, SoundStreamTrainer
from musiclm_pytorch import MusicLM, MuLaNEmbedQuantizer
import sys

sys.path.append("../..")

from utils import *

def download_hubert():

    if not os.path.exists(".cache/hubert"):
        os.makedirs(".cache/hubert")

    # if "hubert_base_ls960.pt" does not exist, download it
    if not os.path.exists(".cache/hubert/hubert_base_ls960.pt"):
        print("Downloading hubert_base_ls960.pt")
        os.system("wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt -P .cache/hubert")

    # if "hubert_base_ls960_L9_km500.bin" does not exist, download it
    if not os.path.exists(".cache/hubert/hubert_base_ls960_L9_km500.bin"):
        print("Downloading hubert_base_ls960_L9_km500.bin")
        os.system("wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin -P .cache/hubert")




if __name__=="__main__":

    # load the config TODO: create a config file
    config = load_config("config_music_lm.yaml")

    # download hubert
    download_hubert()

    # create a wav2vec object
    wav2vec = HubertWithKmeans(
        checkpoint_path = './.cache/hubert/hubert_base_ls960.pt',
        kmeans_path = './.cache/hubert/hubert_base_ls960_L9_km500.bin',
    ).to(config.device)

    # load mulan model
    mulan = torch.load(config.mulan_path).to(config.device)

    # create quantizer
    quantizer = MuLaNEmbedQuantizer(
        mulan=mulan,
        conditioning_dims=(config.semantic_transformer.dim, config.coarse_transformer.dim, config.fine_transformer.dim),
        namespaces=("semantic", "coarse", "fine"),
    ).to(config.device)

    # create a semantic transformer
    semantic_transformer = SemanticTransformer(
        num_semantic_tokens=wav2vec.codebook_size,
        dim=config.semantic_transformer.dim,
        depth=config.semantic_transformer.depth,
        audio_text_condition=True
    ).to(config.device)

    # create a coarse transformer
    coarse_transformer = CoarseTransformer(
        num_semantic_tokens=wav2vec.codebook_size,
        dim=config.coarse_transformer.dim,
        codebook_size=config.coarse_transformer.codebook_size,
        depth=config.coarse_transformer.depth,
        num_coarse_quantizers=config.coarse_transformer.num_coarse_quantizers,
        audio_text_condition=True
    ).to(config.device)

    # create a fine transformer
    fine_transformer = FineTransformer(
        num_coarse_quantizers=config.fine_transformer.num_coarse_quantizers,
        num_fine_quantizers=config.fine_transformer.num_fine_quantizers,
        codebook_size=config.fine_transformer.codebook_size,
        dim=config.fine_transformer.dim,
        depth=config.fine_transformer.depth,
        audio_text_condition=True
    ).to(config.device)

    # create a soundstream
    soundstream = SoundStream(
        codebook_size=config.fine_transformer.codebook_size,
        rq_num_quantizers=config.soundstream.rq_num_quantizers,
        rq_groups=config.soundstream.rq_groups,
        attn_window_size=config.soundstream.attn_window_size,
        attn_depth=config.soundstream.attn_depth
    ).to(config.device)

    # train the soundstream
    trainer = SoundStreamTrainer(
        soundstream,
        folder=config.train_audios_folder,
        batch_size=config.batch_size,
        grad_accum_every=config.soundstream.grad_accum_every,
        data_max_length_seconds=config.data_max_length_seconds,
        num_train_steps=config.num_train_steps
    )

    trainer.train()

    # train the semantic transformer
    trainer = SemanticTransformerTrainer(
        transformer=semantic_transformer,
        wav2vec=wav2vec,
        audio_conditioner=quantizer,
        folder=config.train_audios_folder,
        batch_size=config.batch_size,
        data_max_length=config.data_max_length,
        num_train_steps=config.num_train_steps
    )

    trainer.train()

    # train the coarse transformer
    trainer = CoarseTransformerTrainer(
        transformer=coarse_transformer,
        codec=soundstream,
        audio_conditioner=quantizer,
        folder=config.train_audios_folder,
        batch_size=config.batch_size,
        data_max_length=config.data_max_length,
        num_train_steps=config.num_train_steps
    )

    trainer.train()

    # train the fine transformer
    trainer = FineTransformerTrainer(
        transformer=fine_transformer,
        codec=soundstream,
        audio_conditioner=quantizer,
        folder=config.train_audios_folder,
        batch_size=config.batch_size,
        data_max_length=config.data_max_length,
        num_train_steps=config.num_train_steps
    )

    trainer.train()

    # create an audio lm
    audiolm = AudioLM(
        wav2vec=wav2vec,
        codec=soundstream,
        semantic_transformer=semantic_transformer,
        coarse_transformer=coarse_transformer,
        fine_transformer=fine_transformer,
        audio_conditioner=quantizer
    )

    musiclm = MusicLM(
        audio_lm=audiolm,
        mulan_embed_quantizer=quantizer
    )

    # save the model
    torch.save(musiclm, config.musiclm_path)