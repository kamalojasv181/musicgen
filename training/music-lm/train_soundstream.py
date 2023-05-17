import torch
from audiolm_pytorch import HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer, CoarseTransformer, CoarseTransformerTrainer, FineTransformer, FineTransformerTrainer, AudioLM
import os
from audiolm_pytorch import SoundStream, SoundStreamTrainer
from musiclm_pytorch import MusicLM, MuLaNEmbedQuantizer
import sys

sys.path.append("../..")

from utils import *


if __name__=="__main__":

    # load the config
    config = load_config("config_music_lm.yaml")

    if not os.path.exists("./soundstream_results"):
        os.makedirs("./soundstream_results")

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
        num_train_steps=config.num_train_steps,
        force_clear_prev_results=True,
        save_model_every=100000,
        results_folder="./soundstream_results"
    )

    trainer.train()
