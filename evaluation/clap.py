import argparse
import sys
import numpy as np
import librosa
import torch
import laion_clap
sys.path.append("../")

from utils import *


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_text_pairs", type=str, required=True, help="Path to the audio-text pairs", default="atp.json")

    args = parser.parse_args()

    # load audio-text pairs
    audio_text_pairs = load_json(path=args.audio_text_pairs)

    texts = list(audio_text_pairs.keys())
    audio_paths = list(audio_text_pairs.values())

    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    
    # get audio embeddings
    audio_embeds = model.get_audio_embedding_from_filelist(x = audio_paths, use_tensor=True)

    # get text embeddings
    text_embeds = model.get_text_embedding(texts, use_tensor=True)

    # compute dot product
    dot_product = torch.matmul(audio_embeds, text_embeds.T)

    # compute cosine similarity
    cosine_similarity = dot_product / (torch.norm(audio_embeds, dim=1) * torch.norm(text_embeds, dim=1))

    # compute the average cosine similarity
    avg_cosine_similarity = torch.mean(cosine_similarity)

    # print the average cosine similarity
    print(f"CLAP Score: {avg_cosine_similarity}")

