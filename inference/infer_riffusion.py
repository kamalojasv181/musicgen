from diffusers import DiffusionPipeline
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
import argparse
from transformers import AutoModel
import yaml
import os
import torch
import sys
import shutil
import numpy as np
sys.path.append("../")

from utils import *


def check_output_in_cache(prompt, output_dir):
    """
    Checks if the output is already in the cache
    """
    # check if the cache exists
    if not os.path.exists("./cache"):
        return False

    # if the cache exists, check if the riffusion.json exists
    if not os.path.exists("./cache/riffusion.json"):
        return False

    # check if the prompt is in ./cache/riffusion.json
    riffusion_cache = load_json("./cache/riffusion.json")

    if prompt in riffusion_cache:
        file_path = riffusion_cache[prompt]

        # check if the file exists
        if os.path.exists(file_path) and file_path == os.path.join(output_dir, song_prompt_to_name(prompt) + ".wav"):
            return True

        elif os.path.exists(file_path):
            # copy the file to the output dir
            shutil.copy(file_path, os.path.join(output_dir, song_prompt_to_name(prompt) + ".wav"))
            return True

        else:
            return False

    else:
        return False


def update_cache(prompt, output_dir):
    """
    Updates the cache
    """
    # check if the cache exists
    if not os.path.exists("./cache"):
        os.mkdir("./cache")

    if not os.path.exists("./cache/riffusion.json"):
        with open("./cache/riffusion.json", "w") as f:
            f.write("{}")

    # check if the prompt is in ./cache/riffusion.json
    riffusion_cache = load_json("./cache/riffusion.json")

    if prompt not in riffusion_cache:
        riffusion_cache[prompt] = os.path.join(output_dir, song_prompt_to_name(prompt) + ".wav")

    save_json(riffusion_cache, "./cache/riffusion.json")


def predict(prompt, path, length):
    spec = pipe(
        prompt,
        negative_prompt="",
        width=int(102.4 * length),
    ).images[0]
    
    wav = converter.audio_from_spectrogram_image(image=spec)
    wav.export(path, format='wav')

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./riffusion.yaml")
    parser.add_argument("--text_prompts", type=str, default='../prompts/electronic.txt')
    parser.add_argument("--output_dir", type=str, default='../output/riffusion')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # load the config
    config = load_config(path=args.config_path)

    # set seed
    set_seed(seed=config.seed)

    # load the prompts
    prompts = read_txt(args.text_prompts).split("\n")

    # remove empty prompts
    prompts = [prompt for prompt in prompts if prompt != ""]

    # filter the prompts
    prompts = [prompt for prompt in prompts if not check_output_in_cache(prompt, args.output_dir)]

    # if there are no prompts to generate, exit
    if len(prompts) == 0:
        print("All prompts are already in the cache")
        exit()

    # get the device
    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    params = SpectrogramParams()
    converter = SpectrogramImageConverter(params)

    pipe = DiffusionPipeline.from_pretrained(config.model_name)
    pipe = pipe.to(device)

    for prompt in prompts:

        save_path = song_prompt_to_name(song_prompt=prompt)

        predict(prompt, os.path.join(args.output_dir, save_path + ".wav"), config.length)

        update_cache(prompt, args.output_dir)