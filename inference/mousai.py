import argparse
from transformers import AutoModel
import yaml
import os
import torch
import sys
import shutil
import numpy as np
from tqdm import tqdm

sys.path.append("../")

from utils import *

def check_output_in_cache(prompt, output_dir):
    """
    Checks if the output is already in the cache
    """
    # check if the cache exists
    if not os.path.exists("./cache"):
        return False

    # check if the prompt is in ./cache/mousai.json
    mousai_cache = load_json("./cache/mousai.json")

    if prompt in mousai_cache:
        file_path = mousai_cache[prompt]

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

    if not os.path.exists("./cache/mousai.json"):
        with open("./cache/mousai.json", "w") as f:
            f.write("{}")

    # check if the prompt is in ./cache/mousai.json
    mousai_cache = load_json("./cache/mousai.json")

    if prompt not in mousai_cache:
        mousai_cache[prompt] = os.path.join(output_dir, song_prompt_to_name(prompt) + ".wav")

    save_json(mousai_cache, "./cache/mousai.json")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default='mousai.yaml')
    parser.add_argument("--text_prompts", type=str, default='../prompts/test.txt')
    parser.add_argument("--output_dir", type=str, default='../output_1')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # if the output dir does not exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # load the config
    config = load_config(args.config_path)

    # set seed
    set_seed(config.seed)

    # load the prompts
    prompts = read_txt(args.text_prompts).split('\n')

    # filter the prompts
    # prompts = [prompt for prompt in prompts if not check_output_in_cache(prompt, args.output_dir)]

    # if there are no prompts, exit
    if len(prompts) == 0:
        print("All prompts are in the cache")
        exit()

    # get the device
    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")
    

    # load the model
    model = AutoModel.from_pretrained(config.model_name, trust_remote_code=True, use_auth_token= os.environ["HUGGINGFACE_TOKEN"]).to(device)


    # divide the prompts into batches
    prompts = [prompts[i:i + config.batch_size] for i in range(0, len(prompts), config.batch_size)]


    for prompts_batch in tqdm(prompts):

        # generate the audios
        samples, info = model.sample(
        text=prompts_batch,
        sampling_steps=100,
        decoding_steps=100, 
        cfg_scale=8.0, 
        seed=config.seed, 
        length=2048
        )

        # save the audios
        for i, (sample, prompt) in enumerate(zip(samples, info["text"])):
            sample_cpu = sample.cpu().numpy()

            # transpose the sample
            sample_cpu = np.transpose(sample_cpu)

            save_file_name =  song_prompt_to_name(prompt)

            # save the audio
            save_wav(sample_cpu, os.path.join(args.output_dir, save_file_name + ".wav"), config.sr)

            # update the cache
            update_cache(prompt, args.output_dir)

