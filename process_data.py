import os
import json
import argparse
from datasets import load_dataset
import torch
from x_clip.tokenizer import tokenizer


def encode_text(batch):
    batch["text"] = tokenizer.tokenize(batch["text"])
    batch["audio"] = batch["audio"]["array"]
    return batch

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='sd', help='data directory')
    parser.add_argument('--output_dir', type=str, default='sample_data_hf', help='processed huggingface data directory')
    parser.add_argument('--test_size', type=float, default=0.05 , help='test size')
    parser.add_argument('--seed', type=int, default=42, help='seed')


    args = parser.parse_args()

    data_dir = args.data_dir

    data = load_dataset('audiofolder', data_dir=data_dir)

    data = data.map(encode_text)

    data = data["train"].train_test_split(test_size=args.test_size, seed=args.seed)

    data = data.with_format("torch")

    data.save_to_disk(args.output_dir)