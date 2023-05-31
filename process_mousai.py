import os
import glob
import mutagen
import librosa
import random
import argparse
import pandas as pd
from datasets import Dataset
import json


def get_metadata(audio_file):
    audio = mutagen.File(audio_file)

    if audio is None:
        return None
    
    metadata = {
        "title": [audio.tags["TIT2"].text[0]] if "TIT2" in audio.tags else None,
        "artist": [audio.tags["TPE1"].text[0]] if "TPE1" in audio.tags else None,
        "album": [audio.tags["TALB"].text[0]] if "TALB" in audio.tags else None,
        "year": [str(audio.tags["TDRC"].text[0])[0:4]] if "TDRC" in audio.tags else None,
        "genre": [audio.tags["TCON"].text[0]] if "TCON" in audio.tags else None,
    }

    return metadata

if __name__ == '__main__':

    # create an argument parser

    parser = argparse.ArgumentParser(description="Prepare the dataset for training")

    # add arguments to the parser

    parser.add_argument("--data_folder", type=str, help="Path to the folder containing the audio files")
    parser.add_argument("--start_id", type=int, help="Start id for the audio files", default=0)

    args = parser.parse_args()

    data_folder = args.data_folder

    audio_files = glob.glob(os.path.join(data_folder, "*.mp3"))

    for id, audio_file in enumerate(audio_files):

        if id < args.start_id:
            continue

        metadata = get_metadata(audio_file)

        # get the audio as dual channel
        audio, sr = librosa.load(audio_file, sr=48000, mono=False)

        # if we only get one dimension, then skip
        if len(audio.shape) == 1:
            continue
        
        # find the length of the audio in number of samples
        audio_len = audio.shape[1]

        if audio_len < 2097152:
            continue

        # get 4 random chunks of 2097152 samples each
        starts = []

        for i in range(4):
            starts.append(random.randint(0, audio_len - 2097152))

        # sort the starts

        starts.sort()

        # get the chunks

        for i in range(4):
            audio_chunk = audio[:, starts[i]:starts[i] + 2097152]

            # convert chunk to list
            audio_chunk = audio_chunk.tolist()

            datapoint_new = {
                "wave": audio_chunk,
                "info": metadata,
            }

            datapoint_new["info"]["crop_id"] = i
            datapoint_new["info"]["num_crops"] = 4

            with open(f"{data_folder}/train_data.json", "a") as f:
                f.write(json.dumps(datapoint_new))
                f.write("\n")