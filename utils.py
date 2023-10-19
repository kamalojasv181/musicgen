import json
import os
import numpy as np
from pydub import AudioSegment

def save_json(data, path):

    # save the file
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def set_seed(seed=0, verbose=False):
    import random
    import os

    if verbose: print("[Info] seed set to: {}".format(seed))

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

class Config:
    def __init__(self, **args):
        for key, value in args.items():
            if type(value) == dict:
                args[key] = Config(**value)
        self.__dict__.update(args)

def load_pickle(path):

    import pickle

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data

def save_pickle(data, path):

    import pickle

    with open(path, 'wb') as f:
        pickle.dump(data, f)

def read_txt(path):
    with open(path, 'r') as f:
        data = f.read()
    return data

def write_txt(data, path):
    with open(path, 'w') as f:
        f.write(data)

def load_yaml(path):
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return data

def load_config(path):
    import yaml
    with open(path) as f:
        config = yaml.safe_load(f)
        config = Config(**config)
    return config


def save_wav(data, path, sr):
    import soundfile as sf
    sf.write(path, data, sr, "PCM_24")

def song_prompt_to_name(song_prompt):
    # capitalize every word, remove punctuation marks like comma, period, etc.
    song_prompt = song_prompt.replace(".", "").replace("?", "").replace("!", "").replace(":", "").replace(";", "").replace("(", "").replace(")", "").replace("'", "").replace('"', "").replace("/", "")

    # replace spaces with underscores
    song_prompt = song_prompt.replace(" ", "_").replace(",", "")

    # remove multiple underscores
    song_prompt = song_prompt.replace("__", "_")

    # remove trailing underscores
    song_prompt = song_prompt.strip("_")

    # convert to lowercase
    song_prompt = song_prompt.lower()

    return song_prompt

def save_jsonl(data, path):
    import json
    with open(path, "w") as f:
        for point in data:
            f.write(json.dumps(point) + "\n")

def load_jsonl(path):
    import json
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def save_stereo_audio(channels, output_path, sample_rate=48000):
    if len(channels) != 2:
        raise ValueError("The 'channels' argument must be a list containing two sublists, one for the left channel and one for the right channel.")
    
    left_channel_data = channels[0]
    right_channel_data = channels[1]

    # Convert input lists to numpy arrays
    left_audio_array = np.array(left_channel_data)
    right_audio_array = np.array(right_channel_data)

    # Assuming your audio samples are in 16-bit PCM format (adjust as needed)
    sample_width = 2  # 16 bits = 2 bytes
    channels = 2  # Stereo audio

    # Convert numpy arrays to AudioSegments
    left_audio = AudioSegment(
        left_audio_array.tobytes(), 
        frame_rate=sample_rate, 
        sample_width=sample_width, 
        channels=1
    )
    right_audio = AudioSegment(
        right_audio_array.tobytes(), 
        frame_rate=sample_rate, 
        sample_width=sample_width, 
        channels=1
    )

    # Combine left and right channels into a stereo audio segment
    stereo_audio = AudioSegment.from_mono_audiosegments(left_audio, right_audio)

    # Export the stereo audio as an MP3 file
    stereo_audio.export(output_path, format="mp3")