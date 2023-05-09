import json
import os

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
        self.__dict__.update(args)\

def load_pickle(path):

    import pickle

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data

def save_pickle(data, path):

    import pickle

    with open(path, 'wb') as f:
        pickle.dump(data, f)