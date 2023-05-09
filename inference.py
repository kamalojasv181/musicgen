import argparse
from transformers import AutoModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='archinetai/mousai-v1')
    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model)
    model.save_pretrained(args.model)