from config import __mapping__ as mcfg
import argparse
from src.core.runner import Runner
import os
from typing import Any
# https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models
# os.environ["CURL_CA_BUNDLE"] = ""


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--cfg', help='train config', type=str, default='base')
    parser.add_argument('--img', help='path to image', type=str, required=True)
    return parser.parse_args()


def get_cfg(args):
    cfg = mcfg[args.cfg]
    return cfg


class DocumentClassifier:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

        pass

    def preprocess(self):
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


def main():
    args = parse_args()
    cfg = get_cfg(args)
    classifer = DocumentClassifier(cfg)
    img_path = args.img
    classifer()


# %%
if __name__ == "__main__":
    # %%
    pass
