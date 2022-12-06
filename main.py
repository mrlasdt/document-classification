from config import __mapping__ as mcfg
import argparse
from srcc.runner import Runner
import os
# https://stackoverflow.com/questions/71692354/facing-ssl-error-with-huggingface-pretrained-models
os.environ["CURL_CA_BUNDLE"] = ""


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--cfg', help='train config', type=str, default='base')
    return parser.parse_args()


def get_cfg(args):
    cfg = mcfg[args.cfg]
    return cfg


def main():
    args = parse_args()
    cfg = get_cfg(args)
    runner = Runner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
