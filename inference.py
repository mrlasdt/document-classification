#%%
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
    # # %%
    # setup = """
    # from config import config as cfg
    # from src.core.runner import Runner
    # device = "cuda:0"
    # runner = Runner(cfg.DONUT_CFG)
    # model = runner.model.to(device)
    # data = runner.valid.dataset
    # sample = next(iter(data))
    # processor = data.processor
    # task_prompt = runner.trainer.task_prompt
    # """
    # # %%
    # stmt = """
    # pixel_values = sample["pixel_values"].to(device)
    # # prepare decoder inputs
    # decoder_input_ids = processor.tokenizer(
    #     task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    # decoder_input_ids = decoder_input_ids.to(device)

    # # autoregressively generate sequence

    # outputs = model.generate(
    #     pixel_values,
    #     decoder_input_ids=decoder_input_ids,
    #     max_length=model.decoder.config.max_position_embeddings,
    #     early_stopping=True,
    #     pad_token_id=processor.tokenizer.pad_token_id,
    #     eos_token_id=processor.tokenizer.eos_token_id,
    #     use_cache=True,
    #     num_beams=1,
    #     bad_words_ids=[[processor.tokenizer.unk_token_id]],
    #     return_dict_in_generate=True,
    # )

    # # turn into JSON
    # seq = processor.batch_decode(outputs.sequences)[0]
    # """
    # # %%
    # import timeit
    # timeit.timeit(setup=setup, stmt=stmt, number=100000)

    # %%
    from config import global as cfg
    from src.core.runner import Runner
    
    runner = Runner(cfg.DONUT_CFG)
    device = cfg.DEVICE
    runner.trainer.on_train_begin(runner.train, runner.valid)
    model = runner.model.to(device)
    data = runner.valid.dataset
    sample = next(iter(data))
    processor = data.processor
    task_prompt = runner.trainer.task_prompt
    # %%
    import time
    # %%
    ltimes = []
    for i in range(100):
        start = time.time()
        pixel_values = sample["pixel_values"].to(device)
        # prepare decoder inputs
        decoder_input_ids = processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        decoder_input_ids = decoder_input_ids.to(device)

        # autoregressively generate sequence

        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        # turn into JSON
        seq = processor.batch_decode(outputs.sequences)[0]
        end = time.time() - start
        ltimes.append(end)
    ltimes
    # %%
    print(sum(ltimes) / len(ltimes))

# %%
