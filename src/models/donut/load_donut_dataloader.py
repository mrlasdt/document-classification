# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Donut/RVL-CDIP/Fine_tune_Donut_on_toy_RVL_CDIP_%28document_image_classification%29.ipynb?authuser=2#scrollTo=CfJMb2o31AA-
# %%
import random
from typing import Any, List, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import DonutProcessor
import re


class DonutDatasetForDocumentClassification(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string).
    Args:
        df: dataframe contains image_path and label
        max_length: the max number of tokens for the target sequences
        split: whether to load "train", "validation" or "test" split
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
        prompt_end_token: the special token at the end of the sequences
        sort_json_key: whether or not to sort the JSON keys
    Returns:
        train or val dataset
    """

    def __init__(
        self,
        df: pd.DataFrame,
        processor,
        max_length: int,
        labels: list[int],
        split: str,
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str = None,
        class_token: str = "s_class",
    ):
        super().__init__()
        self.max_length = max_length
        self.processor = processor
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self._class_token = class_token
        self.df = df
        self.gt_token_sequences = self.df["label"].apply(self.label2token_sequence)
        self.add_tokens([self.end_tag(t) for t in labels] + [self.task_start_token, self.prompt_end_token,
                        self.start_tag(self._class_token), self.end_tag(self._class_token)])
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    @staticmethod
    def start_tag(token):
        return f"<{token}>"

    @staticmethod
    def end_tag(token):
        return f"<{token}/>"

    @staticmethod
    def remove_tag(tag):
        match = re.search(r"<(?P<origin>.*?)/?>", tag)
        return match['origin'] if match else ''

    def token_sequence2label(self, token_seq):
        """ 
        '<FWD><s_class><POS04/><s_class/></s>' -> POS04
        """
        seq = token_seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        seq = self.processor.token2json(seq)
        return self.remove_tag(seq["text_sequence"])

    def label2token_sequence(self, label):
        return self.start_tag(
            self._class_token) + self.end_tag(label) + self.end_tag(
            self._class_token) + self.processor.tokenizer.eos_token

    @property
    def total_tokens_num(self):
        return len(self.processor.tokenizer)

    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add tokens to tokenizer and resize the token embeddings of the decoder
        """
        _newly_added_num = self.processor.tokenizer.add_tokens(list_of_tokens)
        # if newly_added_num > 0:
        # model.decoder.resize_token_embeddings(len(self.processor.tokenizer))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels
        Convert gt data into input_ids (tokenized string)
        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """

        sample = Image.open(self.df["img_path"].iloc[idx]).convert("RGB")
        # pixel values (we remove the batch dimension)
        pixel_values = self.processor(sample, random_padding=self.split == "train", return_tensors="pt").pixel_values
        if self.split != "train":
            return dict(pixel_values=pixel_values, labels=self.gt_token_sequences.iloc[idx])

        pixel_values = pixel_values.squeeze()
        # labels, which are the input ids of the target sequence
        target_sequence = self.gt_token_sequences.iloc[idx]  # can be more than one, e.g., DocVQA Task 1
        input_ids = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        # model doesn't need to predict pad token
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id
        # labels[: torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
        encoding = dict(pixel_values=pixel_values,
                        labels=labels)
        return encoding


def split_df(df, test_size, shuffle, seed, stratify):
    y_label = df["label"] if stratify else None
    df_train, df_test = train_test_split(df, test_size=test_size,
                                         shuffle=shuffle, random_state=seed, stratify=y_label)
    return df_train, df_test


def load_data(
        df_path, pretrained_processor_path, task_start_token, prompt_end_token, labels, image_size, max_seq_len,
        batch_size, test_size, shuffle, seed, stratify, num_workers):
    df = pd.read_csv(df_path)
    df_train, df_val = split_df(df, test_size, shuffle, seed, stratify)
    processor = DonutProcessor.from_pretrained(pretrained_processor_path)
    processor.image_processor.size = image_size[::-1]  # should be (width, height)
    processor.image_processor.do_align_long_axis = False
    train_dataset = DonutDatasetForDocumentClassification(
        df_train, processor, split="train", max_length=max_seq_len, labels=labels, task_start_token=task_start_token,
        prompt_end_token=prompt_end_token)
    val_dataset = DonutDatasetForDocumentClassification(
        df_val, processor, split="val", max_length=max_seq_len, labels=labels, task_start_token=task_start_token,
        prompt_end_token=prompt_end_token)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    return train_dataloader, val_dataloader


# %%
if __name__ == "__main__":
    # %% try load_data
    import sys
    sys.path.append("/mnt/ssd500/hungbnt/DocumentClassification")
    from config import config as cfg
    from transformers import DonutProcessor

    df_path = "/mnt/ssd500/hungbnt/DocumentClassification/data/FWD.csv"
    processor = DonutProcessor.from_pretrained(
        "/mnt/ssd500/hungbnt/DocumentClassification/weights/pretrained/clova_donut_processor")
    max_seq_len = 4
    labels = cfg.LABELS
    batch_size = 8
    test_size = 0.2
    shuffle = True
    seed = cfg.SEED
    stratify = True
    train_dataloader, val_dataloader = load_data(
        df_path, processor, max_seq_len, labels, batch_size, test_size, shuffle, seed, stratify)
    # %%
    len(train_dataloader), len(val_dataloader),
    # %%
    train_dataloader.batch_size
    # %%
    b = next(iter(train_dataloader))
    b[0].shape
    # # batch["pixel_values"]
    # # %%
    # df_path = "/mnt/ssd500/hungbnt/DocumentClassification/data/FWD.csv"
    # df = pd.read_csv(df_path)

    # df

    # # %% try dataset
    # from transformers import DonutProcessor
    # import sys
    # sys.path.append("/mnt/ssd500/hungbnt/DocumentClassification")
    # from config import config as cfg
    # processor = DonutProcessor.from_pretrained(
    #     "/mnt/ssd500/hungbnt/DocumentClassification/weights/pretrained/clova_donut_processor")
    # dataset = DonutDatasetForDocumentClassification(df, processor, max_length=4)
    # dataset.add_tokens([dataset.end_tag(l) for l in cfg.LABELS])
    # len(dataset)
    # # %%

    # # %%
    # dataset[0]["labels"], processor.decode(dataset[0]["labels"])
    # # %%
    # dataset[0]['pixel_values'].size(), dataset[0]['pixel_values'][0].size()
    # # %% Try reconstruct the image from pixel_values
    # from PIL import Image
    # import numpy as np

    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)

    # # unnormalize
    # reconstructed_image = (dataset[0]['pixel_values'][0] * torch.tensor(std)
    #                        [:, None, None]) + torch.tensor(mean)[:, None, None]
    # # unrescale
    # reconstructed_image = reconstructed_image * 255
    # # convert to numpy of shape HWC
    # print(reconstructed_image.size())
    # reconstructed_image = torch.moveaxis(reconstructed_image, 0, -1)
    # print(reconstructed_image.size())
    # image = Image.fromarray(reconstructed_image.numpy().astype(np.uint8))
    # image
    # # %%
    # from torch.utils.data import DataLoader
    # train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # train_dataloader.dataset.total_tokens_num

    # %%
    # from transformers import DonutImageProcessor, PreTrainedTokenizerFast
    # import os
    # os.environ["CURL_CA_BUNDLE"] = ""

    # image_processor = DonutImageProcessor.from_pretrained("naver-clova-ix/donut-base")
    # tokenizer = PreTrainedTokenizerFast.from_pretrained("naver-clova-ix/donut-base")
    # processor = DonutProcessor(image_processor, tokenizer)
    # ValueError: Cannot instantiate this tokenizer from a slow version. If it's based on sentencepiece, make sure you have sentencepiece installed.

    # %%
    DonutDatasetForDocumentClassification.remove_tag("<FWD/>")

    # %%
    tag = "<FWD>"
    a = re.search(r"<(?P<origin>.*?)/?>", tag)
    print(a["origin"])
# %%
