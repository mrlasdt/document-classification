# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LiLT/Fine_tune_LiLT_on_a_custom_dataset%2C_in_any_language.ipynb
from torch.utils.data import Dataset
# from PIL import Image #no image input for lilt
import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
import cv2
from externals.ocr.utils import read_ocr_result_from_txt
from sklearn.model_selection import train_test_split
from externals.ocr.word_formation import Word, words_to_lines
from transformers import PhobertTokenizer
from functools import partial


def convert_word_bbox_to_word_group_bbox(lbboxes, lwords) -> Tuple[list, list]:
    lWords = [Word(text=word, bndbox=bbox) for word, bbox in zip(lwords, lbboxes)]
    list_lines, _ = words_to_lines(lWords)
    lwords_ = list()
    lbboxes_word_group = list()
    for line in list_lines:
        for word_group in line.list_word_groups:
            for word in word_group.list_words:
                lwords_.append(word.text)
                lbboxes_word_group.append(word_group.boundingbox)
    return lbboxes_word_group, lwords_


def split_df(df, test_size, shuffle, seed, stratify):
    y_label = df["label"] if stratify else None
    df_train, df_test = train_test_split(df, test_size=test_size,
                                         shuffle=shuffle, random_state=seed, stratify=y_label)
    return df_train, df_test


def load_train_eval_df(df_path, test_size, shuffle, seed, stratify):
    df = pd.read_csv(df_path)
    df_train, df_val = split_df(df, test_size, shuffle, seed, stratify)
    print('Train: ', len(df_train))
    print('Val: ', len(df_val))
    return df_train, df_val


def _normalize_box(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]


def load_ocr_result_and_normalize(example):
    r"""Load OCR labels, i.e. (word, bbox) pairs, into the input DataFrame containing of columns (image_path, ocr_path, label)"""
    # print(example)
    ocr_path = example['ocr_path']
    image_path = example["img_path"]
    assert os.path.exists(ocr_path), ocr_path
    assert os.path.exists(image_path), image_path
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    lbboxes, lwords = read_ocr_result_from_txt(ocr_path)
    lbboxes_word_group, lwords = convert_word_bbox_to_word_group_bbox(lbboxes, lwords)  # exclusive for layoutlmv3
    lbboxes_word_group = [_normalize_box(bbox, w, h) for bbox in lbboxes_word_group]
    example["words"] = lwords
    example["bbox"] = lbboxes_word_group
    return example


class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_seq_len, labels):
        self.df = df
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.labels = labels
        self.label2idx = {labels[i]: i for i in range(len(labels))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get item
        example = load_ocr_result_and_normalize(self.df.iloc[idx])
        bbox = []
        for word, box in zip(example["words"], example["bbox"]):
            n_word_tokens = len(self.tokenizer.tokenize(word))
            bbox.extend([box] * n_word_tokens)

        cls_box = sep_box = [0, 0, 0, 0]
        bbox = [cls_box] + bbox + [sep_box]

        encoding = self.tokenizer(" ".join(example["words"]), truncation=True, max_length=self.max_seq_len)
        sequence_length = len(encoding.input_ids)
        # truncate boxes and labels based on length of input ids
        bbox = bbox[:sequence_length]
        encoding["bbox"] = bbox
        encoding["labels"] = self.label2idx[example["label"]]
        return encoding


def collate_fn(features, tokenizer, max_seq_len):
    boxes = [feature["bbox"] for feature in features]
    # labels = [feature["labels"] for feature in features] uncessary
    # use tokenizer to pad input_ids
    batch = tokenizer.pad(features, padding="max_length", max_length=max_seq_len)
    sequence_length = torch.tensor(batch["input_ids"]).shape[1]
    # batch["labels"] = labels #uncessary
    batch["bbox"] = [boxes_example + [[0, 0, 0, 0]] * (sequence_length - len(boxes_example)) for boxes_example in boxes]
    # convert to PyTorch
    # batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}
    batch = {k: torch.tensor(v) for k, v in batch.items()}
    return batch


def load_data(
        df_path, pretrained_tokenizer_path, labels, max_seq_len, batch_size, test_size, shuffle, seed, stratify,
        num_workers):
    train_df, eval_df = load_train_eval_df(df_path, test_size, shuffle, seed, stratify)
    tokenizer = PhobertTokenizer.from_pretrained(pretrained_tokenizer_path)
    train_dataset = CustomDataset(train_df, tokenizer, max_seq_len, labels)
    eval_dataset = CustomDataset(eval_df, tokenizer, max_seq_len, labels)
    collate_fn_default = partial(collate_fn, max_seq_len=max_seq_len, tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  collate_fn=collate_fn_default, num_workers=num_workers)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size,
                                 collate_fn=collate_fn_default, num_workers=num_workers)
    return train_dataloader, eval_dataloader
