# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LiLT/Fine_tune_LiLT_on_a_custom_dataset%2C_in_any_language.ipynb
from torch.utils.data import Dataset
from PIL import Image
import torch
from torch.utils.data import DataLoader
import pandas as pd
def split_df(df, test_size, shuffle, seed, stratify):
    y_label = df["label"] if stratify else None
    df_train, df_test = train_test_split(df, test_size=test_size,
                                         shuffle=shuffle, random_state=seed, stratify=y_label)
    return df_train, df_test
def load_train_eval_df(df_path):
    df = pd.read_csv(df_path)
    df_train, df_val = split_df(df, test_size, shuffle, seed, stratify)
    print('Train: ', len(df_train))
    print('Val: ', len(df_val))



def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]


class CustomDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get item
        example = self.df[idx]
        image = example["img_path"]
        words = example["words"]
        boxes = example["original_bboxes"]
        ner_tags = example["ner_tags"]

        # prepare for the model
        width, height = image.size

        bbox = []
        labels = []
        for word, box, label in zip(words, boxes, ner_tags):
            box = normalize_bbox(box, width, height)
            n_word_tokens = len(self.tokenizer.tokenize(word))
            bbox.extend([box] * n_word_tokens)
            labels.extend([label] + ([-100] * (n_word_tokens - 1)))

        cls_box = sep_box = [0, 0, 0, 0]
        bbox = [cls_box] + bbox + [sep_box]
        labels = [-100] + labels + [-100]

        encoding = self.tokenizer(" ".join(words), truncation=True, max_length=512)
        sequence_length = len(encoding.input_ids)
        # truncate boxes and labels based on length of input ids
        labels = labels[:sequence_length]
        bbox = bbox[:sequence_length]

        encoding["bbox"] = bbox
        encoding["labels"] = labels

        return encoding


def collate_fn(features, tokenizer):
    boxes = [feature["bbox"] for feature in features]
    labels = [feature["labels"] for feature in features]
    # use tokenizer to pad input_ids
    batch = tokenizer.pad(features, padding="max_length", max_length=512)

    sequence_length = torch.tensor(batch["input_ids"]).shape[1]
    batch["labels"] = [labels_example + [-100] * (sequence_length - len(labels_example)) for labels_example in labels]
    batch["bbox"] = [boxes_example + [[0, 0, 0, 0]] * (sequence_length - len(boxes_example)) for boxes_example in boxes]

    # convert to PyTorch
    # batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}
    batch = {k: torch.tensor(v) for k, v in batch.items()}

    return batch


def load_data(df_path):
    train_df, eval_df = load_train_eval_df(df_path)
    tokenizer = ... #TODO
    train_dataset = CustomDataset(train_df, tokenizer)
    eval_dataset = CustomDataset(eval_df, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    return train_dataloader, eval_dataloader