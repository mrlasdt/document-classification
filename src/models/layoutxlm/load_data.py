from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import cv2
from externals.ocr.utils import read_ocr_result_from_txt
from transformers import LayoutXLMTokenizer, LayoutLMv2FeatureExtractor, LayoutXLMProcessor
from PIL import Image
from functools import partial
import torch


def split_df(df, test_size, shuffle, seed, stratify):
    y_label = df["label"] if stratify else None
    df_train, df_test = train_test_split(df, test_size=test_size,
                                         shuffle=shuffle, random_state=seed, stratify=y_label)
    return df_train, df_test


def _normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def load_ocr_result(example):
    r"""Load OCR labels, i.e. (word, bbox) pairs, into the input DataFrame containing of columns (image_path, ocr_path, label)"""
    ocr_path = example['ocr_path']
    image_path = example["img_path"]
    assert os.path.exists(ocr_path), ocr_path
    assert os.path.exists(image_path), image_path
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    lbboxes, lwords = read_ocr_result_from_txt(ocr_path)
    lbboxes = [_normalize_box(bbox, w, h) for bbox in lbboxes]
    example["words"] = lwords
    example["bbox"] = lbboxes
    return example


def preprocess_data(examples, processor, max_seq_length, labels):
    # Add custom OCR here
    words = examples['words']
    normalized_word_boxes = examples['bbox']
    label2idx = {labels[i]: i for i in range(len(labels))}

    assert all([
        len(_words) == len(boxes)
        for _words, boxes in zip(words, normalized_word_boxes)
    ])

    # Process examples
    images = [Image.open(path).convert("RGB") for path in examples['img_path']]
    encoded_inputs = processor(
        images, padding="max_length", truncation=True, text=words,
        boxes=normalized_word_boxes, max_length=max_seq_length,
    )

    encoded_inputs["labels"] = [label2idx[label] for label in examples["label"]]
    return encoded_inputs


def load_data(
        df_path, pretrained_tokenizer_path, labels, image_shape, max_seq_len,
        batch_size, test_size, shuffle, seed, stratify, num_workers):
    df = pd.read_csv(df_path)
    df_train, df_val = split_df(df, test_size, shuffle, seed, stratify) #image_path, ocr_path, label
    print('Train: ', len(df_train))
    print('Val: ', len(df_val))
    train_dataset = Dataset.from_pandas(df_train)
    val_dataset = Dataset.from_pandas(df_val)

    print("Loading OCR result ...")
    train_dataset = train_dataset.map(lambda example: load_ocr_result(example))
    val_dataset = val_dataset.map(lambda example: load_ocr_result(example))

    print("Preparing dataset ...")
    features = Features({
        'image': Array3D(dtype="int64", shape=image_shape),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(max_seq_len, 4)),  # 4=xywh
        'labels': ClassLabel(num_classes=len(labels), names=labels),
    })
    # processor = LayoutXLMProcessor.from_pretrained(pretrained_processor)
    tokenizer = LayoutXLMTokenizer.from_pretrained(pretrained_tokenizer_path)
    feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
    processor = LayoutXLMProcessor(feature_extractor, tokenizer)
    preprocess_data_default = partial(preprocess_data, max_seq_length=max_seq_len, processor=processor, labels=labels)
    train_dataset = train_dataset.map(
        preprocess_data_default, remove_columns=train_dataset.column_names, features=features,
        batched=True, batch_size=batch_size
    )
    train_dataset.set_format(type="torch")

    val_dataset = val_dataset.map(
        preprocess_data_default, remove_columns=val_dataset.column_names, features=features,
        batched=True, batch_size=batch_size
    )
    val_dataset.set_format(type="torch")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)  # ALREADY SHUFFLE
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    return train_dataloader, val_dataloader
