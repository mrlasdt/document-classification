from datasets import Dataset
from pathlib import Path
import pandas as pd
from src.tools.utils import construct_file_path
def load_img_and_ocr_path_to_df(data_dir:str, ocr_dir:str)->pd.DataFrame:
    ocr_paths = Path(ocr_dir).glob("*.txt")
    ocr_paths_with_existing_img = [] #accept only txt file with corresponding existing img file
    data_paths = []
    for l in ocr_paths:
        img_path = construct_file_path(data_dir, str(l), '.jpg')
        if os.path.exists(img_path):
            data_paths.append(img_path)
            ocr_paths_with_existing_img.append(l)
    df = pd.DataFrame.from_dict({'image_path': data_paths, 'label': ocr_paths_with_existing_img})
    return df

def load_data(
        train_path, val_path, train_label_path, val_label_path, max_seq_len, batch_size, pretrained_processor,
        kie_labels, device):
    train_df = load_image_paths_and_labels(train_path, train_label_path)
    val_df = load_image_paths_and_labels(val_path, val_label_path)
    print('Train: ', len(train_df))
    print('Val: ', len(val_df))
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    print("Loading OCR labels ...")
    train_dataset = train_dataset.map(lambda example: load_ocr_labels(example, "", kie_labels))
    val_dataset = val_dataset.map(lambda example: load_ocr_labels(example, "", kie_labels))

    print("Preparing dataset ...")
    features = Features({
        'image': Array3D(dtype="int64", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(max_seq_len, 4)),
        'labels': Sequence(ClassLabel(names=kie_labels))
    })
    # processor = LayoutXLMProcessor.from_pretrained(pretrained_processor)
    tokenizer = LayoutXLMTokenizer.from_pretrained(pretrained_processor, model_max_length=max_seq_len)
    feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
    processor = LayoutXLMProcessor(feature_extractor, tokenizer)
    preprocess_data_default = partial(preprocess_data, max_seq_length=max_seq_len, processor=processor)
    train_dataset = train_dataset.map(
        preprocess_data_default, remove_columns=train_dataset.column_names, features=features,
        batched=True, batch_size=batch_size
    )
    train_dataset.set_format(type="torch", device=device)

    val_dataset = val_dataset.map(
        preprocess_data_default, remove_columns=val_dataset.column_names, features=features,
        batched=True, batch_size=batch_size
    )
    val_dataset.set_format(type="torch", device=device)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader