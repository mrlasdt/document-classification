from pathlib import Path
import pandas as pd
import os
import argparse

from src.tools.utils import construct_file_path
from externals.ocr_sdsv import ImageReader
image_extensions = ImageReader.supported_ext


def get_args():
    parser = argparse.ArgumentParser()
    # parser image
    parser.add_argument("--img_dir", type=str, required=True, help="path to image data directory")
    parser.add_argument("--ocr_dir", type=str, required=True, help="path to ocr text directory")
    parser.add_argument("--out_file", type=str, required=True, help="path of file to save the dataframe")
    opt = parser.parse_args()
    return opt


def create_df_dataloader_from_data_dir(data_dir: str, ocr_dir: str):
    data_dir = Path(data_dir)
    ddata = {
        "img_path": list(),
        "ocr_path": list(),
        "label": list(),
    }
    for d in data_dir.glob('*'):
        print(d)
        for f in d.glob('*'):
            if f.is_file() and f.suffix.lower() in image_extensions:
                print(f)
                ddata["label"].append(d.name)
                ddata["img_path"].append(str(f.as_posix()))
                ddata["ocr_path"].append(construct_file_path(os.path.join(ocr_dir, d.name), f.name, ext=".txt"))
    df = pd.DataFrame.from_dict(ddata)
    return df


if __name__ == "__main__":
    # data_dir = "data/FWD/33forms_IMG"
    # ocr_dir = "results/ocr/FWD/33forms"
    # save_dir = "data/"

    opt = get_args()
    df = create_df_dataloader_from_data_dir(opt.img_dir, opt.ocr_dir)
    df.to_csv(f"{opt.out_file}", index=False)
