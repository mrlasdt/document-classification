from src.tools.utils import construct_file_path
from pathlib import Path
import pandas as pd
import os

image_extensions = (".jpg", ".jpeg", ".png", ".bmp")  # Add any other extensions you want to include


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
    # data_dir = "/mnt/ssd500/hungbnt/DocumentClassification/data/FWD/33forms_IMG"
    # ocr_dir = "/mnt/ssd500/hungbnt/DocumentClassification/results/ocr/FWD/33forms"
    # save_dir = "/mnt/ssd500/hungbnt/DocumentClassification/data/"
    data_dir = "/mnt/hdd2T/AICR/Projects/2023/FWD/Forms/Images/02_2023/"
    ocr_dir = "/mnt/ssd500/hungbnt/DocumentClassification/results/ocr/FWD/202302_3forms"
    save_dir = "/mnt/ssd500/hungbnt/DocumentClassification/data/"
    df = create_df_dataloader_from_data_dir(data_dir, ocr_dir)
    df.to_csv(f"{save_dir}/202302_3forms.csv", index=False)
