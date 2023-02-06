"""
see scripts/run_ocr.sh to run     
"""
# from pathlib import Path  # add parent path to run debugger
# import sys
# FILE = Path(__file__).absolute()
# sys.path.append(FILE.parents[2].as_posix())

from externals.ocr.ocr_yolox import OcrEngineForYoloX
import argparse
import tqdm
import pandas as pd
from externals.ocr.word_formation import *
from pathlib import Path
from externals.ocr.utils import construct_file_path
PARENT_DIR_LEVEL = 2
IMG_EXTS = ['.jpg', ".png", ".jpeg"]


def sort_bboxes_and_words(lbboxes, lwords) -> tuple[list, list]:
    lWords = [Word(text=word, bndbox=bbox) for word, bbox in zip(lwords, lbboxes)]
    list_lines, _ = words_to_lines(lWords)
    lwords_ = list()
    lbboxes_ = list()
    # # TEMP
    # f = open("test.txt", "w+", encoding="utf-8")
    # for line in list_lines:
    #     f.write("{}\n".format(line.text))
    # f.close()
    # ##
    for line in list_lines:
        for word_group in line.list_word_groups:
            for word in word_group.list_words:
                lwords_.append(word.text)
                lbboxes_.append(word.boundingbox)
    return lbboxes_, lwords_


def get_args():
    parser = argparse.ArgumentParser()
    # parser image
    parser.add_argument("--image", type=str, help="image path", required=True)
    parser.add_argument("--det_cfg", type=str, required=True)
    parser.add_argument("--det_ckpt", type=str, required=True)
    parser.add_argument("--cls_cfg", type=str, required=True)
    parser.add_argument("--cls_ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    opt = parser.parse_args()
    return opt


def load_engine(opt):
    print("[INFO] Loading engine...")
    print(opt.det_cfg)
    print(opt.det_ckpt)
    print(opt.cls_cfg)
    print(opt.cls_ckpt)
    engine = OcrEngineForYoloX(opt.det_cfg, opt.det_ckpt, opt.cls_cfg, opt.cls_ckpt)
    print("[INFO] Engine loaded")
    return engine


def convert_relative_path_to_positive_path(dir):
    dir = Path(dir)
    script_dir = Path(__file__).absolute().parents[PARENT_DIR_LEVEL]
    return dir if dir.is_absolute() else script_dir.joinpath(dir)


def prepare_dirs(opt) -> tuple[Path, Path]:
    input_image = convert_relative_path_to_positive_path(opt.image)
    save_dir = convert_relative_path_to_positive_path(opt.save_dir)
    save_dir.mkdir(exist_ok=True)
    print("[INFO]: Creating folder ", save_dir)
    return input_image, save_dir


def write_to_file(lbboxes, lwords, save_path):
    f = open(save_path, "w+", encoding="utf-8")
    for bbox, text in zip(lbboxes, lwords):
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        f.write("{}\t{}\t{}\t{}\t{}\n".format(xmin, ymin, xmax, ymax, text))
    f.close()


def process_img(img_path, save_dir):
    try:
        lbboxes, lwords = engine.inference(str(img_path))
        lbboxes, lwords = sort_bboxes_and_words(lbboxes, lwords)
    except AssertionError as e:
        print('[ERROR]: ', e, " at ", img_path)
        return None
    save_path = save_dir.joinpath(img_path.stem + ".txt")
    write_to_file(lbboxes, lwords, save_path)


def process_dir(dir_path: Path, save_dir):
    for img_path in tqdm.tqdm(dir_path.iterdir()):
        if img_path.is_dir():
            # save_dir_sub = save_dir.joinpath(img_path.stem)
            save_dir_sub = Path(construct_file_path(save_dir, img_path, ext=""))
            save_dir_sub.mkdir(exist_ok=True)
            process_dir(img_path, save_dir_sub)
        elif img_path.suffix in IMG_EXTS:
            process_img(img_path, save_dir)


if __name__ == "__main__":
    opt = get_args()
    engine = load_engine(opt)
    input_image, save_dir = prepare_dirs(opt)
    if input_image.is_dir():
        process_dir(input_image, save_dir)
    elif input_image.suffix in IMG_EXTS:
        process_img(input_image, save_dir)
    elif input_image.suffix == '.csv':
        df = pd.read_csv(input_image)
        assert 'image_path' in df.columns, 'Cannot found image_path in df headers'
        list_file = list(df.image_path.values)
        for img_path in list_file:
            process_img(Path(img_path), save_dir)
    else:
        raise NotImplementedError('[ERROR]: Unsupported file {}'.format(input_image))

if __name__ == "__main__":
    # %%
    from pathlib import Path
    d = Path("/mnt/ssd500/hungbnt/DocumentClassification/data/FWD/33forms_IMG/4. FATCA_W_BEN_E_Simplifier_combined with UBO form_VNM_Clean 2022")
    d.suffix
# %%
