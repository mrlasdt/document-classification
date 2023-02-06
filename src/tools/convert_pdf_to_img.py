from pathlib import Path
from pdf2image import convert_from_path
from tqdm import tqdm
DATA_DIR = Path("/mnt/ssd500/hungbnt/DocumentClassification/data/FWD/33forms_PDF")
SAVE_DIR = Path("/mnt/ssd500/hungbnt/DocumentClassification/data/FWD/33forms_IMG")


if __name__ == "__main__":
    # %% convert pdf to images
    if not SAVE_DIR.is_dir():
        SAVE_DIR.mkdir()
        print("[INFO]: Creating ", SAVE_DIR)
    for f in DATA_DIR.glob('*.pdf'):
        SAVE_DIR_SUB = SAVE_DIR.joinpath(f.stem)
        if not SAVE_DIR_SUB.is_dir():
            SAVE_DIR_SUB.mkdir()
            print("[INFO]: Creating ", SAVE_DIR_SUB)
        images = convert_from_path(str(f), grayscale=False)
        # for i in tqdm(range(len(images))):
        # images[i].save(str(SAVE_DIR_SUB.joinpath(f.stem + f'_{i}.jpg')), 'JPEG')  # save only the first pages
        images[0].save(str(SAVE_DIR_SUB.joinpath(f.stem + f'_{0}.jpg')), 'JPEG')
