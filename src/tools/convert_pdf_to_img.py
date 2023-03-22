# %%
from pathlib import Path
from pdf2image import convert_from_path
from tqdm import tqdm
import shutil
DATA_DIR = Path("/mnt/hdd2T/AICR/Projects/2023/FWD/Forms/PDFs/01_Classified_forms/")
SAVE_DIR = Path("/mnt/hdd2T/AICR/Projects/2023/FWD/Forms/Images/01_Classified_forms/")
# %%
# def process_dir(data_dir, save_dir):
#     if not save_dir.is_dir():
#         save_dir.mkdir()
#         print("[INFO]: Creating ", save_dir)
#     for f in data_dir.glob('*.pdf'):
#         save_dir_sub = save_dir.joinpath(f.stem)
#         if not save_dir_sub.is_dir():
#             save_dir_sub.mkdir()
#             print("[INFO]: Creating ", save_dir_sub)
#         images = convert_from_path(str(f), grayscale=False)
#         # for i in tqdm(range(len(images))):
#         # images[i].save(str(SAVE_DIR_SUB.joinpath(f.stem + f'_{i}.jpg')), 'JPEG')  # save only the first pages
#         images[0].save(str(save_dir_sub.joinpath(f.stem + f'_{0}.jpg')), 'JPEG')


def process_dir(data_dir, save_dir):
    for f in data_dir.glob('*.pdf'):
        # if not save_dir_sub.is_dir():
        #     save_dir_sub.mkdir()
        #     print("[INFO]: Creating ", save_dir_sub)
        images = convert_from_path(str(f), grayscale=False)
        # for i in tqdm(range(len(images))):
        # images[i].save(str(SAVE_DIR_SUB.joinpath(f.stem + f'_{i}.jpg')), 'JPEG')  # save only the first pages
        images[0].save(f"{str(save_dir)}_{f.stem}_{0}.jpg".replace(" ", "_"), 'JPEG')
    for f in data_dir.glob('*.jpg'):
        # if not save_dir_sub.is_dir():
        #     save_dir_sub.mkdir()
        #     print("[INFO]: Creating ", save_dir_sub)

        # for i in tqdm(range(len(images))):
        # images[i].save(str(SAVE_DIR_SUB.joinpath(f.stem + f'_{i}.jpg')), 'JPEG')  # save only the first pages
        # images[0].save(f"{str(save_dir)}_{f.stem}_{0}.jpg", 'JPEG')
        shutil.copy(f, f"{str(save_dir)}_{f.stem}.jpg".replace(" ", "_"))


# %%
if __name__ == "__main__":
    # %% convert pdf to images
    for data_sub_dir in DATA_DIR.iterdir():
        print(data_sub_dir)
        save_sub_dir = SAVE_DIR.joinpath(data_sub_dir.stem)
        process_dir(data_sub_dir, save_sub_dir)

# %%
