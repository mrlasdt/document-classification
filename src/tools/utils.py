import os
import json


def read_txt_lines(file_path):
    with open(file_path, 'r') as fp:
        lines = fp.read().splitlines()
    return lines


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    return data


def split_file_name(file_path, ext: bool = True):
    file_path_ = os.path.basename(file_path)
    return file_path_ if ext else os.path.splitext(file_path_)[0]

# def split_file_name(file_path: str, parent_level: int = -1, with_extension: bool = True):
#     path_components = os.path.abspath(file_path).split(os.path.sep)
#     file_name = path_components[-1]
#     if not with_extension:
#         file_name = os.path.splitext(file_name)[0]
#     path_components[-1] = file_name
#     return os.path.sep.join(path_components[:-parent_level])


def construct_file_path(dir: str, file_path: str, parent_level: int = -1, ext: str = '') -> str:
    '''
    args:
        dir: /path/to/dir
        file_path /example_path/to/file.txt
        ext = '.json'
    return 
        /path/to/dir/file.json
    '''
    if not ext:
        # return os.path.join(dir, split_file_name(file_path, parent_level, True))
        return os.path.join(dir, split_file_name(file_path, True))
    else:
        # return os.path.join(dir, split_file_name(file_path, parent_level, False)) + ext
        return os.path.join(dir, split_file_name(file_path, False)) + ext


# %%
if __name__ == "__main__":
    # %%
    construct_file_path(
        "/mnt/ssd500/hungbnt/DocumentClassification/results/ocr/Samsung/DCYCBH/1b36d9f280ef58b101fe7.txt")
