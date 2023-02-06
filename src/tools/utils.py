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


def get_name(file_path, ext: bool = True):
    file_path_ = os.path.basename(file_path)
    return file_path_ if ext else os.path.splitext(file_path_)[0]


def construct_file_path(dir, file_path, ext=''):
    '''
    args:
        dir: /path/to/dir
        file_path /example_path/to/file.txt
        ext = '.json'
    return 
        /path/to/dir/file.json
    '''
    return os.path.join(
        dir, get_name(file_path,
                      True)) if ext == '' else os.path.join(
        dir, get_name(file_path,
                      False)) + ext


