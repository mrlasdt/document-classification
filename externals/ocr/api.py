"""
see scripts/run_ocr.sh to run     
"""
# from pathlib import Path  # add parent path to run debugger
# import sys
# FILE = Path(__file__).absolute()
# sys.path.append(FILE.parents[2].as_posix())  

from externals.ocr.ocr_yolox import OcrEngineForYoloX
from glob import glob
import os
import argparse
import tqdm
import pandas as pd
from externals.ocr.word_formation import *



def sort_bboxes_and_words(lbboxes, lwords)->tuple[list, list]:
    lWords = [Word(text=word, bndbox = bbox) for word, bbox in zip(lwords, lbboxes)]
    list_lines, _ = words_to_lines(lWords)
    lwords_ = list()
    lbboxes_ = list()
    ## TEMP
    f = open("test.txt","w+", encoding="utf-8")
    for line in list_lines:
        f.write("{}\n".format(line.text))
    f.close()
    ##
    for line in list_lines:
        for word_group in line.list_word_groups:
            for word in word_group.list_words:
                lwords_.append(word.text)
                lbboxes_.append(word.boundingbox)
    return lbboxes_, lwords_

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser image
    parser.add_argument("--image", type=str, help="image path", required=True)
    parser.add_argument("--det_cfg", type= str, required=True)
    parser.add_argument("--det_ckpt", type= str, required=True)
    parser.add_argument("--cls_cfg", type = str, required=True)
    parser.add_argument("--cls_ckpt", type = str, required=True)
    parser.add_argument("--save_dir", type = str, required=True)
    opt = parser.parse_args()
    if not os.path.isdir(opt.save_dir):
        os.mkdir(opt.save_dir)
        print("[INFO]: Creating folder...", opt.save_dir)

    print("[INFO] Loading engine...")
    print(opt.det_cfg)
    print(opt.det_ckpt)
    print(opt.cls_cfg)
    print(opt.cls_ckpt)
    engine = OcrEngineForYoloX(opt.det_cfg, opt.det_ckpt,  opt.cls_cfg, opt.cls_ckpt)    
    print("[INFO] Engine loaded")
    if os.path.isdir(opt.image):
        list_file = sorted(glob(opt.image+"/*.jpg", recursive=True))
    elif opt.image.endswith('.jpg'):
        list_file = [opt.image]
    elif opt.image.endswith('.csv'):
        df = pd.read_csv(opt.image)
        assert 'image_path' in df.columns, 'Cannot found image_path in df headers'
        list_file = list(df.image_path.values)
    else:
        raise NotImplementedError('Invalid --image arg')

    list_text=[]
    image_names=[]
    # list_val = [os.path.basename(img_path) for img_path in glob('/home/sds/hungbnt/KIE_pretrained/data/GPLX/val/crop_blx_5_10_2022/*.jpg')]
    cnt = -1
    for img_path in tqdm.tqdm(list_file):
        # cnt+=1
        # if cnt==50:#test with cpu #TODO: REmove this after test
        #     break
        # if not os.path.basename(img_path) in list_val:
        #     continue
        pseudo_label_path = os.path.join(f"{opt.save_dir}",os.path.basename(img_path[:-3])+"txt")    
        try:
            lbboxes, lwords = engine.inference(img_path)
        except AssertionError as e:
            print('[ERROR]: ', e, " at ", img_path)
            continue
        # lbbox_words = [(bbox, word) for bbox, word in zip(lbboxes, lwords)]
        # lbbox_words.sort(key = lambda x: (x[0][1], x[0][0]))
        # f = open(pseudo_label_path,"w+", encoding="utf-8")
        # for bbox_word in lbbox_words:
        #     bbox, word = bbox_word
        #     xmin, ymin, xmax, ymax = bbox
        #     f.write("{}\t{}\t{}\t{}\t{}\n".format(xmin,ymin,xmax,ymax,word))
        # f.close()

            
        f = open(pseudo_label_path,"w+", encoding="utf-8")
        lbboxes, lwords = sort_bboxes_and_words(lbboxes, lwords)
        for bbox, text in zip(lbboxes, lwords):
            xmin, ymin, xmax, ymax = bbox[0],bbox[1],bbox[2],bbox[3]
            f.write("{}\t{}\t{}\t{}\t{}\n".format(xmin,ymin,xmax,ymax,text))
        f.close()

