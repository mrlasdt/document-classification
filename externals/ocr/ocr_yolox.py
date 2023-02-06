# temp #for debug
import sys
sys.path.append("/home/sds/thucpd/develop_OCR/TextDetectionApi/components/mmdetection")
sys.path.append("/home/sds/datnt/mmocr")
from pathlib import Path  # add parent path to run debugger
#import sys
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[2].as_posix())
# import time
# now = time.time()
from mmdet.apis import async_inference_detector, inference_detector, init_detector
# print('1', time.time() - now)
from mmocr.apis import init_detector as init_classifier
# print('2', time.time() - now)
from mmocr.apis.inference import model_inference
# print('3', time.time() - now)
from config import config as cfg
import numpy as np
from externals.ocr.utils import read_image_file, get_crop_img_and_bbox
# print('4', time.time() - now)


def flatten(l):
    """ 
    https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
    """
    return [item for sublist in l for item in sublist]


def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class YoloX():
    def __init__(self, config, checkpoint):
        self.model = init_detector(config, checkpoint, cfg.DEVICE)

    def inference(self, img):
        return inference_detector(self.model, img)


class Classifier_SATRN:
    def __init__(self, config, checkpoint):
        self.model = init_classifier(config, checkpoint, cfg.DEVICE)

    def inference(self, numpy_image):
        result = model_inference(self.model, numpy_image, batch_mode=True)
        preds_str = [r["text"] for r in result]
        confidence = [r["score"] for r in result]
        return preds_str, confidence


class OcrEngineForYoloX():
    def __init__(self, det_cfg, det_ckpt, cls_cfg, cls_ckpt):
        self.det = YoloX(det_cfg, det_ckpt)
        self.cls = Classifier_SATRN(cls_cfg, cls_ckpt)

    def single_det_to_lbboxes_and_lwords(self, img, pred_det):
        bboxes = np.vstack(pred_det)
        lbboxes = []
        lcropped_img = []
        if len(bboxes) == 0:  # no bbox found
            return [], []
        for bbox in bboxes:
            try:
                crop_img, bbox_ = get_crop_img_and_bbox(img, bbox, extend=True)
                lbboxes.append(bbox_)
                lcropped_img.append(crop_img)
            except AssertionError:
                print(f'[ERROR]: Skipping invalid bbox {bbox} in ', img)
        lwords, _ = self.cls.inference(lcropped_img)
        return lbboxes, lwords

    def inference(self, img, batch_mode=False, batch_size=16):
        """
        Accept image path or ndarray or list of them, return ocr result or list of them
        """
        if not batch_mode:
            if isinstance(img, str):
                img = read_image_file(img)
            pred_det = self.det.inference(img)
            return self.single_det_to_lbboxes_and_lwords(img, pred_det)
        else:
            if not isinstance(img, list):
                return self.inference(img, batch_mode=False)
            lllbboxes, lllwords = [], []
            for imgs in chunks(img, batch_size):  # chunks to reduce memory footprint
                if isinstance(imgs[0], str):
                    imgs = [read_image_file(img) for img in imgs]
                pred_dets = self.det.inference(imgs)
                llbboxes, llwords = [], []
                for img, pred_det in zip(imgs, pred_dets):
                    lbboxes, lwords = self.single_det_to_lbboxes_and_lwords(img, pred_det)
                    llbboxes.append(lbboxes)
                    llwords.append(lwords)
                lllbboxes.append(llbboxes)
                lllwords.append(llwords)
            return flatten(lllbboxes), flatten(lllwords)


if __name__ == "__main__":
    DET_CFG = "/home/sds/datnt/mmdetection/logs/textdet-fwd/yolox_s_8x8_300e_cocotext_1280.py"
    DET_CKPT = "/home/sds/datnt/mmdetection/logs/textdet-fwd/best_bbox_mAP_epoch_100.pth"
    CLS_CFG = "/home/sds/datnt/mmocr/logs/satrn_big_2022-10-31/satrn_big.py"
    CLS_CKPT = "/home/sds/datnt/mmocr/logs/satrn_big_2022-10-31/best.pth"
    imgs = [
        "/mnt/ssd500/hungbnt/DocumentClassification/data/FWD/POS01/1_PDFsam_Scan.pdf.jpg",
        "/mnt/ssd500/hungbnt/DocumentClassification/data/FWD/POS01/1.pdf.jpg",
        "/mnt/ssd500/hungbnt/DocumentClassification/data/FWD/POS01/2.pdf.jpg",
    ]
    engine = OcrEngineForYoloX(DET_CFG, DET_CKPT, CLS_CFG, CLS_CKPT)
    print(engine.inference(imgs, batch_mode=True))
    