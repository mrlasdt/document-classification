# %%
# import os
# os.environ["JUPYTER_PATH"] = "/home/sds/thucpd/develop_OCR/TextDetectionApi/components/mmdetection:/home/sds/datnt/mmocr"
# os.environ["PYTHONPATH"] = "/home/sds/thucpd/develop_OCR/TextDetectionApi/components/mmdetection:/home/sds/datnt/mmocr"
from tqdm import tqdm
from pathlib import Path
from pdf2image import convert_from_path
from externals.ocr.ocr_yolox import OcrEngineForYoloX
from externals.ocr.api import sort_bboxes_and_words
import numpy as np
from typing import Dict, List, Tuple, Union
import re
# intentionally use from external to unified with how ocr module read image
from externals.ocr.utils import read_image_file, read_ocr_result_from_txt
from collections import defaultdict
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score
# INPUT_DIR = Path("/mnt/ssd500/hungbnt/DocumentClassification/data/Sample_input/Case_1_tach_roi-toan-bo")
# SAVE_DIR = Path("/mnt/ssd500/hungbnt/DocumentClassification/results").joinpath(INPUT_DIR.name)

DET_CFG = "/home/sds/datnt/mmdetection/logs/textdet-fwd/yolox_s_8x8_300e_cocotext_1280.py"
DET_CKPT = "/home/sds/datnt/mmdetection/logs/textdet-fwd/best_bbox_mAP_epoch_100.pth"
CLS_CFG = "/home/sds/datnt/mmocr/logs/satrn_big_2022-10-31/satrn_big.py"
CLS_CKPT = "/home/sds/datnt/mmocr/logs/satrn_big_2022-10-31/best.pth"

DDOC_LABELS_TO_TITLE = {
    "POS01": "Phiếu Yêu Cầu Điều Chỉnh Thông Tin Cá Nhân",
    "POS02": "Phiếu Yêu Cầu Điều Chỉnh Sản Phẩm Bảo Hiểm",
    "POS03": "Phiếu Yêu Cầu Điều Chỉnh Thông Tin Hợp Đồng",
    "POS04": "Phiếu Yêu Cầu Khôi Phục Hiệu Lực Hợp Đồng",
    "POS05": "Phiếu Yêu Cầu Thanh Toán",
    "POS06": "Phiếu Yêu Cầu Điều Chỉnh Dành Cho Nghiệp Vụ Hợp Đồng Bảo Hiểm Liên Kết Đơn Vị",
    "POS08": "Thông Báo Đi Nước Ngoài",
    "CCCD_front": "CĂN CƯỚC CÔNG DÂN",
    "CCCD_back": "Đặc điểm nhân dạng CỤC TRƯỞNG CỤC",
    "CMND_front": "CHỨNG MINH NHÂN DÂN",
    "CMND_back": "Tôn giáo DẤU VẾT RIÊNG VÀ DỊ HÌNH",
    "DXN102": "ĐƠN XIN XÁC NHẬN HAI NGƯỜI LÀ MỘT",
    "BIRTH_CERT": "GIẤY KHAI SINH",
}
ACCEPTED_EXT = [".pdf", ".png", ".jpg"]
OTHERS_LABEL = "OTHERS"
DDOC_LABELS_TO_NO_PAGES = {
    "POS01": 2,
    "POS02": 2,
    "POS03": 2,
    "POS04": 2,
    "POS05": 2,
    "POS06": 2,
    "POS08": 2,
    "CCCD_front": 1,
    "CCCD_back": 1,
    "CMND_front": 1,
    "CMND_back": 1,
    "DXN102": 1,
    "BIRTH_CERT": 1,
    OTHERS_LABEL: 0,
}

DDOC_LABELS_TO_IDX = {k: i for i, k in enumerate(DDOC_LABELS_TO_NO_PAGES)}
DIDX_TO_DOC_LABELS = {v: k for k, v in DDOC_LABELS_TO_IDX.items()}

CONFIG = {
    "accepted_ext": ACCEPTED_EXT,
    "ddoc_label_to_title": DDOC_LABELS_TO_TITLE,
    "ddoc_label_to_no_pages": DDOC_LABELS_TO_NO_PAGES,
    "others_label": OTHERS_LABEL
}
# %%

# def longest_common_substring(s1, s2):
#     m = len(s1)
#     n = len(s2)
#     # https://www.geeksforgeeks.org/longest-common-substring-dp-29/
#     LCSuff = [[0 for k in range(n + 1)] for l in range(m + 1)]
#     result = 0
#     for i in range(m + 1):
#         for j in range(n + 1):
#             if (i == 0 or j == 0):
#                 LCSuff[i][j] = 0
#             elif (s1[i - 1] == s2[j - 1]):
#                 LCSuff[i][j] = LCSuff[i - 1][j - 1] + 1
#                 result = max(result, LCSuff[i][j])
#             else:
#                 LCSuff[i][j] = 0
#     return result


def longestCommonSubsequence(text1: str, text2: str) -> int:
    # https://leetcode.com/problems/longest-common-subsequence/discuss/351689/JavaPython-3-Two-DP-codes-of-O(mn)-and-O(min(m-n))-spaces-w-picture-and-analysis
    dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]
    for i, c in enumerate(text1):
        for j, d in enumerate(text2):
            dp[i + 1][j + 1] = 1 + dp[i][j] if c == d else max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1]


class RuleBaseDocClassifier:
    cfg = CONFIG  # static

    def __init__(self, ocr_engine: OcrEngineForYoloX):
        self.ocr_engine = ocr_engine

    @staticmethod
    def read_from_dir(dir_path: str) -> Dict[str, List[np.ndarray]]:
        dir_ = Path(dir_path)
        assert dir_.is_dir(), "Not a directory"
        res = dict()
        for f in dir_.glob("*"):
            if f.suffix in RuleBaseDocClassifier.cfg["accepted_ext"]:
                res[f.name] = RuleBaseDocClassifier.read_from_file(f)
        return res

    @staticmethod
    def read_from_file(file_path: str) -> List[np.ndarray]:
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        if file_path.suffix not in RuleBaseDocClassifier.cfg["accepted_ext"]:
            raise NotImplementedError("{} is not supported".format(file_path.suffix))
        if file_path.suffix == ".pdf":
            return RuleBaseDocClassifier.read_from_pdf(str(file_path))
        elif file_path.suffix in [".png", ".jpg"]:
            return RuleBaseDocClassifier.read_from_image(str(file_path))
        else:
            raise NotImplementedError("{} is not supported".format(file_path.suffix))

    @staticmethod
    def read_from_pdf(pdf_path: str) -> List[np.ndarray]:
        return [np.array(img) for img in convert_from_path(pdf_path)]

    @staticmethod
    def read_from_image(img_path: str) -> List[np.ndarray]:
        return [read_image_file(img_path)]

    def run_ocr(self, img: np.ndarray, max_length=-1, batch_mode=False, batch_size=16) -> List[Tuple[Tuple, str]]:
        '''
        return list of (bbox, text)
        '''
        lbboxes, lwords = self.ocr_engine.inference(img, batch_mode=batch_mode, batch_size=batch_size)
        lbboxes, lwords = sort_bboxes_and_words(lbboxes, lwords)
        return lbboxes[:max_length], lwords[:max_length]

    @staticmethod
    def classify_by_template_number(lwords: List[str], max_length: int) -> str:
        # TODO: add confident score of word and bbox to return value
        # TODO: valid assumption that there is only 1 template number in a page, currently return first occurence
        ocr_str = "".join(lwords[:max_length])
        match = re.search(r"POS0(?P<cls>\d{1})_20", ocr_str)
        return "POS0{}".format(match["cls"]) if match else -1
        # return -1  # to test classify_by_title
        # for word in lwords: #maybe match each word will be more efficient?
        #     match = re.search(r"POS0(?P<cls>\d{1})_20", word)
        # return "POS0{}".format(match["cls"]) if match else -1
        # return -1

    @staticmethod
    def classify_by_title(lwords: List[str], threshold: float, max_length: int) -> str:
        ocr_str = "".join(lwords[:max_length])
        for cls_, title in RuleBaseDocClassifier.cfg["ddoc_label_to_title"].items():
            title = title.replace(" ", "")
            if longestCommonSubsequence(title, ocr_str) / len(title) > threshold:
                return cls_
        return -1

    @staticmethod
    def classify(lwords: List[str], threshold=0.85, max_length=50) -> int:
        """
        threshold: threshold of longest common subsequence to match with title
        max_length: number of first words extracted from page to compare with title
        """
        # TODO: implement ensemble classifier
        cls_ = RuleBaseDocClassifier.classify_by_template_number(lwords, max_length)
        if cls_ == -1:
            cls_ = RuleBaseDocClassifier.classify_by_title(lwords, threshold, max_length)
        return RuleBaseDocClassifier.cfg["others_label"] if cls_ == -1 else cls_

    def predict(self, inp: Union[str, np.ndarray, List[str], List[np.ndarray]], batch_mode=False, batch_size=16) -> str:
        """
        Accept image path or ndarray or list of them, return class
        """
        _lbboxes, lwords = self.run_ocr(inp, batch_mode=batch_mode, batch_size=batch_size)
        cls_ = self.classify(lwords) if not batch_mode else [self.classify(l) for l in lwords]
        return cls_

    def infer(self, dir_path: str):
        dinput = self.read_from_dir(dir_path)
        dout = dict()
        for file_name, limages in dinput.items():
            dout[file_name] = defaultdict(list)
            curr_first_page_of_doc_idx = 0
            curr_cls_of_doc = None
            for i, img in enumerate(limages):
                # skip this page if it is not the first page of document
                if i != curr_first_page_of_doc_idx and i < curr_first_page_of_doc_idx + self.cfg["ddoc_label_to_no_pages"][curr_cls_of_doc]:
                    dout[file_name][curr_cls_of_doc].append(i)
                    continue
                cls_ = self.predict(img, batch_mode=False)
                dout[file_name][cls_].append(i)
                curr_first_page_of_doc_idx = i
                curr_cls_of_doc = cls_
                break
        return dout

    @ staticmethod
    def eval(df_path: str, threshold=0.85, max_length=50):
        df = pd.read_csv(df_path)
        y_true = [DDOC_LABELS_TO_IDX[label] for label in df["label"]]
        y_pred = []
        for i, ocr_path in tqdm(enumerate(df["ocr_path"])):
            _lbboxes, lwords = read_ocr_result_from_txt(ocr_path)
            pred = DDOC_LABELS_TO_IDX[RuleBaseDocClassifier.classify(lwords, threshold, max_length)]
            y_pred.append(pred)
            if pred != y_true[i]:
                # RuleBaseDocClassifier.classify(lwords, threshold, max_length)  # for debugging
                print("*" * 100)
                print(df["img_path"].iloc[i])
                print(ocr_path)
                print(y_true[i], pred)
        print(classification_report(y_true, y_pred))
        return y_true, y_pred


# %%
if __name__ == "__main__":
    # %%
    engine = OcrEngineForYoloX(DET_CFG, DET_CKPT, CLS_CFG, CLS_CKPT)
    cls_model = RuleBaseDocClassifier(engine)
    print("Done init")
    # cls_model.infer("data/Sample_input/Case_2_ghep_toan_bo/")  # OK
    cls_model.infer("/mnt/ssd500/hungbnt/DocumentClassification/data")  # OK

    # %%
    df_path = "/mnt/ssd500/hungbnt/DocumentClassification/data/FWD.csv"
    y_true, y_pred = RuleBaseDocClassifier.eval(df_path, threshold=0.95, max_length=50)

    # %%
    print(classification_report(y_true, y_pred))
    # %%
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 13]))

    # %%
    # ocr_path = "/mnt/ssd500/hungbnt/DocumentClassification/results/ocr/POS08/50.pdf.txt"
    # ocr_path = "/mnt/ssd500/hungbnt/DocumentClassification/results/ocr/POS01/SKM_458e\ Ag22101217531.pdf.txt"
    # _lbboxes, lwords = read_ocr_result_from_txt(ocr_path)
    # RuleBaseDocClassifier.classify(lwords)
    # RuleBaseDocClassifier.classify_by_template_number(lwords)
    # RuleBaseDocClassifier.classify_by_title(lwords, 0.85, 50)
    # %%
    # cls_model.infer("data/Sample_input/Case_1_tach_roi-toan-bo/")  # OK
    # cls_model.infer("data/Sample_input/Case_2_ghep_mot_phan/")  # OK
    # cls_model.infer("data/Sample_input/Case_2_ghep_toan_bo/")  # OK

    # cls_model.infer("data/Sample_input/Case_1_tach_roi-toan-bo/CMND.pdf")
    # cls_model.infer("data/Sample_input/Case_1_tach_roi-toan-bo/Giay_khai_sinh.pdf")
    # cls_model.infer("data/Sample_input/Case_1_tach_roi-toan-bo/Giay_xac_nhan.pdf")
    # cls_model.infer("data/Sample_input/Case_1_tach_roi-toan-bo/Hoa_don.pdf")
    # cls_model.infer("/mnt/hdd2T/AICR/Projects/FWD/Phase1/Completed_forms_SAMSUNG/POS01/1_PDFsam_Scan.pdf")
    # cls_model.infer("/mnt/hdd2T/AICR/Projects/FWD/Phase1/Completed_forms_SAMSUNG/POS02/1.pdf")
    # cls_model.infer("/mnt/hdd2T/AICR/Projects/FWD/Phase1/Completed_forms_SAMSUNG/POS03/1.pdf")
    # cls_model.infer("/mnt/hdd2T/AICR/Projects/FWD/Phase1/Completed_forms_SAMSUNG/POS03/1.pdf")
    # cls_model.infer("/mnt/hdd2T/AICR/Projects/FWD/Phase1/Completed_forms_SAMSUNG/POS04/1.pdf")
    # cls_model.infer("/mnt/hdd2T/AICR/Projects/FWD/Phase1/Completed_forms_SAMSUNG/POS05/1.pdf")
    # cls_model.infer("/mnt/hdd2T/AICR/Projects/FWD/Phase1/Completed_forms_SAMSUNG/POS06/1.pdf")
    # cls_model.infer("/mnt/hdd2T/AICR/Projects/FWD/Phase1/Completed_forms_SAMSUNG/POS08/1.pdf")
