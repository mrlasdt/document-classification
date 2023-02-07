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
import pandas as pd
from base_doc_classifier import BaseDocClasifier
# INPUT_DIR = Path("/mnt/ssd500/hungbnt/DocumentClassification/data/Sample_input/Case_1_tach_roi-toan-bo")
# SAVE_DIR = Path("/mnt/ssd500/hungbnt/DocumentClassification/results").joinpath(INPUT_DIR.name)

DET_CFG = "/home/sds/datnt/mmdetection/logs/textdet-fwd/yolox_s_8x8_300e_cocotext_1280.py"
DET_CKPT = "/home/sds/datnt/mmdetection/logs/textdet-fwd/best_bbox_mAP_epoch_100.pth"
CLS_CFG = "/home/sds/datnt/mmocr/logs/satrn_big_2022-10-31/satrn_big.py"
CLS_CKPT = "/home/sds/datnt/mmocr/logs/satrn_big_2022-10-31/best.pth"
DF_DOC_PATH = "/mnt/ssd500/hungbnt/DocumentClassification/data/Static/FWD_documents_list.xlsx"
ACCEPTED_EXT = [".pdf", ".png", ".jpg", ".jpeg"]
OTHERS_LABEL = "OTHERS"
THRESHHOLDS = [0.8, 0.95]
MAX_LENGTH = 55
OFFSET = 0.1
# DDOC_LABELS_TO_TITLE = {
#     "POS01": "Phiếu Yêu Cầu Điều Chỉnh Thông Tin Cá Nhân",
#     "POS02": "Phiếu Yêu Cầu Điều Chỉnh Sản Phẩm Bảo Hiểm",
#     "POS03": "Phiếu Yêu Cầu Điều Chỉnh Thông Tin Hợp Đồng",
#     "POS04": "Phiếu Yêu Cầu Khôi Phục Hiệu Lực Hợp Đồng",
#     "POS05": "Phiếu Yêu Cầu Thanh Toán",
#     "POS06": "Phiếu Yêu Cầu Điều Chỉnh Dành Cho Nghiệp Vụ Hợp Đồng Bảo Hiểm Liên Kết Đơn Vị",
#     "POS08": "Thông Báo Đi Nước Ngoài",
#     "CCCD_front": "CĂN CƯỚC CÔNG DÂN",
#     "CCCD_back": "Đặc điểm nhân dạng CỤC TRƯỞNG CỤC",
#     "CMND_front": "CHỨNG MINH NHÂN DÂN",
#     "CMND_back": "Tôn giáo DẤU VẾT RIÊNG VÀ DỊ HÌNH",
#     "DXN102": "ĐƠN XIN XÁC NHẬN HAI NGƯỜI LÀ MỘT",
#     "BIRTH_CERT": "GIẤY KHAI SINH",
# }

# DDOC_LABELS_TO_NO_PAGES = {
#     "POS01": 2,
#     "POS02": 2,
#     "POS03": 2,
#     "POS04": 2,
#     "POS05": 2,
#     "POS06": 2,
#     "POS08": 2,
#     "CCCD_front": 1,
#     "CCCD_back": 1,
#     "CMND_front": 1,
#     "CMND_back": 1,
#     "DXN102": 1,
#     "BIRTH_CERT": 1,
#     OTHERS_LABEL: 0,
# }

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


def longest_common_subsequence_with_idx(X, Y):
    """
    This implementation uses dynamic programming to calculate the length of the LCS, and uses a path array to keep track of the characters in the LCS. 
    The longest_common_subsequence function takes two strings as input, and returns a tuple with three values: 
    the length of the LCS, 
    the index of the first character of the LCS in the first string, 
    and the index of the last character of the LCS in the first string.
    """
    m, n = len(X), len(Y)
    L = [[0 for i in range(n + 1)] for j in range(m + 1)]

    # Following steps build L[m+1][n+1] in bottom up fashion. Note
    # that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

        # Create a string variable to store the lcs string
    lcs = L[i][j]
    # Start from the right-most-bottom-most corner and
    # one by one store characters in lcs[]
    i = m
    j = n

    while i > 0 and j > 0:

        # If current character in X[] and Y are same, then
        # current character is part of LCS
        if X[i - 1] == Y[j - 1]:

            i -= 1
            j -= 1

        # If not same, then find the larger of two and
        # go in the direction of larger value
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1

        else:
            j -= 1
    return lcs, i, i + lcs


class RuleBaseDocClassifier(BaseDocClasifier):
    def __init__(
            self, df_doc_path: str, ocr_engine: OcrEngineForYoloX = None, accepted_ext: list[str] = ACCEPTED_EXT,
            other_docid: str = OTHERS_LABEL, thresholds: float = THRESHHOLDS, max_length: int = MAX_LENGTH,
            offset: float = OFFSET):
        """ Classify document base on defined rule
        Args:
            df_doc_path (str): _description_
            ocr_engine (OcrEngineForYoloX, optional): _description_. Defaults to None.
            accepted_ext (list[str], optional): _description_. Defaults to ACCEPTED_EXT.
            other_docid (str, optional): the docid of other documents. Defaults to OTHERS_LABEL.
            thresholds (tuple[float,float], optional): respectively coarse and fine threshold of longest common subsequence to match with title. Defaults to THRESH_HOLD.
            max_length (int, optional): number of first words extracted from page to compare with title. Defaults to MAX_LENGTH.
        """
        self.ocr_engine = ocr_engine
        self.df_doc_path = df_doc_path
        self.other_docid = other_docid
        self.accepted_ext = accepted_ext
        self.ddoc_title_to_docid, self.ddoc_title_to_no_pages = self.extract_dict_from_excel(df_doc_path)
        self.max_length = max_length
        self.thresholds = thresholds
        self.offset = offset

    @staticmethod
    def _sort_dict_by_key_length(d: dict, reverse=False) -> dict:
        # https://www.geeksforgeeks.org/python-program-to-sort-dictionary-by-key-lengths/
        l = sorted(list(d.items()), key=lambda key: len(key[0]), reverse=reverse)
        res = {ele[0]: ele[1] for ele in l}
        return res

    def extract_dict_from_excel(self, df_path: str) -> tuple[dict[str, str], dict[str, str]]:
        df = pd.read_excel(df_path, index_col=0).dropna(how="all")
        df = df[df["Do_classify(Y/N)"] == 1]
        # prioritize the form with longer title length
        # df.sort_values(by='Title', key=lambda x: len(x), inplace=True, ascending=False)
        ddoc_title_to_docid = dict(zip(df['Title'], df["DocID"]))
        ddoc_title_to_docid = self._sort_dict_by_key_length(ddoc_title_to_docid, reverse=True)

        ddoc_title_to_no_pages = dict(zip(df['Title'], df["No. pages"])) | {self.other_docid: 0}
        return ddoc_title_to_docid, ddoc_title_to_no_pages

    def read_from_dir(self, dir_path: str, include_txt: bool = True) -> Dict[str,
                                                                             Union[List[np.ndarray],
                                                                                   np.ndarray, tuple[list[tuple],
                                                                                                     list[str]]]]:
        dir_ = Path(dir_path)
        assert dir_.is_dir(), "{} is not a directory".format(dir_)
        res = dict()
        for f in dir_.glob("*"):
            if not include_txt and f.suffix == ".txt":
                continue
            if f.suffix in self.accepted_ext:
                res[f.name] = RuleBaseDocClassifier.read_from_file(f)
        return res

    def read_from_file(self, file_path: str) -> Union[List[np.ndarray], np.ndarray, tuple[list[tuple], list[str]]]:
        """read from files in the supported extensions
            if file_path is txt -> return ocr results (bboxes and texts)
            if file_path is pdf -> return a list of np.ndarray
            if file_path is image -> return a np.ndarray
        Args:
            file_path (str): path to file to read

        Raises:
            NotImplementedError:

        Returns:
            Union[List[np.ndarray], np.ndarray, tuple[list[tuple], list[str]]]: bboxes and texts
        """
        file_path = Path(file_path)
        if file_path.suffix == ".txt":
            return read_ocr_result_from_txt(str(file_path))
        elif file_path.suffix == ".pdf":
            return self.read_from_pdf(str(file_path))
        elif file_path.suffix in self.accepted_ext:  # the rest should be image files
            return read_image_file(str(file_path))
        else:
            raise NotImplementedError("{} is not supported".format(file_path.suffix))

    @staticmethod
    def read_from_pdf(pdf_path: str) -> List[np.ndarray]:
        return [np.array(img) for img in convert_from_path(pdf_path)]

    def run_ocr(self, img: Union[np.ndarray, list[np.ndarray]],
                batch_mode=False, batch_size=16) -> Union[tuple[list, list],
                                                          list[tuple[list, list]]]:
        '''
        return list of (bbox, text) or list of list of (bbox, text)
        '''
        lbboxes, lwords = self.ocr_engine.inference(img, batch_mode=batch_mode, batch_size=batch_size)
        if not isinstance(img, list):
            lbboxes, lwords = sort_bboxes_and_words(lbboxes, lwords)
            return lbboxes, lwords
        else:
            llbboxes, llwords = [], []
            for i in range(len(lwords)):
                lb, lw = sort_bboxes_and_words(lbboxes[i], lwords[i])
                llbboxes.append(lb)
                llwords.append(lw)
            return llbboxes, llwords

    def classify_by_template_number(self, lwords: List[str], max_length: int) -> str:
        # # TODO: valid assumption that there is only 1 template number in a page, currently return first occurence
        # ocr_str = "".join(lwords[:max_length])
        # match = re.search(r"POS0(?P<cls>\d{1})_20", ocr_str)
        # return "POS-0{}".format(match["cls"]) if match else -1
        # return -1  # to test classify_by_title
        # for word in lwords: #maybe match each word will be more efficient?
        #     match = re.search(r"POS0(?P<cls>\d{1})_20", word)
        # return "POS0{}".format(match["cls"]) if match else -1
        # return -1
        return ""

    def classify_by_title(
            self, lwords: List[str],
            thresholds: tuple[float, float],
            max_length: int, offset: float) -> str:
        ocr_str = "".join(lwords[:max_length])
        for title, docid in self.ddoc_title_to_docid.items():
            title = title.replace(" ", "")
            lcs_len, start_idx_lcs, end_idx_lcs = longest_common_subsequence_with_idx(ocr_str, title)
            if lcs_len / len(title) > thresholds[0]:
                shorten_ocr_str = ocr_str[int(start_idx_lcs * (1 - offset)):int((end_idx_lcs + 1) * (1 + offset))]
                lcs_len = longestCommonSubsequence(shorten_ocr_str, title)
                if lcs_len / len(title) > thresholds[1]:
                    return docid
        return ""

    def preprocess(self, input_path: str) -> list[str]:
        input_ = self.read_from_file(input_path)
        if isinstance(input_, tuple):
            return input_  # read from txt
        if not self.ocr_engine:
            raise ValueError("Please provide an OCR engine to read from {}".format(input_path))
        input_ = input_[0] if isinstance(input_, list) else input_
        return self.run_ocr(input_)

    def classify(self, input_path: str) -> str:
        # TODO: implement ensemble classifier
        _lbboxes, lwords = self.preprocess(input_path)
        # cls_ = RuleBaseDocClassifier.classify_by_template_number(lwords, max_length)
        # if cls_ == -1:
        cls_ = self.classify_by_title(lwords, max_length=self.max_length,
                                      thresholds=self.thresholds, offset=self.offset)
        return self.other_docid if not cls_ else cls_

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
                cls_ = self.classify(img)
                dout[file_name][cls_].append(i)
                curr_first_page_of_doc_idx = i
                curr_cls_of_doc = cls_
                break
        return dout

    def eval(self, df_val_path: str):
        df = pd.read_csv(df_val_path)
        # y_true = [DDOC_LABELS_TO_IDX[label] for label in df["label"]]
        y_true = []
        y_pred = []
        diff = []
        for i, ocr_path in tqdm(enumerate(df["ocr_path"])):
            if ocr_path == "/mnt/ssd500/hungbnt/DocumentClassification/results/ocr/FWD/33forms/Phiếu Yêu Cầu Điều Chỉnh HĐBHNT và Tờ Khai Sức Khỏe (cập nhật 31052021)/Phiếu Yêu Cầu Điều Chỉnh HĐBHNT và Tờ Khai Sức Khỏe (cập nhật 31052021)_0.txt":
                print("DEBUGGING")

            pred = self.classify(ocr_path)
            gt = df["label"].iloc[i]
            y_pred.append(pred)
            y_true.append(gt)
            diff.append(pred == gt)
            if pred != gt:
                print("*" * 100)
                print(df["img_path"].iloc[i])
                print(ocr_path)
                print(gt, pred)
        df["pred"] = y_pred
        df["diff"] = diff
        df.to_csv(f"{df_path}_pred.csv")
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        return y_true, y_pred


# %%
if __name__ == "__main__":
    # %%
    # engine = OcrEngineForYoloX(DET_CFG, DET_CKPT, CLS_CFG, CLS_CKPT)
    # cls_model = RuleBaseDocClassifier(engine)
    # print("Done init")
    # # cls_model.infer("data/Sample_input/Case_2_ghep_toan_bo/")  # OK
    # cls_model.infer("/mnt/ssd500/hungbnt/DocumentClassification/data")  # OK

    # %%
    # df_path = "/mnt/ssd500/hungbnt/DocumentClassification/data/33forms.csv"
    df_path = "/mnt/ssd500/hungbnt/DocumentClassification/data/FWD_and_Samsung.csv"
    # df_path = "/mnt/ssd500/hungbnt/DocumentClassification/data/FWD_val.csv`"
    y_true, y_pred = RuleBaseDocClassifier(df_doc_path=DF_DOC_PATH).eval(df_path)
    # # TODO:
    # # Tờ khai sức khỏe bị nhầm với Phiếu Yêu Cầu Điều Chỉnh Hợp Đồng Bảo Hiểm Nhân Thọ và Tờ Khai Sức Khỏe
    # # Hồ Sơ Yêu Cầu Bảo Hiểm bị nhầm với Xác nhận đồng ý

    # # %%
    # # %%

    # # %%
    # # ocr_path = "/mnt/ssd500/hungbnt/DocumentClassification/results/ocr/POS08/50.pdf.txt"
    # # ocr_path = "/mnt/ssd500/hungbnt/DocumentClassification/results/ocr/POS01/SKM_458e\ Ag22101217531.pdf.txt"
    # # _lbboxes, lwords = read_ocr_result_from_txt(ocr_path)
    # # RuleBaseDocClassifier.classify(lwords)
    # # RuleBaseDocClassifier.classify_by_template_number(lwords)
    # # RuleBaseDocClassifier.classify_by_title(lwords, 0.85, 50)
    # # %%
    # # cls_model.infer("data/Sample_input/Case_1_th_roi-toan-bo/")  # OK
    # # cls_model.infer("data/Sample_input/Case_2_ghep_mot_phan/")  # OK
    # # cls_model.infer("data/Sample_input/Case_2_ghep_toan_bo/")  # OK

    # # cls_model.infer("data/Sample_input/Case_1_tach_roi-toan-bo/CMND.pdf")
    # # cls_model.infer("data/Sample_input/Case_1_tach_roi-toan-bo/Giay_khai_sinh.pdf")
    # # cls_model.infer("data/Sample_input/Case_1_tach_roi-toan-bo/Giay_xac_nhan.pdf")
    # # cls_model.infer("data/Sample_input/Case_1_tach_roi-toan-bo/Hoa_don.pdf")
    # # cls_model.infer("/mnt/hdd2T/AICR/Projects/FWD/Phase1/Completed_forms_SAMSUNG/POS01/1_PDFsam_Scan.pdf")
    # # cls_model.infer("/mnt/hdd2T/AICR/Projects/FWD/Phase1/Completed_forms_SAMSUNG/POS02/1.pdf")
    # # cls_model.infer("/mnt/hdd2T/AICR/Projects/FWD/Phase1/Completed_forms_SAMSUNG/POS03/1.pdf")
    # # cls_model.infer("/mnt/hdd2T/AICR/Projects/FWD/Phase1/Completed_forms_SAMSUNG/POS03/1.pdf")
    # # cls_model.infer("/mnt/hdd2T/AICR/Projects/FWD/Phase1/Completed_forms_SAMSUNG/POS04/1.pdf")
    # # cls_model.infer("/mnt/hdd2T/AICR/Projects/FWD/Phase1/Completed_forms_SAMSUNG/POS05/1.pdf")
    # # cls_model.infer("/mnt/hdd2T/AICR/Projects/FWD/Phase1/Completed_forms_SAMSUNG/POS06/1.pdf")
    # # cls_model.infer("/mnt/hdd2T/AICR/Projects/FWD/Phase1/Completed_forms_SAMSUNG/POS08/1.pdf")

    # # %%
    # import pandas as pd
    # df_path = "/mnt/ssd500/hungbnt/DocumentClassification/data/Static/FWD_documents_list.xlsx"
    # df = pd.read_excel(df_path, index_col=0).dropna(how="all")
    # df = df[df["Do_classify(Y/N)"] == 1]
    # ddoc_title_to_docid = dict(zip(df['Title'], df["DocID"]))
    # ddoc_title_to_no_pages = dict(zip(df['Title'], df["No. pages"]))

    # # %%
    # len(ddoc_title_to_docid)

    # # %%
    # len(df['Title'])

    # # %%
    # df.T
    # # %%

    # # %%
    # df.dropna(subset=["Title"], how="all", inplace=True)
    # len(df)
    # # %%
    # df["Title"].tolist()

    # %%
    # {k:v for k in [1,1,2,2] for v in [3,4,5,6]}
    # df.sort_values(by='Title', key=lambda x: len(str(x)), inplace=True, ascending=False)
    # df.loc["Ti"]

    # %%
