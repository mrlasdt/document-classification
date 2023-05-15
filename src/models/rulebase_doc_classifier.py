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
from unidecode import unidecode
from collections import defaultdict
from src.models.base_doc_classifier import BaseDocClasifier
# INPUT_DIR = Path("data/Sample_input/Case_1_tach_roi-toan-bo")
# SAVE_DIR = Path("results").joinpath(INPUT_DIR.name)

DET_CFG = "/home/sds/datnt/mmdetection/logs/textdet-fwd/yolox_s_8x8_300e_cocotext_1280.py"
DET_CKPT = "/home/sds/datnt/mmdetection/logs/textdet-fwd/best_bbox_mAP_epoch_100.pth"
CLS_CFG = "/home/sds/datnt/mmocr/logs/satrn_big_2022-10-31/satrn_big.py"
CLS_CKPT = "/home/sds/datnt/mmocr/logs/satrn_big_2022-10-31/best.pth"
# DF_DOC_PATH = "data/Static/FWD_documents_list.csv"
# DF_DOC_PATH = "data/Static/FWD_documents_list_custom_for_16forms.csv"
DF_DOC_PATH = "data/Static/230306_forms.xlsx"
DF_VAL_PATH = "data/230306_forms.csv"
ACCEPTED_EXT = [".pdf", ".png", ".jpg", ".jpeg"]
OTHERS_LABEL = "OTHERS"
# fine_corrected_coef seems to decrease the performance? (see the case of results/ocr/Samsung/TDDG/e5e48b5e1449cc1795584.txt)
THRESHHOLDS = {"coarse": 0.7, "fine": 0.9,
               "fine_corrected": 0.45, "fine_corrected_skip": 1.0}
MAX_LENGTH = 60
OFFSET_LCS = 0.0  # use for the lcs with index function to offset the return len of string
# a title is considered prior to another title if lcs(t1,t2) > THRESHOLDs['fine'] + OFFSET_PRIOR and len(t1) > len(t2)
OFFSET_PRIOR = 0.03


def longestCommonSubsequence(text1: str, text2: str) -> int:
    # https://leetcode.com/problems/longest-common-subsequence/discuss/351689/JavaPython-3-Two-DP-codes-of-O(mn)-and-O(min(m-n))-spaces-w-picture-and-analysis
    dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]
    for i, c in enumerate(text1):
        for j, d in enumerate(text2):
            dp[i + 1][j + 1] = 1 + \
                dp[i][j] if c == d else max(dp[i][j + 1], dp[i + 1][j])
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
    right_idx = 0
    max_lcs = 0
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
                if L[i][j] > max_lcs:
                    max_lcs = L[i][j]
                    right_idx = i
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

        # Create a string variable to store the lcs string
    lcs = L[i][j]
    # Start from the right-most-bottom-most corner and
    # one by one store characters in lcs[]
    i = m
    j = n
    # right_idx = 0
    while i > 0 and j > 0:
        # If current character in X[] and Y are same, then
        # current character is part of LCS
        if X[i - 1] == Y[j - 1]:

            i -= 1
            j -= 1
        # If not same, then find the larger of two and
        # go in the direction of larger value
        elif L[i - 1][j] > L[i][j - 1]:
            # right_idx = i if not right_idx else right_idx #the first change in L should be the right index of the lcs
            i -= 1
        else:
            j -= 1
    return lcs, i, max(i + lcs, right_idx)


class RuleBaseDocClassifier(BaseDocClasifier):
    def __init__(
            self, df_doc_path: str, ocr_engine: OcrEngineForYoloX = None, accepted_ext: List[str] = ACCEPTED_EXT,
            other_docid: str = OTHERS_LABEL, thresholds: dict = THRESHHOLDS, max_length: int = MAX_LENGTH,
            offset_lcs: float = OFFSET_LCS, offset_prior: float = OFFSET_PRIOR):
        """ Classify document base on defined rule
        Args:
            df_doc_path (str): _description_
            ocr_engine (OcrEngineForYoloX, optional): _description_. Defaults to None.
            accepted_ext (List[str], optional): _description_. Defaults to ACCEPTED_EXT.
            other_docid (str, optional): the docid of other documents. Defaults to OTHERS_LABEL.
            thresholds (Tuple[float,float], optional): respectively coarse and fine threshold of longest common subsequence to match with title. Defaults to THRESH_HOLD.
            max_length (int, optional): number of first words extracted from page to compare with title. Defaults to MAX_LENGTH.
        """
        self.ocr_engine = ocr_engine
        self.df_doc_path = df_doc_path
        self.other_docid = other_docid
        self.accepted_ext = accepted_ext
        self.ddoc_title_to_docid, self.ddoc_title_to_no_pages = self.extract_dict_from_excel(
            df_doc_path)
        self.max_length = max_length
        self.thresholds = thresholds
        self.offset_lcs = offset_lcs
        self.offset_prior = offset_prior
        self.dpriority_docid = self.generate_priority_dict()

    def generate_priority_dict(self):
        """
        generate a priority dictionary of documents to solve the following case:
        doci = "Xác nhận động ý và tờ khai sức khỏe"
        docj = "Tờ khai sức khỏe"
        Since doci contains docj and longer than docj, it has a high chance to be missclassied as document j
        => We should prioritize doci over docj
        dprior = {docj : [docj1 docj2]}...
        """
        dprior = defaultdict(list)
        for ititle, idocid in self.ddoc_title_to_docid.items():
            for jtitle, jdocid in self.ddoc_title_to_docid.items():
                match_score = longestCommonSubsequence(
                    jtitle, ititle) / len(jtitle)
                if match_score > self.thresholds["fine"] + self.offset_prior and len(jtitle) < len(ititle):
                    dprior[jdocid].append(idocid)
        return dprior

    @staticmethod
    def _sort_dict_by_key_length(d: dict, reverse=False) -> dict:
        """
        sort the self.ddoc_title_to_docid list so that the longer title can be process first
        this is align with the self.classify_title algorithm and self.dpriority_docid dict
        """
        # https://www.geeksforgeeks.org/python-program-to-sort-dictionary-by-key-lengths/
        l = sorted(list(d.items()), key=lambda key: len(
            key[0]), reverse=reverse)
        res = {ele[0]: ele[1] for ele in l}
        return res

    def extract_dict_from_excel(self, df_path: str) -> Tuple[dict[str, str], dict[str, int]]:
        # df = pd.read_excel(df_path, index_col=0) if df_path.endswith(".xlsx") else pd.read_csv(df_path, index_col=0)
        # df.dropna(how="all", inplace=True)
        # df = df[df["Do_classify(Y/N)"] == 1]
        # prioritize the form with longer title length
        # df.sort_values(by='Title', key=lambda x: len(x), inplace=True, ascending=False)
        df = pd.read_excel(df_path) if df_path.endswith(
            ".xlsx") else pd.read_csv(df_path, index_col=0)
        ddoc_title_to_docid = {title: docid for title, docid in zip(df['Title'], df["DocID"]) if isinstance(
            title, str) and isinstance(docid, str)}  # eliminate all nan columns (float value)
        ddoc_title_to_docid = self._sort_dict_by_key_length(
            ddoc_title_to_docid, reverse=True)
        df_no_pages = [1] * len(df["Title"])  # TODO: just for testing
        # ddoc_title_to_no_pages = dict(zip(df['Title'], df["No. pages"])) | {self.other_docid: 0}
        ddoc_title_to_no_pages = dict(zip(df['Title'], df_no_pages)) | {
            self.other_docid: 1}
        return ddoc_title_to_docid, ddoc_title_to_no_pages

    def read_from_dir(self, dir_path: str, include_txt: bool = True) -> Dict[str,
                                                                             Union[List[np.ndarray],
                                                                                   np.ndarray, Tuple[List[tuple],
                                                                                                     List[str]]]]:
        dir_ = Path(dir_path)
        assert dir_.is_dir(), "{} is not a directory".format(dir_)
        res = dict()
        for f in dir_.glob("*"):
            if not include_txt and f.suffix == ".txt":
                continue
            if f.suffix in self.accepted_ext:
                res[f.name] = RuleBaseDocClassifier.read_from_file(f)
        return res

    def read_from_file(self, file_path: str) -> Union[List[np.ndarray], np.ndarray, Tuple[List[tuple], List[str]]]:
        """read from files in the supported extensions
            if file_path is txt -> return ocr results (bboxes and texts)
            if file_path is pdf -> return a list of np.ndarray
            if file_path is image -> return a np.ndarray
        Args:
            file_path (str): path to file to read

        Raises:
            NotImplementedError:

        Returns:
            Union[List[np.ndarray], np.ndarray, Tuple[List[tuple], List[str]]]: bboxes and texts
        """
        file_path = Path(file_path)
        if file_path.suffix == ".txt":
            return read_ocr_result_from_txt(str(file_path))
        elif file_path.suffix == ".pdf":
            return self.read_from_pdf(str(file_path))
        elif file_path.suffix in self.accepted_ext:  # the rest should be image files
            return read_image_file(str(file_path))
        else:
            raise NotImplementedError(
                "{} is not supported".format(file_path.suffix))

    @staticmethod
    def read_from_pdf(pdf_path: str) -> List[np.ndarray]:
        return [np.array(img) for img in convert_from_path(pdf_path)]

    def run_ocr(self, img: Union[np.ndarray, List[np.ndarray]],
                batch_mode=False, batch_size=16) -> Union[Tuple[list, list],
                                                          List[Tuple[list, list]]]:
        '''
        return list of (bbox, text) or list of list of (bbox, text)
        '''
        lbboxes, lwords = self.ocr_engine.inference(
            img, batch_mode=batch_mode, batch_size=batch_size)
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
        # ocr_str = "".join(lwords[:max_length])
        # match = re.search(r"POS0(?P<cls>\d{1})_20", ocr_str)
        # return "POS-0{}".format(match["cls"]) if match else -1
        return ""

    def customize_preprocess_for_specific_docid(self, doc_id: str, s: str) -> str:
        if doc_id == "GUQ1":
            s = re.sub(r"-+0+", "", s)
        return s

    def lcs_matching(self, ocr_str: str, title: str, mode: str) -> Tuple[float, int, int]:
        if mode == "coarse":
            lcs_len, start_idx_lcs, end_idx_lcs = longest_common_subsequence_with_idx(
                ocr_str, title)
            coarse_score = lcs_len / len(title)
            return coarse_score, start_idx_lcs, end_idx_lcs
        elif mode == "fine":
            # remove accent and lowercase may improve performance since the ocr text may have minor error
            lcs_len = longestCommonSubsequence(
                unidecode(ocr_str).lower(), unidecode(title).lower())
            # there is a case like POS03 shorten_ocr_str contains POS01 inside, so POS01 was misclassied as POS03 since the lcs/len(title) score was 1.0 => corrected by a term len(title)/len(shorten_ocr_str)
            fine_score = lcs_len / len(title)
            return fine_score, 0, -1
        else:
            raise ValueError("Invalid mode: ", mode)

    def compute_corrected_score(self, fine_score_: float, title_: str, shorten_ocr_str_: str) -> float:
        corrected_coef = len(title_) / len(shorten_ocr_str_)
        left_score, _, _ = self.lcs_matching(
            shorten_ocr_str_[:len(title_)], title_, "fine")
        right_score, _, _ = self.lcs_matching(
            shorten_ocr_str_[-len(title_):], title_, "fine")
        corrected_score_ = (fine_score_ * corrected_coef +
                            right_score + left_score) / 3
        return corrected_score_

    def classify_by_title(
            self, lwords: List[str],
            thresholds: dict[str, float],
            max_length: int, offset: float) -> str:
        ocr_str = "".join(lwords[:max_length])
        best_docid = ""
        best_score = 0.0
        for title, docid in self.ddoc_title_to_docid.items():
            title = title.replace(" ", "")
            coarse_score, start_idx_lcs, end_idx_lcs = self.lcs_matching(
                ocr_str, title, "coarse")
            if coarse_score < thresholds["coarse"]:
                continue
            shorten_ocr_str = ocr_str[int(
                start_idx_lcs * (1 - offset)):int((end_idx_lcs) * (1 + offset))]
            shorten_ocr_str = self.customize_preprocess_for_specific_docid(
                docid, shorten_ocr_str)
            fine_score, _, _ = self.lcs_matching(
                shorten_ocr_str, title, "fine")
            if fine_score < thresholds["fine"]:
                continue
            corrected_score = self.compute_corrected_score(
                fine_score, title, shorten_ocr_str)
            if corrected_score > max(best_score, thresholds["fine_corrected"]) and best_docid not in self.dpriority_docid[docid]:
                best_docid = docid
                best_score = corrected_score

            if best_score >= thresholds["fine_corrected_skip"]:
                return best_docid  # improve efficiency
        return best_docid

    def preprocess(self, input_path: str) -> List[str]:
        input_ = self.read_from_file(input_path)
        if isinstance(input_, tuple):
            return input_  # read from txt
        if not self.ocr_engine:
            raise ValueError(
                "Please provide an OCR engine to read from {}".format(input_path))
        input_ = input_[0] if isinstance(input_, list) else input_
        return self.run_ocr(input_)

    def classify(self, input_path: str) -> str:
        # TODO: implement ensemble classifier
        _lbboxes, lwords = self.preprocess(input_path)
        # cls_ = RuleBaseDocClassifier.classify_by_template_number(lwords, max_length)
        # if cls_ == -1:
        cls_ = self.classify_by_title(lwords, max_length=self.max_length,
                                      thresholds=self.thresholds, offset=self.offset_lcs)
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

    def print_config(self):
        print("[INFO]: Configuration")
        print("thresholds: ", self.thresholds)

    def eval(self, df_val_path: str, save_pred: bool = True):
        self.print_config()
        df = pd.read_csv(df_val_path)
        # y_true = [DDOC_LABELS_TO_IDX[label] for label in df["label"]]
        y_true = []
        y_pred = []
        diff = []
        for i, ocr_path in tqdm(enumerate(df["ocr_path"])):

            gt = df["label_char"].iloc[i]
            # if ocr_path in [
            # "results/ocr/FWD/7forms_IMG/POS01/1.pdf.txt",  # lost title
            # "results/ocr/FWD/7forms_IMG/POS04/27.pdf.txt",  # lost title
            # "results/ocr/FWD/7forms_IMG/POS04/32.pdf.txt",  # lost title
            # "results/ocr/Samsung/TDDG/4e603ec0a1d7798920c6.txt", #threshold was a bit high (0.86 vs 0.91)
            #     "results/ocr/Samsung/DCYCBH/5a77fc5784405c1e05512.txt", #threshold was a bit high
            #     "results/ocr/Samsung/GUQ2/9f42f901a71c7f42260d1.txt", #shorten ocr str was too long => should have a threshold for corrected score #TODO
            # ]:
            #     print("DEBUGGING ", ocr_path)
            # ocr_path ="results/ocr/FWD/7forms_IMG/POS04/32.pdf.txt"
            if ocr_path in [
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/Bang cau hoi so thich dua xe/00206BF7E1DF230302130859.txt",  # blank
                    # no title (should be other)
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/Thong tin to chuc doanh nghiep/TT TO CHUC DN-2.txt",
                    # no title (should be other)
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/Thong tin to chuc doanh nghiep/TT TO CHUC DN-1.txt",
                    # no title (should be other)
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/Thong tin to chuc doanh nghiep/00206BF7E1DF230302130859.txt",  # blank
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/BCH tieu duong/00206BF7E1DF230302130859.txt",  # blank
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/BCH dong kinh/00206BF7E1DF230302130859.txt",  # blank
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/BCH su dung ruou bia/00206BF7E1DF230302130859.txt",  # blank
                    # too much noise due to screen captured
                    # "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/BCH su dung ruou bia/20230305_214607.txt",
                    # too much noise due to screen captured
                    # "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/BCH su dung ruou bia/20230305_214613.txt",
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/BCH benh tim mach/00206BF7E1DF230302130859.txt",  # blank
                    # failed to rotate due to 2 documents presented
                    # "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/DS NV tham gia BH/DS NV THAM GIA BH-3.txt",
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/DS NV tham gia BH/00206BF7E1DF230302130859.txt",  # blank
                    # failed to rotate due to 2 documents presented (no bbox found)
                    # "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/bang cau hoi ve noi cu ngu danh cho ng nuoc ngoai/BCH NGUOI NUOC NGOAI-2.txt",
                    # failed to rotate due to 2 documents presented
                    # "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/bang cau hoi ve noi cu ngu danh cho ng nuoc ngoai/BCH NGUOI NUOC NGOAI-3.txt",
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/bang cau hoi ve noi cu ngu danh cho ng nuoc ngoai/00206BF7E1DF230302130859.txt",  # blank
                    # # too much noise due to screen captured
                    # "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/BCH benh hen suyen/20230305_214743.txt",
                    # "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/BCH benh hen suyen/BCH SUYEN-5.txt" #words to lines error
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/BCH benh hen suyen/00206BF7E1DF230302130859.txt",  # blank
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/BCH chi tiet ve suc khoe_da day/00206BF7E1DF230302130859.txt",  # blank
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/cau hoi tai chinh/00206BF7E1DF230302130859.txt",  # blank
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/BCH chi tiet SK/00206BF7E1DF230302130859.txt",  # blank
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/Bang cau hoi ve nganh hang khong/BCH NGANH HANG KHONG-4.txt",  # wrong label
                    # failed to rotate due to 2 documents presented
                    # "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/Bang cau hoi ve nganh hang khong/BCH NGANH HANG KHONG-6.txt",
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/Bang cau hoi ve nganh hang khong/BCH NGANH HANG KHONG-1.txt",  # wrong label
                    # failed to rotate due to 2 documents presented
                    # "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/Bang cau hoi ve nganh hang khong/BCH NGANH HANG KHONG-7.txt",
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/Bang cau hoi ve nganh hang khong/00206BF7E1DF230302130859.txt",  # blank
                    # "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/cau hoi ve chan thuong dau/BCH CHAN THUONG DAU-3.txt", #failed to rotate
                    # "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/cau hoi ve chan thuong dau/BCH CHAN THUONG DAU-2.txt", #ocr failed
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/cau hoi ve chan thuong dau/00206BF7E1DF230302130859.txt",  # blank
                    # "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/BCH tai chinh PO LA/BCH TAI CHINH PO-LA-3.txt", #failed to rotate
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/BCH tai chinh PO LA/BCH TAI CHINH PO-LA-5.txt",  # wrong label
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/BCH tai chinh PO LA/00206BF7E1DF230302130859.txt",  # blank
                    # "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/BCH dau that nguc/BCH DAU THAT NGUC-1.txt", #ocr failed
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/BCH dau that nguc/00206BF7E1DF230302130859.txt",  # blank
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/mau kk thong tin hoan tien/00206BF7E1DF230302130859.txt",  # blank
                    # "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/mau ke khai truong hop nguoi dong phi la nguoi than cua BMBH/DONG PHI LA NGUOI THAN-3.txt", #failed to rotate
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/mau ke khai truong hop nguoi dong phi la nguoi than cua BMBH/00206BF7E1DF230302130859.txt",  # blank
                    "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/Don chap thuan bao hiem/00206BF7E1DF230302130859.txt",  # blank
            ]:
                continue
            # ocr_path = "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/bang cau hoi ve noi cu ngu danh cho ng nuoc ngoai/BCH NGUOI NUOC NGOAI-3.txt"
            # ocr_path = "/mnt/ssd1T/hungbnt/DocumentClassification/results/ocr/FWD/230306_forms/01_Classified_forms/Bang cau hoi cao huyet ap/20230228_095128.txt"
            # if gt not in ["POS01"]:
            # continue
            pred = self.classify(ocr_path)
            y_pred.append(pred)
            y_true.append(gt)
            diff.append(pred == gt)
            if pred != gt:
                print("*" * 100)
                print(df["img_path"].iloc[i])
                print(ocr_path)
                print(pred, gt)
        if save_pred:
            df["pred"] = y_pred
            df["diff"] = diff
            df.to_csv(f"{df_val_path}_pred.csv")
        print(classification_report(y_true, y_pred))
        # more options can be specified also
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            # https://stackoverflow.com/questions/19124601/pretty-print-an-entire-pandas-series-dataframe
            labels = list(set(list(df["label_char"].values) + ["OTHERS"]))
            # print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels))
        return y_true, y_pred


# %%
if __name__ == "__main__":
    # %%
    # engine = OcrEngineForYoloX(DET_CFG, DET_CKPT, CLS_CFG, CLS_CKPT)
    # cls_model = RuleBaseDocClassifier(engine)
    # print("Done init")
    # # cls_model.infer("data/Sample_input/Case_2_ghep_toan_bo/")  # OK
    # cls_model.infer("data")  # OK

    # %%
    # df_path = "data/33forms_pred.csv"
    # df_path = "data/FWD_and_Samsung.csv"
    # df_path = "data/FWD_val.csv`"
    # df_path = "data/202302_3forms.csv"
    y_true, y_pred = RuleBaseDocClassifier(
        df_doc_path=DF_DOC_PATH).eval(DF_VAL_PATH, save_pred=False)
    # # %%
    # # %%

    # # %%
    # # ocr_path = "results/ocr/POS08/50.pdf.txt"
    # # ocr_path = "results/ocr/POS01/SKM_458e\ Ag22101217531.pdf.txt"
    # # _lbboxes, lwords = read_ocr_result_from_txt(ocr_path)
    # # RuleBaseDocClassifier.classify(lwords)
    # # RuleBaseDocClassifier.classify_by_template_number(lwords)
    # # RuleBaseDocClassifier.classify_by_title(lwords, 0.85, 50)P
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
    # df_path = "data/Static/FWD_documents_list.xlsx"
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
