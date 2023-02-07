from PIL import ImageFont, ImageDraw, Image
# import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


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


def read_image_file(img_path):
    image = cv2.imread(img_path)
    return image


def read_ocr_result_from_txt(file_path: str) -> tuple[list, list]:
    '''
    return list of bounding boxes, list of words
    '''
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    boxes, words = [], []
    for line in lines:
        if line == "":
            continue
        x1, y1, x2, y2, text = line.split("\t")
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if text and text != " ":
            words.append(text)
            boxes.append((x1, y1, x2, y2))
    return boxes, words


def normalize_bbox(x1, y1, x2, y2, w, h):
    x1 = int(float(min(max(0, x1), w)))
    x2 = int(float(min(max(0, x2), w)))
    y1 = int(float(min(max(0, y1), h)))
    y2 = int(float(min(max(0, y2), h)))
    return (x1, y1, x2, y2)


def extend_crop_img(left, top, right, bottom, margin_l=0, margin_t=0.03, margin_r=0.02, margin_b=0.05):
    top = top - (bottom - top) * margin_t
    bottom = bottom + (bottom - top) * margin_b
    left = left - (right - left) * margin_l
    right = right + (right - left) * margin_r
    return left, top, right, bottom


def get_crop_img_and_bbox(img, bbox, extend: bool):
    """
    img : numpy array img
    bbox : should be xyxy format
    """
    if len(bbox) == 5:
        left, top, right, bottom, _conf = bbox
    elif len(bbox) == 4:
        left, top, right, bottom = bbox
    # left_, top_, right_, bottom_ = self.extend_crop_img(left, top, right, bottom)
    if extend:
        left, top, right, bottom = extend_crop_img(left, top, right, bottom)
    left, top, right, bottom = normalize_bbox(left, top, right, bottom, img.shape[1], img.shape[0])
    # left_, top_, right_, bottom_ = self._normalize_bbox(left_, top_, right_, bottom_, img.shape[1], img.shape[0])
    assert (bottom - top) * (right - left) > 0, 'bbox is invalid'
    # assert (bottom_ - top_) * (right_ - left_) > 0, 'bbox is invalid'
    # crop_img = img[top_:bottom_, left_:right_]
    crop_img = img[top:bottom, left:right]
    return crop_img, (left, top, right, bottom)


def get_xyxywh_base_on_format(bbox, format):
    if format == "xywh":
        x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        x2, y2 = x1 + w, y1 + h
    elif format == "xyxy":
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
    else:
        raise NotImplementedError("Invalid format {}".format(format))
    return (x1, y1, x2, y2, w, h)


def get_dynamic_params_for_bbox_of_label(text, x1, y1, w, h, img_h, img_w, font):
    font_scale_factor = img_h / (img_w + img_h)
    font_scale = w / (w + h) * font_scale_factor  # adjust font scale by width height
    thickness = int(font_scale_factor) + 1
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
    text_offset_x = x1
    text_offset_y = y1 - thickness
    box_coords = ((text_offset_x, text_offset_y + 1), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))
    return (font_scale, thickness, text_height, box_coords)


def visualize_bbox_and_label(
        img, bboxes, texts, bbox_color=(60, 180, 200),
        text_color=(0, 0, 0),
        format="xyxy", is_vnese=False):
    ori_img_type = type(img)
    if is_vnese:
        img = Image.fromarray(img) if ori_img_type is np.ndarray else img
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size
        font_pil_str = "fonts/arial.ttf"
        font_cv2 = cv2.FONT_HERSHEY_SIMPLEX
    else:
        img_h, img_w = img.shape[0], img.shape[1]
        font_cv2 = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(bboxes)):
        text = texts[i]  # text = "{}: {:.0f}%".format(LABELS[classIDs[i]], confidences[i]*100)
        x1, y1, x2, y2, w, h = get_xyxywh_base_on_format(bboxes[i], format)
        font_scale, thickness, text_height, box_coords = get_dynamic_params_for_bbox_of_label(
            text, x1, y1, w, h, img_h, img_w, font=font_cv2)
        if is_vnese:
            font_pil = ImageFont.truetype(font_pil_str, size=text_height)
            fdraw_text = draw.text
            fdraw_bbox = draw.rectangle
            # Pil use different coordinate => y = y+thickness = y-thickness + 2*thickness
            arg_text = ((box_coords[0][0], box_coords[1][1]), text)
            kwarg_text = {"font": font_pil, "fill": text_color, "width": thickness}
            arg_rec = ((x1, y1, x2, y2),)
            kwarg_rec = {"outline": bbox_color, "width": thickness}
            arg_rec_text = ((box_coords[0], box_coords[1]),)
            kwarg_rec_text = {"fill": bbox_color, "width": thickness}
        else:
            # cv2.rectangle(img, box_coords[0], box_coords[1], color, cv2.FILLED)
            # cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(50, 0,0), thickness=thickness)
            # cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            fdraw_text = cv2.putText
            fdraw_bbox = cv2.rectangle
            arg_text = (img, text, box_coords[0])
            kwarg_text = {"fontFace": font_cv2, "fontScale": font_scale, "color": text_color, "thickness": thickness}
            arg_rec = (img, (x1, y1), (x2, y2))
            kwarg_rec = {"color": bbox_color, "thickness": thickness}
            arg_rec_text = (img, box_coords[0], box_coords[1])
            kwarg_rec_text = {"color": bbox_color, "thickness": cv2.FILLED}
        # draw a bounding box rectangle and label on the img
        fdraw_bbox(*arg_rec, **kwarg_rec)
        fdraw_bbox(*arg_rec_text, **kwarg_rec_text)
        fdraw_text(*arg_text, **kwarg_text)  # text have to put in front of rec_text
    return np.array(img) if ori_img_type is np.ndarray and is_vnese else img
