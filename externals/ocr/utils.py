import cv2


def read_image_file(img_path):
    image = cv2.imread(img_path)
    return image


def read_ocr_result_from_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    boxes, words = [], []
    for line in lines:
        if line == "":
            continue
        x1, y1, x2, y2, text = line.split("\t")
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if text != " ":
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
