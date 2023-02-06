from builtins import dict

MIN_IOU_HEIGHT = 0.7
MIN_WIDTH_LINE_RATIO = 0.05


class Word:
    def __init__(
        self,
        text="",
        image=None,
        conf_detect=0.0,
        conf_cls=0.0,
        bndbox=None,
        kie_label="",
    ):
        self.type = "word"
        self.text = text
        self.image = image
        self.conf_detect = conf_detect
        self.conf_cls = conf_cls
        # [left, top,right,bot] coordinate of top-left and bottom-right point
        self.boundingbox = bndbox if bndbox is not None else [-1, -1, -1, -1]
        self.word_id = 0  # id of word
        self.word_group_id = 0  # id of word_group which instance belongs to
        self.line_id = 0  # id of line which instance belongs to
        self.paragraph_id = 0  # id of line which instance belongs to
        self.kie_label = kie_label

    def invalid_size(self):
        return (self.boundingbox[2] - self.boundingbox[0]) * (
            self.boundingbox[3] - self.boundingbox[1]
        ) > 0

    def is_special_word(self):
        left, top, right, bottom = self.boundingbox
        width, height = right - left, bottom - top
        text = self.text

        if text is None:
            return True

        # if len(text) > 7:
        #     return True
        if len(text) >= 7:
            no_digits = sum(c.isdigit() for c in text)
            return no_digits / len(text) >= 0.3

        return False


class Word_group:
    def __init__(self):
        self.type = "word_group"
        self.list_words = []  # dict of word instances
        self.word_group_id = 0  # word group id
        self.line_id = 0  # id of line which instance belongs to
        self.paragraph_id = 0  # id of paragraph which instance belongs to
        self.text = ""
        self.boundingbox = [-1, -1, -1, -1]
        self.kie_label = ""

    def add_word(self, word: Word):  # add a word instance to the word_group
        if word.text != "✪":
            for w in self.list_words:
                if word.word_id == w.word_id:
                    print("Word id collision")
                    return False
            word.word_group_id = self.word_group_id  #
            word.line_id = self.line_id
            word.paragraph_id = self.paragraph_id
            self.list_words.append(word)
            self.text += " " + word.text
            if self.boundingbox == [-1, -1, -1, -1]:
                self.boundingbox = word.boundingbox
            else:
                self.boundingbox = [
                    min(self.boundingbox[0], word.boundingbox[0]),
                    min(self.boundingbox[1], word.boundingbox[1]),
                    max(self.boundingbox[2], word.boundingbox[2]),
                    max(self.boundingbox[3], word.boundingbox[3]),
                ]
            return True
        else:
            return False

    def update_word_group_id(self, new_word_group_id):
        self.word_group_id = new_word_group_id
        for i in range(len(self.list_words)):
            self.list_words[i].word_group_id = new_word_group_id

    def update_kie_label(self):
        list_kie_label = [word.kie_label for word in self.list_words]
        dict_kie = dict()
        for label in list_kie_label:
            if label not in dict_kie:
                dict_kie[label] = 1
            else:
                dict_kie[label] += 1
        total = len(list(dict_kie.values()))
        max_value = max(list(dict_kie.values()))
        list_keys = list(dict_kie.keys())
        list_values = list(dict_kie.values())
        self.kie_label = list_keys[list_values.index(max_value)]

    def update_text(self):  # update text after changing positions of words in list word
        text = ""
        for word in self.list_words:
            text += " " + word.text
        self.text = text


class Line:
    def __init__(self):
        self.type = "line"
        self.list_word_groups = []  # list of Word_group instances in the line
        self.line_id = 0  # id of line in the paragraph
        self.paragraph_id = 0  # id of paragraph which instance belongs to
        self.text = ""
        self.boundingbox = [-1, -1, -1, -1]

    def add_group(self, word_group: Word_group):  # add a word_group instance
        if word_group.list_words is not None:
            for wg in self.list_word_groups:
                if word_group.word_group_id == wg.word_group_id:
                    print("Word_group id collision")
                    return False

            self.list_word_groups.append(word_group)
            self.text += word_group.text
            word_group.paragraph_id = self.paragraph_id
            word_group.line_id = self.line_id

            for i in range(len(word_group.list_words)):
                word_group.list_words[
                    i
                ].paragraph_id = self.paragraph_id  # set paragraph_id for word
                word_group.list_words[i].line_id = self.line_id  # set line_id for word
            return True
        return False

    def update_line_id(self, new_line_id):
        self.line_id = new_line_id
        for i in range(len(self.list_word_groups)):
            self.list_word_groups[i].line_id = new_line_id
            for j in range(len(self.list_word_groups[i].list_words)):
                self.list_word_groups[i].list_words[j].line_id = new_line_id

    def merge_word(self, word):  # word can be a Word instance or a Word_group instance
        if word.text != "✪":
            if self.boundingbox == [-1, -1, -1, -1]:
                self.boundingbox = word.boundingbox
            else:
                self.boundingbox = [
                    min(self.boundingbox[0], word.boundingbox[0]),
                    min(self.boundingbox[1], word.boundingbox[1]),
                    max(self.boundingbox[2], word.boundingbox[2]),
                    max(self.boundingbox[3], word.boundingbox[3]),
                ]
            self.list_word_groups.append(word)
            self.text += " " + word.text
            return True
        return False

    def __cal_ratio(self, top1, bottom1, top2, bottom2):
        sorted_vals = sorted([top1, bottom1, top2, bottom2])
        intersection = sorted_vals[2] - sorted_vals[1]
        min_height = min(bottom1 - top1, bottom2 - top2)
        if min_height == 0:
            return -1
        ratio = intersection / min_height
        return ratio

    def __cal_ratio_height(self, top1, bottom1, top2, bottom2):

        height1, height2 = top1 - bottom1, top2 - bottom2
        ratio_height = float(max(height1, height2)) / float(min(height1, height2))
        return ratio_height

    def in_same_line(self, input_line, thresh=0.7):
        # calculate iou in vertical direction
        _, top1, _, bottom1 = self.boundingbox
        _, top2, _, bottom2 = input_line.boundingbox

        ratio = self.__cal_ratio(top1, bottom1, top2, bottom2)
        ratio_height = self.__cal_ratio_height(top1, bottom1, top2, bottom2)

        if (
            (top1 in range(top2, bottom2) or top2 in range(top1, bottom1))
            and ratio >= thresh
            and (ratio_height < 2)
        ):
            return True
        return False


class Paragraph:
    def __init__(self, id=0, lines=None):
        self.list_lines = lines if lines is not None else []  # list of all lines in the paragraph
        self.paragraph_id = id  # index of paragraph in the ist of paragraph
        self.text = ""
        self.boundingbox = [-1, -1, -1, -1]

    def add_line(self, line: Line):  # add a line instance
        if line.list_word_groups is not None:
            for l in self.list_lines:
                if line.line_id == l.line_id:
                    print("Line id collision")
                    return False
            for i in range(len(line.list_word_groups)):
                line.list_word_groups[
                    i
                ].paragraph_id = (
                    self.paragraph_id
                )  # set paragraph id for every word group in line
                for j in range(len(line.list_word_groups[i].list_words)):
                    line.list_word_groups[i].list_words[
                        j
                    ].paragraph_id = (
                        self.paragraph_id
                    )  # set paragraph id for every word in word groups
            line.paragraph_id = self.paragraph_id  # set paragraph id for line
            self.list_lines.append(line)  # add line to paragraph
            self.text += " " + line.text
            return True
        else:
            return False

    def update_paragraph_id(
        self, new_paragraph_id
    ):  # update new paragraph_id for all lines, word_groups, words inside paragraph
        self.paragraph_id = new_paragraph_id
        for i in range(len(self.list_lines)):
            self.list_lines[
                i
            ].paragraph_id = new_paragraph_id  # set new paragraph_id for line
            for j in range(len(self.list_lines[i].list_word_groups)):
                self.list_lines[i].list_word_groups[
                    j
                ].paragraph_id = new_paragraph_id  # set new paragraph_id for word_group
                for k in range(len(self.list_lines[i].list_word_groups[j].list_words)):
                    self.list_lines[i].list_word_groups[j].list_words[
                        k
                    ].paragraph_id = new_paragraph_id  # set new paragraph id for word
        return True


def resize_to_original(
    boundingbox, scale
):  # resize coordinates to match size of original image
    left, top, right, bottom = boundingbox
    left *= scale[1]
    right *= scale[1]
    top *= scale[0]
    bottom *= scale[0]
    return [left, top, right, bottom]


def check_iomin(word: Word, word_group: Word_group):
    min_height = min(
        word.boundingbox[3] - word.boundingbox[1],
        word_group.boundingbox[3] - word_group.boundingbox[1],
    )
    intersect = min(word.boundingbox[3], word_group.boundingbox[3]) - max(
        word.boundingbox[1], word_group.boundingbox[1]
    )
    if intersect / min_height > 0.7:
        return True
    return False


def prepare_line(words):
    lines = []
    visited = [False] * len(words)
    for id_word, word in enumerate(words):
        if word.invalid_size() == 0:
            continue
        new_line = True
        for i in range(len(lines)):
            if (
                lines[i].in_same_line(word) and not visited[id_word]
            ):  # check if word is in the same line with lines[i]
                lines[i].merge_word(word)
                new_line = False
                visited[id_word] = True

        if new_line == True:
            new_line = Line()
            new_line.merge_word(word)
            lines.append(new_line)

    # print(len(lines))
    # sort line from top to bottom according top coordinate
    lines.sort(key=lambda x: x.boundingbox[1])
    return lines


def __create_word_group(word, word_group_id):
    new_word_group = Word_group()
    new_word_group.word_group_id = word_group_id
    new_word_group.add_word(word)

    return new_word_group


def __sort_line(line):
    line.list_word_groups.sort(
        key=lambda x: x.boundingbox[0]
    )  # sort word in lines from left to right

    return line


def __merge_text_for_line(line):
    line.text = ""
    for word in line.list_word_groups:
        line.text += " " + word.text

    return line


def __update_list_word_groups(line, word_group_id, word_id, line_width):

    old_list_word_group = line.list_word_groups
    list_word_groups = []

    inital_word_group = __create_word_group(old_list_word_group[0], word_group_id)
    old_list_word_group[0].word_id = word_id
    list_word_groups.append(inital_word_group)
    word_group_id += 1
    word_id += 1

    for word in old_list_word_group[1:]:
        check_word_group = True
        word.word_id = word_id
        word_id += 1

        if (
            (not list_word_groups[-1].text.endswith(":"))
            and (
                (word.boundingbox[0] - list_word_groups[-1].boundingbox[2])
                / line_width
                < MIN_WIDTH_LINE_RATIO
            )
            and check_iomin(word, list_word_groups[-1])
        ):
            list_word_groups[-1].add_word(word)
            check_word_group = False

        if check_word_group:
            new_word_group = __create_word_group(word, word_group_id)
            list_word_groups.append(new_word_group)
            word_group_id += 1
    line.list_word_groups = list_word_groups
    return line, word_group_id, word_id


def construct_word_groups_in_each_line(lines):
    line_id = 0
    word_group_id = 0
    word_id = 0
    for i in range(len(lines)):
        if len(lines[i].list_word_groups) == 0:
            continue

        # left, top ,right, bottom
        line_width = lines[i].boundingbox[2] - lines[i].boundingbox[0]  # right - left

        lines[i] = __sort_line(lines[i])

        # update text for lines after sorting
        lines[i] = __merge_text_for_line(lines[i])

        lines[i], word_group_id, word_id = __update_list_word_groups(
            lines[i],
            word_group_id,
            word_id,
            line_width)
        lines[i].update_line_id(line_id)
        line_id += 1
    return lines


def words_to_lines(words, check_special_lines=True):  # words is list of Word instance
    # sort word by top
    words.sort(key=lambda x: (x.boundingbox[1], x.boundingbox[0]))
    number_of_word = len(words)
    # print(number_of_word)
    # sort list words to list lines, which have not contained word_group yet
    lines = prepare_line(words)

    # construct word_groups in each line
    lines = construct_word_groups_in_each_line(lines)
    return lines, number_of_word


def near(word_group1: Word_group, word_group2: Word_group):
    min_height = min(
        word_group1.boundingbox[3] - word_group1.boundingbox[1],
        word_group2.boundingbox[3] - word_group2.boundingbox[1],
    )
    overlap = min(word_group1.boundingbox[3], word_group2.boundingbox[3]) - max(
        word_group1.boundingbox[1], word_group2.boundingbox[1]
    )

    if overlap > 0:
        return True
    if abs(overlap / min_height) < 1.5:
        print("near enough", abs(overlap / min_height), overlap, min_height)
        return True
    return False


def calculate_iou_and_near(wg1: Word_group, wg2: Word_group):
    min_height = min(
        wg1.boundingbox[3] - wg1.boundingbox[1], wg2.boundingbox[3] - wg2.boundingbox[1]
    )
    overlap = min(wg1.boundingbox[3], wg2.boundingbox[3]) - max(
        wg1.boundingbox[1], wg2.boundingbox[1]
    )
    iou = overlap / min_height
    distance = min(
        abs(wg1.boundingbox[0] - wg2.boundingbox[2]),
        abs(wg1.boundingbox[2] - wg2.boundingbox[0]),
    )
    if iou > 0.7 and distance < 0.5 * (wg1.boundingboxp[2] - wg1.boundingbox[0]):
        return True
    return False


def construct_word_groups_to_kie_label(list_word_groups: list):
    kie_dict = dict()
    for wg in list_word_groups:
        if wg.kie_label == "other":
            continue
        if wg.kie_label not in kie_dict:
            kie_dict[wg.kie_label] = [wg]
        else:
            kie_dict[wg.kie_label].append(wg)

    new_dict = dict()
    for key, value in kie_dict.items():
        if len(value) == 1:
            new_dict[key] = value
            continue

        value.sort(key=lambda x: x.boundingbox[1])
        new_dict[key] = value
    return new_dict


def invoice_construct_word_groups_to_kie_label(list_word_groups: list):
    kie_dict = dict()

    for wg in list_word_groups:
        if wg.kie_label == "other":
            continue
        if wg.kie_label not in kie_dict:
            kie_dict[wg.kie_label] = [wg]
        else:
            kie_dict[wg.kie_label].append(wg)

    return kie_dict


def postprocess_total_value(kie_dict):
    if "total_in_words_value" not in kie_dict:
        return kie_dict

    for k, value in kie_dict.items():
        if k == "total_in_words_value":
            continue
        l = []
        for v in value:
            if v.boundingbox[3] <= kie_dict["total_in_words_value"][0].boundingbox[3]:
                l.append(v)

        if len(l) != 0:
            kie_dict[k] = l

    return kie_dict


def postprocess_tax_code_value(kie_dict):
    if "buyer_tax_code_value" in kie_dict or "seller_tax_code_value" not in kie_dict:
        return kie_dict

    kie_dict["buyer_tax_code_value"] = []
    for v in kie_dict["seller_tax_code_value"]:
        if "buyer_name_key" in kie_dict and (
            v.boundingbox[3] > kie_dict["buyer_name_key"][0].boundingbox[3]
            or near(v, kie_dict["buyer_name_key"][0])
        ):
            kie_dict["buyer_tax_code_value"].append(v)
            continue

        if "buyer_name_value" in kie_dict and (
            v.boundingbox[3] > kie_dict["buyer_name_value"][0].boundingbox[3]
            or near(v, kie_dict["buyer_name_value"][0])
        ):
            kie_dict["buyer_tax_code_value"].append(v)
            continue

        if "buyer_address_value" in kie_dict and near(
            kie_dict["buyer_address_value"][0], v
        ):
            kie_dict["buyer_tax_code_value"].append(v)
    return kie_dict


def postprocess_tax_code_key(kie_dict):
    if "buyer_tax_code_key" in kie_dict or "seller_tax_code_key" not in kie_dict:
        return kie_dict
    kie_dict["buyer_tax_code_key"] = []
    for v in kie_dict["seller_tax_code_key"]:
        if "buyer_name_key" in kie_dict and (
            v.boundingbox[3] > kie_dict["buyer_name_key"][0].boundingbox[3]
            or near(v, kie_dict["buyer_name_key"][0])
        ):
            kie_dict["buyer_tax_code_key"].append(v)
            continue

        if "buyer_name_value" in kie_dict and (
            v.boundingbox[3] > kie_dict["buyer_name_value"][0].boundingbox[3]
            or near(v, kie_dict["buyer_name_value"][0])
        ):
            kie_dict["buyer_tax_code_key"].append(v)
            continue

        if "buyer_address_value" in kie_dict and near(
            kie_dict["buyer_address_value"][0], v
        ):
            kie_dict["buyer_tax_code_key"].append(v)

    return kie_dict


def invoice_postprocess(kie_dict: dict):
    # all keys or values which are below total_in_words_value will be thrown away
    kie_dict = postprocess_total_value(kie_dict)
    kie_dict = postprocess_tax_code_value(kie_dict)
    kie_dict = postprocess_tax_code_key(kie_dict)
    return kie_dict


def throw_overlapping_words(list_words):
    new_list = [list_words[0]]
    for word in list_words:
        overlap = False
        area = (word.boundingbox[2] - word.boundingbox[0]) * (
            word.boundingbox[3] - word.boundingbox[1]
        )
        for word2 in new_list:
            area2 = (word2.boundingbox[2] - word2.boundingbox[0]) * (
                word2.boundingbox[3] - word2.boundingbox[1]
            )
            xmin_intersect = max(word.boundingbox[0], word2.boundingbox[0])
            xmax_intersect = min(word.boundingbox[2], word2.boundingbox[2])
            ymin_intersect = max(word.boundingbox[1], word2.boundingbox[1])
            ymax_intersect = min(word.boundingbox[3], word2.boundingbox[3])
            if xmax_intersect < xmin_intersect or ymax_intersect < ymin_intersect:
                continue

            area_intersect = (xmax_intersect - xmin_intersect) * (
                ymax_intersect - ymin_intersect
            )
            if area_intersect / area > 0.7 or area_intersect / area2 > 0.7:
                overlap = True
        if overlap == False:
            new_list.append(word)
    return new_list


class Box:
    def __init__(self, xmin=0, ymin=0, xmax=0, ymax=0, label="", kie_label=""):
        self.xmax = xmax
        self.ymax = ymax
        self.xmin = xmin
        self.ymin = ymin
        self.label = label
        self.kie_label = kie_label


def check_iou(box1: Word, box2: Box, threshold=0.9):
    area1 = (box1.boundingbox[2] - box1.boundingbox[0]) * (
        box1.boundingbox[3] - box1.boundingbox[1]
    )
    area2 = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)
    xmin_intersect = max(box1.boundingbox[0], box2.xmin)
    ymin_intersect = max(box1.boundingbox[1], box2.ymin)
    xmax_intersect = min(box1.boundingbox[2], box2.xmax)
    ymax_intersect = min(box1.boundingbox[3], box2.ymax)
    if xmax_intersect < xmin_intersect or ymax_intersect < ymin_intersect:
        area_intersect = 0
    else:
        area_intersect = (xmax_intersect - xmin_intersect) * (
            ymax_intersect - ymin_intersect
        )
    union = area1 + area2 - area_intersect
    iou = area_intersect / union
    if iou > threshold:
        return True
    return False
