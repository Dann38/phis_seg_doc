import cv2
import numpy as np
from typing import List
from doc_img_utils.bbox import BBox

OFFSET_ROW = 2
BOLD_ROW = 1
REGULAR_ROW = 0

COLOR_BOLD_ROW = (255, 0, 0)
COLOR_OFFSET_ROW = (0, 0, 255)
COLOR_REGULAR_ROW = (0, 255, 0)


class Marker:
    def mark(self, img: np.ndarray, bboxes: List[BBox], style: List[float]) -> np.ndarray:
        h = img.shape[0]
        w = img.shape[1]

        img_cope = img.copy()

        coef = h / w
        exist_style = len(style) != 0

        for j in range(len(bboxes)):
            border = 1
            word = bboxes[j]
            color = (155, 155, 155)

            if exist_style:
                style_word = style[j]
                if style_word == BOLD_ROW:
                    color = COLOR_BOLD_ROW
                elif style_word == OFFSET_ROW:
                    color = COLOR_OFFSET_ROW
                elif style_word == REGULAR_ROW:
                    color = COLOR_REGULAR_ROW
            cv2.rectangle(img_cope, (word.x_top_left, word.y_top_left),
                          (word.x_bottom_right, word.y_bottom_right), color, border)
        return img_cope
