from typing import List

import numpy as np
from pytesseract import pytesseract

from doc_img_utils.bbox import BBox


class TesseractReaderConfig:
    def __init__(self, lang="rus"):
        self.lang = lang
        pass

    def get_args_str(self) -> str:
        args_str = f"-l {self.lang}"
        return args_str


class TesseractReader:
    def __init__(self, conf: TesseractReaderConfig = None):
        self.config = conf

    def read(self, image: np.ndarray) -> List[BBox]:
        tesseract_bboxes = pytesseract.image_to_data(
            config=self.config.get_args_str(),
            image=image,
            output_type=pytesseract.Output.DICT)
        list_bbox = []
        for index_bbox, level in enumerate(tesseract_bboxes["level"]):
            if level == 5:
                x_top_left = tesseract_bboxes["left"][index_bbox]
                y_top_left = tesseract_bboxes["top"][index_bbox]
                width = tesseract_bboxes["width"][index_bbox]
                height = tesseract_bboxes["height"][index_bbox]
                list_bbox.append(BBox(x_top_left, y_top_left, width, height))
        return list_bbox


