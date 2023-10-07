import numpy as np
from typing import List, Tuple
import os
from utils.bbox import BBox
from .text_detector import TextDetector


class DocumentReader:
    def __init__(self):
        pass

    def read(self, img: np.ndarray) -> Tuple[np.ndarray, List[List[BBox]]]:
        lines = self._segmentation_lines(img)
        return img, lines

    def _segmentation_lines(self, img) -> List[List[BBox]]:
        # HOT to use
        '''
        checkpoint_path = "путь до папки с весами для модели детекции"
        image: np.ndarray
        '''

        checkpoint_path = r"C:\Users\danii\YandexDisk\Работа\03.Исходники_программ\веса"
        # 3 - segmentation words into lines

        text_detector = TextDetector(on_gpu=False, checkpoint_path=checkpoint_path,
                                     with_vertical_text_detection=False,
                                     config={})
        # 2 - detect text
        boxes, conf = text_detector.predict(img)
        lines = text_detector.sort_bboxes_by_coordinates(boxes)
        return self.union_lines(lines)

    @staticmethod
    def union_lines(lines: List[List[BBox]]) -> List[List[BBox]]:
        filtered_lines = []
        one_id = 0
        while one_id < len(lines) - 1:
            one_line = lines[one_id]
            two_line = lines[one_id + 1]
            merge_line = one_line + two_line
            min_h_one = min([box.y_bottom_right - box.y_top_left for box in one_line])
            min_h_two = min([box.y_bottom_right - box.y_top_left for box in two_line])

            one_bottom = max([box.y_bottom_right for box in one_line])
            two_top = min([box.y_top_left for box in two_line])

            interval_between_lines = two_top - one_bottom
            if interval_between_lines < 0 or (min_h_one > min_h_two and interval_between_lines < min_h_two / 2):
                union_line = sorted(merge_line, key=lambda x: x.x_top_left)
                filtered_lines.append(union_line)
                one_id += 2
            else:
                filtered_lines.append(one_line)
                one_id += 1

        return filtered_lines

