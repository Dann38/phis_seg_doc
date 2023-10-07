from abc import ABC, abstractmethod
from typing import List

import numpy as np

from utils.bbox import BBox


class BaseBoldClassifier(ABC):
    @abstractmethod
    def classify(self, image: np.ndarray,  bboxes: List[BBox]) -> List[float]:
        pass


