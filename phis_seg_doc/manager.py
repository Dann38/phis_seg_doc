import numpy as np
import os
import cv2

from doc_img_utils.bold_classifier import PsBoldClassifier
from doc_img_utils.tesseract_reader import TesseractReader, TesseractReaderConfig
from doc_img_utils.marker import Marker


class Manager:
    def __init__(self):
        self.img = None
        self.words = None
        self.style = None

    def get_hello(self):
        return "H I !!!"

    def set_img(self, img: bytes):
        chunk_arr = np.frombuffer(img, dtype=np.uint8)
        self.img = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)

    def classify(self):
        conf_doc_reader = TesseractReaderConfig(lang="rus")
        doc_reader = TesseractReader(conf_doc_reader)
        classifier = PsBoldClassifier()
        classifier.clusterizer.significance_level = 0.50
        marker = Marker()
        self.words = doc_reader.read(self.img)
        self.style = classifier.classify(self.img, self.words)
        self.img = marker.mark(self.img, self.words, self.style)

    def save_img(self):
        file_ = "temp.jpg"
        if os.path.exists(file_):
            os.remove(file_)
        cv2.imwrite(file_, self.img)
        return




