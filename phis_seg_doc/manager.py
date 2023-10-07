import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from utils.bold_classifier import PsBoldClassifier
from utils.document_reader import DocumentReader
from utils.marker import Marker

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
        doc_reader = DocumentReader()
        classifier = PsBoldClassifier()
        marker = Marker()
        img, lines = doc_reader.read(self.img)
        self.words = [word for line in lines for word in line]
        print(self.words)
        self.style = classifier.classify(self.img, self.words)
        self.img = marker.mark(self.img, self.words, self.style)

    def save_img(self):
        file_ = "temp.jpg"
        if os.path.exists(file_):
            os.remove(file_)
        cv2.imwrite(file_, self.img)
        return




