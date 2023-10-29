import numpy as np
import os
import cv2

from doc_img_utils.bold_classifier import PsBoldClassifier
from doc_img_utils.tesseract_reader import TesseractReader, TesseractReaderConfig
from doc_img_utils.marker import Marker
from db_manager import ManagerDB


class Manager:
    def __init__(self):
        self.img = None
        self.words = None
        self.style = None

        self.db_manager = ManagerDB()
        self.db_manager.open_db()

    def get_hello(self):
        self.db_manager.first_start()
        return "H I !!!"

    def set_img(self, img: bytes):
        id_image = self.db_manager.add_origin_image(img)
        return id_image

    def classify(self, id_image: int):
        print(id_image)
        # Получить IMG
        image_bytes = self.db_manager.get_row_image_id(id_image)[0]
        chunk_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_cv2 = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)

        # Обработка
        conf_doc_reader = TesseractReaderConfig(lang="rus")
        doc_reader = TesseractReader(conf_doc_reader)
        classifier = PsBoldClassifier()
        classifier.clusterizer.significance_level = 0.50
        marker = Marker()
        self.words = doc_reader.read(img_cv2)
        self.style = classifier.classify(img_cv2, self.words)
        img_cv2 = marker.mark(img_cv2, self.words, self.style)

        bytes_img = bytearray(cv2.imencode(".jpg", img_cv2)[1])
        print(bytes_img)
        self.db_manager.add_result_image(id_image, bytes_img)

    def get_result_image_id(self, id_image: int):
        return self.db_manager.get_row_image_id(id_image)[1]

    def save_img(self):
        file_ = "temp.jpg"
        if os.path.exists(file_):
            os.remove(file_)
        cv2.imwrite(file_, self.img)
        return




