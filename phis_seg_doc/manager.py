import numpy as np
import cv2

from phis_seg_doc.doc_img_utils.bold_classifier import PsBoldClassifier, MeanBoldClassifier, MedianBoldClassifier
from phis_seg_doc.doc_img_utils.tesseract_reader import TesseractReader, TesseractReaderConfig
from phis_seg_doc.doc_img_utils.marker import Marker
from phis_seg_doc.db_manager import ManagerDB


class Manager:
    def __init__(self):
        self.db_manager = ManagerDB()
        self.db_manager.open_db()
        self.marker = Marker()
        conf_doc_reader = TesseractReaderConfig(lang="rus")
        self.doc_reader = TesseractReader(conf_doc_reader)
        self.classifiers = {1: MeanBoldClassifier(),
                            2: MedianBoldClassifier(),
                            3: PsBoldClassifier()}

    def set_img(self, img: bytes):
        id_image = self.db_manager.add_row_image_id()
        self.db_manager.add_origin_image(id_image, img)
        return id_image

    def classify(self, id_image: int, method: int, coef: float):
        # Получить IMG
        img_cv2 = self.get_origin_image_id(id_image)

        # Настройка классификатора
        classifier = self.classifiers[method]
        classifier.clusterizer.significance_level = coef
        self.db_manager.add_set_classifier(id_image, method, coef)

        # Классификация и разметка
        words = self.doc_reader.read(img_cv2)
        style = classifier.classify(img_cv2, words)
        img_cv2 = self.marker.mark(img_cv2, words, style)

        bytes_img = self.img2bytes(img_cv2)
        self.db_manager.add_result_image(id_image, bytes_img)

    def get_result_bytes_image_id(self, id_image: int) -> bytes:
        image_bytes = self.db_manager.get_row_image_id(id_image)[1]
        return image_bytes

    def get_origin_bytes_image_id(self, id_image: int) -> bytes:
        image_bytes = self.db_manager.get_row_image_id(id_image)[0]
        return image_bytes

    def get_origin_image_id(self, id_image: int) -> np.ndarray:
        image_bytes = self.get_origin_bytes_image_id(id_image)
        chunk_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_cv2 = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
        return img_cv2

    def img2bytes(self, image: np.ndarray) -> bytes:
        return bytearray(cv2.imencode(".jpg", image)[1])

    def get_list_id(self) -> list[int]:
        rez = self.db_manager.get_id_10_last()
        return rez

    def get_method_and_coef(self, id_image: int):
        return self.db_manager.get_method_and_coef(id_image)

