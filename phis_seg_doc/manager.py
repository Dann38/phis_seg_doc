import numpy as np
import cv2

from doc_img_utils.bold_classifier import PsBoldClassifier
from doc_img_utils.tesseract_reader import TesseractReader, TesseractReaderConfig
from doc_img_utils.marker import Marker
from db_manager import ManagerDB


class Manager:
    def __init__(self):
        self.db_manager = ManagerDB()
        self.db_manager.open_db()
        self.marker = Marker()
        conf_doc_reader = TesseractReaderConfig(lang="rus")
        self.doc_reader = TesseractReader(conf_doc_reader)

    def set_img(self, img: bytes):
        id_image = self.db_manager.add_row_image_id()
        self.db_manager.add_origin_image(id_image, img)
        return id_image

    def classify(self, id_image: int):
        # Получить IMG
        img_cv2 = self.get_origin_image_id(id_image)

        # Настройка классификатора
        classifier = PsBoldClassifier()
        classifier.clusterizer.significance_level = 0.50

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
        return self.db_manager.get_id_10_last()



