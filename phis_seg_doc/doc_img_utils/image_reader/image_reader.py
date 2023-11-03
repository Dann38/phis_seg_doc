import numpy as np
import cv2


class ImageReader:
    def read(self, path_image: str) -> np.ndarray:
        with open(path_image, "rb") as f:
            chunk = f.read()
        chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
        image = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
        return image

