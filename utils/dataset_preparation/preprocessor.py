from abc import ABC, abstractmethod
import math
import cv2

import numpy as np
import sys

from pathlib import Path
sys.path.append(Path(__file__).resolve().parents[2].as_posix())

from containers import Rect2i


class Preprocessor(ABC):
    @abstractmethod
    def process(self, image: np.ndarray, bounding_box: Rect2i) -> np.ndarray:
        raise NotImplementedError
    
    def __call__(self, image: np.ndarray, bounding_box: Rect2i) -> np.ndarray:
        return self.process(image, bounding_box)
    

class PreprocessorBoxExpansion(Preprocessor):
    def __init__(self, area_coeff: float = 9.0):
        self.area_coeff = area_coeff
        
    def process(self, image: np.ndarray, bounding_box: Rect2i) -> np.ndarray:
        return self._preprocess(image, bounding_box, self.area_coeff)
    
    def _preprocess(self, image: np.ndarray, bounding_box: Rect2i, area_coeff: float = 9.0) -> np.ndarray:
        """Crops the face area via a square with the area of bounding_box are multiplied by the area_coeff

        Args:
            image (np.ndarray): Original image
            bounding_box (Rect2i): Detection bounding box

        Returns:
            np.ndarray: Image, cropped with enlarged bounding box
        """
        # cut_box
        area_out = bounding_box.area * area_coeff
        lout = math.sqrt(area_out)

        dw = int((lout - bounding_box.width) / 2)
        dh = int((lout - bounding_box.height) / 2)

        cut_x1 = bounding_box.x1 - dw
        cut_y1 = bounding_box.y1 - dh
        cut_x2 = bounding_box.x2 + dw
        cut_y2 = bounding_box.y2 + dh

        # Check out of bounds
        rows, cols, _ = image.shape
        margin_left = - cut_x1 if cut_x1 < 0 else 0
        margin_top = - cut_y1 if cut_y1 < 0 else 0
        margin_right = cut_x2 - cols if cut_x2 > cols else 0
        margin_bottom = cut_y2 - rows if cut_y2 > rows else 0
        
        image_expanded = cv2.copyMakeBorder(
            src=image,
            top=margin_top,
            bottom=margin_bottom,
            left=margin_left,
            right=margin_right,
            borderType=cv2.BORDER_REPLICATE
            )
        
        # Shift the bounding box to keep it within the image
        cut_x1 += margin_left
        cut_y1 += margin_top
        cut_x2 += margin_left
        cut_y2 += margin_top

        image_cropped = image_expanded[cut_y1:cut_y2, cut_x1:cut_x2]
        return image_cropped
