import unittest

import cv2
import numpy as np

from server.reference import describe_shift, estimate_shift


def scene():
    """Mesa sintética: fondo claro, alfombrilla oscura, tres cubos y cintas."""
    image = np.full((480, 640, 3), 235, np.uint8)
    cv2.rectangle(image, (240, 60), (560, 440), (40, 40, 40), -1)
    for y, color in [(80, (150, 150, 150)), (190, (160, 40, 20)), (330, (245, 245, 245))]:
        cv2.circle(image, (170, y), 65, color, -1)
    for x, y in [(510, 60), (560, 430), (120, 440)]:
        cv2.rectangle(image, (x - 30, y - 15), (x + 30, y + 15), (120, 120, 120), -1)
    return image


class ReferenceTests(unittest.TestCase):
    def test_measures_shift_and_rotation(self):
        reference = scene()
        moved = cv2.warpAffine(reference, np.float32([[1, 0, 15], [0, 1, -8]]), (640, 480), borderMode=cv2.BORDER_REPLICATE)
        shift = estimate_shift(reference, moved)
        self.assertAlmostEqual(shift["dx"], 15, delta=1.5)
        self.assertAlmostEqual(shift["dy"], -8, delta=1.5)
        self.assertIn("15 px a la derecha", describe_shift(shift)["message"])
        turned = cv2.warpAffine(reference, cv2.getRotationMatrix2D((320, 240), -2, 1.0), (640, 480), borderMode=cv2.BORDER_REPLICATE)
        self.assertAlmostEqual(estimate_shift(reference, turned)["angle"], 2, delta=0.3)

    def test_same_view_is_aligned(self):
        self.assertTrue(describe_shift(estimate_shift(scene(), scene()))["aligned"])
