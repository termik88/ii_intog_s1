import os
from seleniumbase import BaseCase
import cv2
import time


class ComponentsTest(BaseCase):
    def test_basic(self):

        # open the app and take a screenshot
        self.open("http://localhost:8501")

        time.sleep(1)  # give leaflet time to load from web
        self.save_screenshot("current-screenshot.png")

        # test screenshots look exactly the same
        original = cv2.imread(
            'ii_itog_s1/tests/latest_logs/test_script.ComponentsTest.test_basic/screenshot.png'
            #"archived_logs/logs_1674870189/test_script.ComponentsTest.test_basic/screenshot.png"
        )
        duplicate = cv2.imread("current-screenshot.png")
        assert original.shape == duplicate.shape

        difference = cv2.subtract(original, duplicate)
        b, g, r = cv2.split(difference)
        assert cv2.countNonZero(b) == cv2.countNonZero(g) == cv2.countNonZero(r) == 0