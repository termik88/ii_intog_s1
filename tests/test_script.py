from seleniumbase import BaseCase
import cv2
import time

#Сравнение двух скриншотов Страницы. Так же производим сравнение по RGB
class ComponentsTest(BaseCase):
    def test_basic(self):

        # open the app and take a screenshot
        self.open("https://termik88-ii-itog-s1-streamlit-app-r1ykkj.streamlit.app/")

        time.sleep(10)  # give leaflet time to load from web
        self.save_screenshot("current-screenshot.png")

        # test screenshots look exactly the same
        original = cv2.imread(
            'archived_logs/1/screenshot.png'
            #"archived_logs/logs_1674870189/test_script.ComponentsTest.test_basic/screenshot.png"
        )
        duplicate = cv2.imread("current-screenshot.png")
        assert original.shape == duplicate.shape

        difference = cv2.subtract(original, duplicate)
        b, g, r = cv2.split(difference)
        assert cv2.countNonZero(b) == cv2.countNonZero(g) == cv2.countNonZero(r) == 0