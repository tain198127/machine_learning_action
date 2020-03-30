from tesstlog import Log

logger = Log.init_log(__name__, False)
import sys
import cv2
import pytesseract
class testcv:
    def totxt(self):
        pytesseract.pytesseract.tesseract_cmd ='/usr/local/Cellar/tesseract/4.1.1/bin/tesseract'
        img = cv2.imread('/Users/danebrown/Desktop/realtest.jpg')
        text = pytesseract.image_to_string(img,lang='chi_sim')
        print(text)