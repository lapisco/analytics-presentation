from time import time
from transformers import pipeline
from copy import copy
import pytesseract
import cv2
import os

IMAGE_SHOW = False

classifier = pipeline('sentiment-analysis')

if __name__ == '__main__':
    print("ocr-analysis")

    while True:
        frame = cv2.imread("../stream/frame.jpg")
        frame_processed = copy(frame)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
        ocr_result = pytesseract.image_to_string(frame_rgb, lang='por').replace("\n", " ").replace("  ", " ")
        
        if len(ocr_result) > 20:
            analysis_result = classifier(ocr_result)[0]
            frame_processed = cv2.putText(frame_processed, analysis_result['label'], (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imwrite("./frame_temp.jpg", frame_processed)
        os.system("mv frame_temp.jpg frame.jpg")

        if IMAGE_SHOW:
            cv2.imshow('frame',frame_processed)
            if cv2.waitKey(22) & 0xFF == ord('q'):
                break