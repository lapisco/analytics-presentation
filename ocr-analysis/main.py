from transformers import pipeline
from translate import Translator
from copy import copy
import pytesseract
import cv2
import os

IMAGE_SHOW = False

classifier = pipeline('sentiment-analysis')

translator = Translator(from_lang="pt", to_lang="en")


if __name__ == '__main__':
    print("ocr-analysis")

    while True:
        frame = cv2.imread("../stream/frame.jpg")
        frame_processed = copy(frame)
        print(frame_processed.shape)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ocr_result = pytesseract.image_to_string(frame_rgb, lang='por').replace("\n", " ").replace("  ", " ")
        ocr_result = translator.translate(ocr_result)

        if ocr_result:
            analysis_result = classifier(ocr_result)[0]
            frame_processed = cv2.putText(frame_processed, analysis_result['label'], (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imwrite("./frame_temp.jpg", frame_processed)
        os.system("mv frame_temp.jpg frame.jpg")

        if IMAGE_SHOW:
            cv2.imshow('frame',frame_processed)
            if cv2.waitKey(22) & 0xFF == ord('q'):
                break