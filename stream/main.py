import os
from time import sleep
import cv2

STREAM = 0

IMAGE_SHOW = False

if __name__ == "__main__":
    print("stream")
    cap = cv2.VideoCapture(STREAM)

    while True:
        ret, frame = cap.read()

        if frame is not None:
            # TODO: Do frame preprocessing (check if image is black, currupted, etc.)
            cv2.imwrite("./frame_temp.jpg", frame)
            os.system("mv frame_temp.jpg frame.jpg")

            if IMAGE_SHOW:
                cv2.imshow('frame',frame)
                if cv2.waitKey(22) & 0xFF == ord('q'):
                    break