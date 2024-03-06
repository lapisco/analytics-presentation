import os
from time import time
import cv2

STREAM = 0

IMAGE_SHOW = False

if __name__ == "__main__":
    print("stream")
    cap = cv2.VideoCapture(STREAM)

    while True:
        ret, frame = cap.read()

        # Start timer
        new_frame_time = time()

        if frame is not None:
            # TODO: Do frame preprocessing (check if image is black, currupted, etc.)
            cv2.imwrite("./frame_temp.jpg", frame)
            os.system("mv frame_temp.jpg frame.jpg")

            # Stop timer
            prev_frame_time = time()
            fps = 1/(prev_frame_time-new_frame_time)

            if IMAGE_SHOW:
                cv2.imshow('frame', frame)
                if cv2.waitKey(22) & 0xFF == ord('q'):
                    break
