import os
from time import time
import cv2

STREAM = "rtsp://admin:lapisco123@192.168.8.107/LiveMedia/ch1/Media1"
IMAGE_SHOW = False

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

if __name__ == "__main__":
    while True:
        if STREAM[:5] == "rtsp": # Cameras streamings
            cap = cv2.VideoCapture(STREAM, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        else: cap = cv2.VideoCapture(STREAM) # Other streamings

        print("monitor")

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
            else:
                cap.release()
                time.sleep(10)
                break
