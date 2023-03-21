from copy import copy
import os
from lib.ppe_processing import PPE
import cv2
from time import time

IMAGE_SHOW = False

ppe = PPE()

font = cv2.FONT_HERSHEY_SIMPLEX

if __name__ == '__main__':
    print("ppe-detector")

    while True:
        frame = cv2.imread("../stream/frame.jpg")

        # Start timer
        new_frame_time = time()

        dets = ppe.detect(frame)

        frame_processed = copy(frame)
        for i in range(len(dets)):
            label = dets[i]['label']
            y_topleft = max(0, dets[i]['topleft']['y'])
            y_bottomright = min(dets[i]['bottomright']['y'], frame.shape[0])
            x_topleft = max(0, dets[i]['topleft']['x'])
            x_bottomright = min(dets[i]['bottomright']['x'], frame.shape[1])
            color = dets[i]['color']

            frame_processed = cv2.rectangle(
                frame_processed, (x_topleft, y_topleft), (x_bottomright, y_bottomright), color, 2)
            frame_processed = cv2.putText(
                frame_processed, label, (x_topleft, y_topleft-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Stop timer
        prev_frame_time = time()
        fps = 1/(prev_frame_time-new_frame_time)

        cv2.putText(frame_processed, "FPS: {:.2f}".format(fps), (5, 25), font,
                    1, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imwrite("./frame_temp.jpg", frame_processed)
        os.system("mv frame_temp.jpg frame.jpg")

        if IMAGE_SHOW:
            cv2.imshow('frame', frame_processed)
            if cv2.waitKey(22) & 0xFF == ord('q'):
                break
