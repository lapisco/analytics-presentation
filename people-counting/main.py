import os
import cv2
import csv
import time
import cvzone
import numpy as np
import pandas as pd

from tracker import *
from heatmap import HeatMap
from ultralytics import YOLO


def main():
    # Load model
    verbose = False
    model = YOLO('./yolov8s.pt')

    # Load Tracker class
    tracker = Tracker()

    # Load HeatMap class
    heat_map = HeatMap()

    # Load initial frame
    initial_frame = cv2.imread("../monitor/frame.jpg")
    
    frame_shape = np.shape(initial_frame)

    # Get object classes
    file = open('./coco.names', 'r')
    data = file.read()
    class_list = data.split('\n')

    # Variables
    count = 0
    counter1 = []
    persondown = {}

    personup = {}
    counter2 = []

    cy1 = 400
    cy2 = 425

    offset = 6
    alpha = 0.1

    black_image = np.zeros(frame_shape, dtype=np.uint64)
    accumulated_image = np.zeros(
        (frame_shape[0], frame_shape[1]), dtype=np.uint64)

    # output_heatmap_path = './heatmap_images'
    output_csv_path = './dados.csv'

    time_to_save_heatmap = 60*2  # in seconds
    last_increment_time = time.time()
    last_csv_update_time = time.time()

    while True:
        frame = cv2.imread("../monitor/frame.jpg")


        count += 1
        if (count % 3 != 0):
            continue

        results = model.predict(frame, verbose=verbose)

        a = results[0].boxes.data
        try:
            px = pd.DataFrame(a.cpu().numpy()).astype("float")
        except Exception as e:
            print(f"Error: {e}")

        list = []

        for index, row in px.iterrows():
            d = int(row[5])
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])

            c = class_list[d]
            if 'person' in c:
                list.append([x1, y1, x2, y2])

        bbox_id = tracker.update(list)
        for bbox in bbox_id:

            x3, y3, x4, y4, id = bbox
            cx = int(x3 + x4) // 2
            cy = int(y3 + y4) // 2

            cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)

            # for down going
            if (cy1 < (cy + offset) and (cy1 > cy - offset)):

                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                persondown[id] = (cx, cy)

            if (id in persondown):
                if (cy2 < (cy+offset) and (cy2 > cy-offset)):
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                    cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                    if counter1.count(id) == 0:
                        counter1.append(id)

            # for up going
            if (cy2 < (cy+offset) and (cy2 > cy-offset)):

                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                personup[id] = (cx, cy)

            if (id in personup):
                if (cy1 < (cy+offset) and (cy1 > cy-offset)):
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                    cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                    if counter2.count(id) == 0:
                        counter2.append(id)

            # Create heat map
            accumulated_image[y3:y4, x3:x4] += 1

            current_time = time.time()
            if current_time - last_increment_time >= time_to_save_heatmap:
                HEATMAP = heat_map.plot_heatmap(
                    accumulated_image, black_image, alpha, color_map=cv2.COLORMAP_JET)

                cv2.imwrite(os.path.join(
                    './', f'heatmap_{time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))}.png'), HEATMAP)

                last_increment_time = current_time

        cv2.line(frame, (0, cy1), (frame_shape[1], cy1), (0, 255, 0), 2)
        cv2.line(frame, (0, cy2), (frame_shape[1], cy2), (0, 255, 255), 2)

        downcount = len(counter1)
        upcount = len(counter2)
        inside = upcount - downcount

        if inside < 0:
            inside = 0

        cvzone.putTextRect(frame, f'Down: {downcount}', (50, 60), 2, 2)
        cvzone.putTextRect(frame, f'Up: {upcount}', (50, 160), 2, 2)

        # Atualizar o arquivo CSV a cada hora
        current_time = time.time()
        if current_time - last_csv_update_time >= 10:  # 3600 segundos = 1 hora
            with open(output_csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([time.strftime(
                    '%Y-%m-%d %H:%M:%S', time.localtime()), downcount, upcount, inside])
            last_csv_update_time = current_time

        cv2.imwrite("./frame_temp.jpg", frame)
        os.system("mv frame_temp.jpg frame.jpg")

if __name__ == '__main__':
    main()
