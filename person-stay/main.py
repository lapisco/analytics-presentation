import numpy as np
import copy
import cv2
import random
from scipy.spatial import distance
from lib.object_detector import ObjectDetector
import time
from lib.sort.sort import Sort
import os

od = ObjectDetector()

tracker = Sort()
memory = {}

IMAGE_SHOW = False

font = cv2.FONT_HERSHEY_SIMPLEX

def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]


def gen_colors(classes):
    colors = {}
    for class_ in classes:
        colors[class_] = tuple(int(''.join([random.choice(
            '0123456789ABCDEF') for j in range(6)])[i:i+2], 16) for i in (0, 2, 4))

    return colors

colors = gen_colors(od.classes)

start_time = 0

def display_detections(image, detections, colors):
    global tracker, memory, start_time
    img_out = copy.copy(image)
    frame = copy.copy(image)

    labels = []
    dets = []
    centroid_dets = []
    dets_dict = {}

    for i in range(len(detections)):
        lb = detections[i]['label']
        if lb == 'person':
            tlx = int(detections[i]['topleft']['x'])
            tly = int(detections[i]['topleft']['y'])
            brx = int(detections[i]['bottomright']['x'])
            bry = int(detections[i]['bottomright']['y'])
            conf = detections[i]['confidence']
            x = int((tlx + brx) / 2)
            y = int((tly + bry) / 2)
            centroid_dets.append((x, y))
            dets_dict[(x, y)] = lb
            labels.append(lb)
            dets.append([tlx, tly, brx, bry, conf])

    if len(centroid_dets) > 0:
        dets = np.asarray(dets).astype(np.float)
        tracks = tracker.update(dets)
        boxes = []
        indexIDs = []
        previous = memory.copy()
        memory = {}

        for j, track in enumerate(tracks):
            # AM: Print ID - Class - Confidence
            (x1, y1) = (int(track[0]), int(track[1]))
            (x2, y2) = (int(track[2]), int(track[3]))
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)

            # track[0] a [3] -> bounding box
            # track[4] -> ID
            boxes.append([track[0], track[1], track[2], track[3], int(
                track[4]), dets_dict[closest_node((x, y), centroid_dets)], dets[j, -1]])
            indexIDs.append(int(track[4]))

            memory[indexIDs[-1]] = boxes[-1]
            if indexIDs[-1] in previous.keys():
                if len(previous[indexIDs[-1]]) > 7:
                    end_time = time.time()
                    memory[indexIDs[-1]
                           ].append((end_time - start_time) + previous[indexIDs[-1]][-1])
            else:
                end_time = time.time()
                memory[indexIDs[-1]].append(start_time - end_time)

            box = boxes[-1]
            tlx = int(box[0])
            tly = int(box[1])
            brx = int(box[2])
            bry = int(box[3])
            lb = box[5]

            img_out = cv2.rectangle(frame, (tlx, tly),
                                    (brx, bry), (0, 255, 0), 3)

            cv2.circle(img_out, (x, y), 48, (0, 255, 0), -1)

            TEXT = "{:.2f}s".format(box[-1])

            text_size, _ = cv2.getTextSize(
                TEXT, cv2.FONT_HERSHEY_DUPLEX, 1, 2)
            text_origin = (
                int(x - text_size[0] / 2), int(y + text_size[1] / 2))

            cv2.putText(img_out, TEXT, text_origin, cv2.FONT_HERSHEY_DUPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

            putText = ''

            putText = f'{box[-4]-1} - {lb}'
            (w, h), _ = cv2.getTextSize(
                putText, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            if tly < 20:
                img_out = cv2.rectangle(img_out, (tlx-1, bry + h + 10),
                                        (tlx-1 + w, bry), (0, 255, 0), -1)
                img_out = cv2.putText(img_out, putText, (tlx, bry + 15),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                img_out = cv2.rectangle(img_out, (tlx-1, tly - h - 15),
                                        (tlx-1 + w, tly), (0, 255, 0), -1)
                img_out = cv2.putText(img_out, putText, (tlx, tly - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return img_out


if __name__ == '__main__':
    print("person-stay")

    while True:
        start_time = time.time()
        frame = cv2.imread("../stream/frame.jpg")

        # Start timer
        new_frame_time = time.time()

        detections = od.detect(frame)

        frame_processed = display_detections(frame, detections, colors)

        # Stop timer
        prev_frame_time = time.time()
        fps = 1/(prev_frame_time-new_frame_time)

        cv2.putText(frame_processed, "FPS: {:.2f}".format(fps), (5, 25), font,
                    1, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imwrite("./frame_temp.jpg", frame_processed)
        os.system("mv frame_temp.jpg frame.jpg")

        if IMAGE_SHOW:
            cv2.imshow('frame', frame_processed)
            if cv2.waitKey(22) & 0xFF == ord('q'):
                break