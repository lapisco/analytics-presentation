from lib.object_detector import ObjectDetector
from lib.ocsort_tracker.ocsort import OCSort
from copy import copy
import numpy as np
import os
import cv2

IMAGE_SHOW = False

od = ObjectDetector()
tracker = OCSort(det_thresh=0.6, iou_threshold=0.3, use_byte=False)

if __name__ == '__main__':
    print("object-tracking")

    while True:
        frame = cv2.imread("../stream/frame.jpg")
        dets = od.detect(frame)
        height, width = frame.shape[:2]
        frame_processed = copy(frame)
        outputs = []
        for i in range(len(dets)):
            outputs.append([max(0, dets[i]['topleft']['x'])/width, max(0, dets[i]['topleft']['y'])/height, min(dets[i]['bottomright']['x'], frame.shape[1])/width, min(dets[i]['bottomright']['y'], frame.shape[0])/height, float(dets[i]['confidence'])/100])

        if len(outputs):
            # TODO: Make labels reassignment
            outputs = np.array([outputs])
            online_targets = tracker.update(outputs[0][:,:5], [height, width], (height, width))
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                frame_processed = cv2.rectangle(frame_processed, (int(tlwh[0]*width), int(tlwh[1]*height)), (int((tlwh[0]+tlwh[2])*width), int((tlwh[1]+tlwh[3])*height)), (0,255,0), 2)
                frame_processed = cv2.putText(frame_processed, str(int(tid)), (int(tlwh[0]*width), int(tlwh[1]*height)-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imwrite("./frame_temp.jpg", frame_processed)
        os.system("mv frame_temp.jpg frame.jpg")

        if IMAGE_SHOW:
            cv2.imshow('frame',frame_processed)
            if cv2.waitKey(22) & 0xFF == ord('q'):
                break