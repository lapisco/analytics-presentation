import random
from lib import config
import copy
import os
import sys

os.environ['DARKNET_PATH'] = os.getcwd()+'/lib/darknet'
sys.path.append(os.environ['DARKNET_PATH'])

import lib.darknet.darknet as darknet

def hex2rgb(h):
    return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

class ObjectDetector(object):
    def __init__(self):
        self.net = darknet.load_net_custom(config.YOLO_OPTIONS['model'].encode(
            "ascii"), config.YOLO_OPTIONS['load'].encode("ascii"), 0, 1)
        self.meta = darknet.load_meta(config.YOLO_OPTIONS['meta'].encode(
            "ascii"))
        self.classes_file = config.YOLO_OPTIONS['labels']
        self.classes = []
        with open(self.classes_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if '\n' in line:
                    line = line.replace('\n','')
                self.classes.append(line)
        f.close()
        number_of_colors = len(self.classes)

        self.colors = [hex2rgb("#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
                    for i in range(number_of_colors)]

        self.threshold = config.YOLO_OPTIONS['threshold']
        self.darknet_image = None
        self.darknet_image_flag = False

    def detect(self, image):
        self.darknet_image = darknet.make_image(image.shape[1], image.shape[0], 3)
        darknet.copy_image_from_bytes(self.darknet_image, image[..., ::-1].tobytes())
        detections = darknet.detect_image(self.net, self.classes, self.darknet_image, thresh=self.threshold)
        detections = self.list_to_dict(detections)
        darknet.free_image(self.darknet_image)
        return detections
    
    def convertBack(self, x, y, w, h):
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def list_to_dict(self, list_yolo):
        result_dict = [None]*len(list_yolo)
        result_temp = {}
        for i, item in enumerate(list_yolo):
            x_item, y_item, w_item, h_item = item[2][0], item[2][1], item[2][2], item[2][3]
            result_temp['label'] = item[0]
            result_temp['confidence'] = item[1]
            tlx, tly, brx, bry = self.convertBack(
                float(x_item), float(y_item), float(w_item), float(h_item))
            result_temp['topleft'] = {}
            result_temp['topleft']['x'] = tlx
            result_temp['topleft']['y'] = tly
            result_temp['bottomright'] = {}
            result_temp['bottomright']['y'] = bry
            result_temp['bottomright']['x'] = brx
            result_temp["color"] = self.colors[self.classes.index(result_temp['label'])]
            result_dict[i] = copy.deepcopy(result_temp)
        return result_dict