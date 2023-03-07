import os

lib_path = os.path.dirname(os.path.dirname(__file__))

# Options to configure Yolo to detect objects
YOLO_OPTIONS = {
    "model": os.path.join(lib_path, "resources/yolov4-tiny.cfg"),
    "load": os.path.join(lib_path, "resources/yolov4-tiny.weights"),
    "labels": os.path.join(lib_path, "resources/coco.names"),
    'meta': os.path.join(lib_path, "resources/coco.data"),
    "threshold": 0.5,
}