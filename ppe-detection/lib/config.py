import os

lib_path = os.path.dirname(os.path.dirname(__file__))

# Options to lib/configure Yolo
YOLO_OPTIONS = {
    "model": os.path.join(lib_path, "resources/ppe.cfg"),
    "load": os.path.join(lib_path, "resources/ppe.weights"),
    "labels": os.path.join(lib_path, "resources/ppe.names"),
    'meta': os.path.join(lib_path, "resources/ppe.data"),
    "threshold": 0.5,
}