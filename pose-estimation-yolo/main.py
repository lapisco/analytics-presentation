import numpy as np
from utils.torch_utils import select_device
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import colors, plot_one_box
import cv2
from time import time
import os
IMAGE_SHOW = False

torch.cuda.empty_cache()
total_memory = torch.cuda.get_device_properties(0).total_memory

device = select_device('0')
half = device.type != 'cpu'
weights = './resources/yolov7-w6-pose.pt'

model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())
if half:
    model.half()  # to FP16

names = model.module.names if hasattr(
    model, 'module') else model.names  # get class names

img_size = 320
rect = False
conf_thres = 0.25
iou_thres = 0.45
augment = False
classes = None
agnostic_nms = False
kpt_label = True
hide_labels = True
hide_conf = True
line_thickness = 3

font = cv2.FONT_HERSHEY_SIMPLEX

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    print(new_shape)
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def detection_and_show(img):
    global total_memory
    img0 = img.copy()

    # Letterbox
    img = letterbox(img0, img_size, auto=False,
                    stride=stride)[0]

    # Stack
    img = np.stack(img, 0)

    # Convert
    # BGR to RGB, to bsx3x416x416
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=augment)[0]
    print(pred[..., 4].max())
    pred = non_max_suppression(pred, conf_thres, iou_thres,
                               classes=classes, agnostic=agnostic_nms, kpt_label=kpt_label)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        im0 = img0.copy()
        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
            scale_coords(img.shape[2:], det[:, 6:],
                         im0.shape, kpt_label=kpt_label, step=3)

            # Write results
            for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                # Add bbox to image
                c = int(cls)  # integer class
                label = None if hide_labels else (
                    names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                kpts = det[det_index, 6:]
                plot_one_box(xyxy, im0, label=label, color=colors(
                    c, True), line_thickness=line_thickness, kpt_label=kpt_label, kpts=kpts, steps=3, orig_shape=im0.shape[:2])

    del pred, img
    torch.cuda.empty_cache()
    # torch.empty(total_memory // 2, dtype=torch.int8, device='cuda')
    return im0

if __name__ == '__main__':
    print("pose-estimation-yolo")

    while True:
        frame = cv2.imread("../stream/frame.jpg")

        # Start timer
        new_frame_time = time()

        with torch.no_grad():
            frame_processed = detection_and_show(frame)
        
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
