from ultralytics import YOLO
from functools import partial
from collections import defaultdict
from facial_emotions import HSEmotionRecognizer
from concurrent.futures import ThreadPoolExecutor

import os
import cv2
import time
import dlib
import numpy as np

emotions = {
    "Anger": {
        "color": (193, 69, 42)
    },
    "Contempt": {
        "color": (164, 175, 49)
    },
    "Disgust": {
        "emotion": "Disgust",
        "color": (40, 52, 155)
    },
    "Fear": {
        "color": (23, 164, 28)
    },
    "Happiness": {
        "color": (164, 93, 23)
    },
    "Neutral": {
        "color": (218, 229, 97)
    },
    "Sadness": {
        "emotion": "Sadness",
        "color": (108, 72, 200)
    },
    "Surprise": {
        "color": (88, 158, 38)
    }
}


# Load the YOLOv8 model
model = YOLO('yolov8n-face.pt')

model_name = 'enet_b0_8_best_afew'
fer = HSEmotionRecognizer(model_name=model_name)

predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
facial_recognition_model = dlib.face_recognition_model_v1(
    "./dlib_face_recognition_resnet_model_v1.dat")

jitters = 11
max_embeddings = 10
persons_counter = 0
embeddings_global_list = []

padding = 10
font = cv2.FONT_HERSHEY_SIMPLEX

# Store the track history and track start times
track_history = defaultdict(lambda: [])
track_start_times = defaultdict(lambda: 0)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageProto = "./age_deploy.prototxt"
ageModel = "./age_net.caffemodel"
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']

ageNet = cv2.dnn.readNet(ageModel, ageProto)


def age_classifier(face):
    # blob = cv2.dnn.blobFromImages([face], 1.5, (227, 227), MODEL_MEAN_VALUES, swapRB=False) # FIXME: batch
    blob = cv2.dnn.blobFromImage(
        face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]

    return age


def euclidean_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)


def extract_face_embeddings(frame, faceBox):
    embeddings_list = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceBoxDlib = dlib.rectangle(
        faceBox[0], faceBox[1], faceBox[2], faceBox[3])
    landmarks = predictor(gray, faceBoxDlib)

    face_embedding = facial_recognition_model.compute_face_descriptor(
        frame, landmarks)

    face_embedding_np = np.array(face_embedding)

    embeddings_list.append(face_embedding_np)

    return embeddings_list


def process_faces(frame, box, track_id, track_history):
    global persons_counter, embeddings_global_list

    x, y, w, h = box.cpu().numpy()
    track = track_history[track_id]
    track.append((float(x), float(y)))  # x, y center point

    # Check if it's a new track, and record the start time
    if len(track) == 1:
        track_start_times[track_id] = time.time()

    if len(track) > 30:  # retain 90 tracks for 90 frames
        track.pop(0)

    # Crop and store the detected face

    tly, bry, tlx, brx = int(y - h/2), int(y + h/2), int(x - w/2), int(x + w/2)
    face = frame[max(0, tly - padding):min(bry + padding, frame.shape[0] - 1),
                 max(0, tlx - padding):min(brx + padding, frame.shape[1] - 1)]

    faceBox = [tly, bry, tlx, brx]
    embeddings = extract_face_embeddings(frame, faceBox)
    # embeddings = fr.face_encodings(cv2.cvtColor(face, cv2.COLOR_BGR2RGB),
    #                                  boxes, num_jitters=jitters)

    for embedding in embeddings:
        if len(embeddings_global_list) == 0:
            embeddings_global_list.append(embedding)
            persons_counter += 1
        else:
            new_face = True
            for stored_embedding in embeddings_global_list:
                distance = euclidean_distance(embedding, stored_embedding)
                # print(distance)
                if distance < 0.55:
                    new_face = False
                    break
            if new_face:
                if len(embeddings_global_list) == max_embeddings:
                    embeddings_global_list.pop(0)
                embeddings_global_list.append(embedding)
                persons_counter += 1

    emotion_prediction, emotion_probability = fer.predict_emotions(
        face, logits=False)

    if (np.max(emotion_probability) > 0.36):
        ages = age_classifier(face)

        age_text = f'{ages[1:-1]} anos'
        color = emotions[emotion_prediction]['color']

        frame = cv2.rectangle(
            frame, (tlx, tly), (tlx + int(w), tly + int(h)), color, 2)
        frame = cv2.line(frame, (tlx, tly + int(h)), (tlx + 20, tly + int(h) + 20),
                         color,
                         thickness=2)

        frame = cv2.putText(frame, f'{emotion_prediction}',
                            (tlx + 25, tly + int(h) +
                             36), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)

        frame = cv2.putText(frame, age_text,
                            (tlx + 25, tly + int(h) +
                             56), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)

        # Calculate the time duration the detection stayed on the screen
        current_time = time.time()
        duration = current_time - track_start_times[track_id]

        time_text = f"{duration:.2f}s"
        frame = cv2.putText(frame, time_text,
                            (tlx + 25, tly + int(h) +
                             76), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
    else:
        color = (255, 255, 255)
        frame = cv2.rectangle(
            frame, (tlx, tly), (tlx + int(w), tly + int(h)), color, 2)

    return frame


while True:
    frame = cv2.imread("../stream/frame.jpg")
    # frame = cv2.resize(frame, None, fx=0.75, fy=0.75)

    # Start timer
    new_frame_time = time.time()
    
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True, conf=0.5, verbose=False)

    # Get the boxes and track IDs
    boxes = results[0].boxes.xywh.cpu()
        
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Process faces in parallel
        with ThreadPoolExecutor() as executor:
            processed_frames = list(executor.map(
                partial(process_faces, track_history=track_history),
                [frame] * len(track_ids), boxes, track_ids))

        frame = processed_frames[0]

    prev_frame_time = time.time()
    fps = 1/(prev_frame_time-new_frame_time)
        
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (5, 25), font,
                    1, (255, 0, 0), 1, cv2.LINE_AA)
    
    text = f"Visitantes: {persons_counter}"
    text_size = cv2.getTextSize(text, font, 1, 1)[0]
    text_width, text_height = text_size[0], text_size[1]
    text_x = (frame.shape[1] - text_width) // 2
    cv2.putText(frame, text, (text_x, 25), font,
                1, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite("./frame_temp.jpg", frame)
    os.system("mv frame_temp.jpg frame.jpg")