from keras.models import load_model
from scipy.spatial.distance import cdist
from lib.sort.sort import Sort
import numpy as np
import time
import cv2
import os
import dlib

IMAGE_SHOW = True

tracker = Sort()
memory = {}
start_time = 0

faceProto="./opencv_face_detector.pbtxt"

faceModel="./opencv_face_detector_uint8.pb"

ageProto = "./age_deploy.prototxt"

ageModel = "./age_net.caffemodel"

predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

facial_recognition_model = dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)

emotion_offsets = (20, 40)
emotions = {
    0: {
        "emotion": "Angry",
        "color": (193, 69, 42)
    },
    1: {
        "emotion": "Disgust",
        "color": (164, 175, 49)
    },
    2: {
        "emotion": "Fear",
        "color": (40, 52, 155)
    },
    3: {
        "emotion": "Happy",
        "color": (23, 164, 28)
    },
    4: {
        "emotion": "Sad",
        "color": (164, 93, 23)
    },
    5: {
        "emotion": "Suprise",
        "color": (218, 229, 97)
    },
    6: {
        "emotion": "Neutral",
        "color": (108, 72, 200)
    }
}

emotionModelPath = './emotionModel.hdf5'  # fer2013_mini_XCEPTION.110-0.65
emotionClassifier = load_model(emotionModelPath, compile=False)
emotionTargetSize = emotionClassifier.input_shape[1:3]

embeddings_global_list = []
max_embeddings = 10
persons_counter = 0

def euclidean_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

def highlightFace(net, frame, conf_threshold=0.8):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
    return faceBoxes

def extract_face_embeddings(frame, faceBox):
    embeddings_list = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceBoxDlib = dlib.rectangle(faceBox[0], faceBox[1], faceBox[2], faceBox[3])
    landmarks = predictor(gray, faceBoxDlib)

    face_embedding = facial_recognition_model.compute_face_descriptor(frame, landmarks)

    face_embedding_np = np.array(face_embedding)

    embeddings_list.append(face_embedding_np)

    return embeddings_list

def age_classifier(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]

    return age

def emotion_classifier(grayFace):
    grayFace = cv2.resize(grayFace, (emotionTargetSize))

    grayFace = grayFace.astype('float32')
    grayFace = grayFace / 255.0
    grayFace = (grayFace - 0.5) * 2.0
    grayFace = np.expand_dims(grayFace, 0)
    grayFace = np.expand_dims(grayFace, -1)
    emotion_prediction = emotionClassifier.predict(grayFace)
    emotion_probability = np.max(emotion_prediction)

    return emotion_probability, emotion_prediction

def closest_node(node, nodes):
    closest_index = cdist([node], nodes).argmin()
    return nodes[closest_index]

def track_detections(frame, detections):
    global tracker, memory, start_time, persons_counter, embeddings_global_list
    img_out = frame.copy()
    boxes_times = []
    labels = []
    dets = []
    centroid_dets = []
    dets_dict = {}

    for i in range(len(detections)):
        lb = 'face'
        if lb == 'face':
            tlx = int(detections[i][0])
            tly = int(detections[i][1])
            brx = int(detections[i][2])
            bry = int(detections[i][3])

            x = int((tlx + brx) / 2)
            y = int((tly + bry) / 2)
            centroid_dets.append((x, y))
            dets_dict[(x, y)] = lb
            labels.append(lb)
            dets.append([tlx, tly, brx, bry])

    if len(centroid_dets) > 0:
        dets = np.asarray(dets).astype(float)
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
            faceBox = [tlx, tly, brx, bry]
            face=frame[max(0,tly-padding):
                    min(bry+padding,frame.shape[0]-1),max(0,tlx-padding)
                    :min(brx+padding, frame.shape[1]-1)]
            try:
                grayFace = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            except:
                continue

            embeddings = extract_face_embeddings(frame, faceBox)

            for embedding in embeddings:
                if len(embeddings_global_list) == 0:
                    embeddings_global_list.append(embedding)
                    persons_counter += 1
                else:
                    new_face = True
                    for stored_embedding in embeddings_global_list:
                        distance = euclidean_distance(embedding, stored_embedding)
                        print(distance)
                        if distance < 0.6:
                            new_face = False
                            break
                    if new_face:
                        if len(embeddings_global_list) == max_embeddings:
                            embeddings_global_list.pop(0)
                        embeddings_global_list.append(embedding)
                        persons_counter += 1

            age = age_classifier(face)

            emotion_probability, emotion_prediction = emotion_classifier(grayFace)

            x1, y1, x2, y2 = faceBox
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1

            if (emotion_probability > 0.36):
                emotion_label_arg = np.argmax(emotion_prediction)
                color = emotions[emotion_label_arg]['color']
                
                img_out = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                img_out = cv2.line(img_out, (x, y + h), (x + 20, y + h + 20),
                        color,
                        thickness=2)
                img_out = cv2.rectangle(img_out, (x + 20, y + h + 40), (x + 110, y + h + 40),
                            color, -1)
                img_out = cv2.putText(img_out, emotions[emotion_label_arg]['emotion'],
                            (x + 25, y + h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                age_text = f'{age[1:-1]} anos'
                text_size = cv2.getTextSize(age_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                img_out = cv2.putText(img_out, age_text,
                    (x + 25, y + h + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
                
                time_text = "{:.2f}s".format(box[-1] / 10000000000)
                time_text_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                img_out = cv2.putText(img_out, time_text,
                    (x + 25, y + h + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
            else:
                color = (255, 255, 255)
                img_out = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    return img_out

font = cv2.FONT_HERSHEY_SIMPLEX
padding=20
if __name__ == "__main__":
    print("face-recognition")
    # loop over frames from the video file stream
    while True:
        frame = cv2.imread("../stream/frame.jpg")
        
        # Start timer
        new_frame_time = time.time()
        try:
            # convert the input frame from BGR to RGB then resize it
            faceBoxes=highlightFace(faceNet,frame)
            processed_frame = track_detections(frame, faceBoxes)


            prev_frame_time = time.time()
            fps = 1/(prev_frame_time-new_frame_time)

            cv2.putText(frame, "FPS: {:.2f}".format(fps), (5, 25), font,
                        1, (255, 0, 0), 1, cv2.LINE_AA)
            
            text = f"Visitantes: {persons_counter}"
            text_size = cv2.getTextSize(text, font, 1, 1)[0]
            text_width, text_height = text_size[0], text_size[1]
            text_x = (frame.shape[1] - text_width) // 2
            cv2.putText(processed_frame, text, (text_x, 25), font,
                        1, (255, 0, 0), 1, cv2.LINE_AA)
        except:
            continue

        cv2.imwrite("./frame_temp.jpg", processed_frame)
        os.system("mv frame_temp.jpg frame.jpg")
