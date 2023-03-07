import face_recognition
from time import time
import imutils
import pickle
import cv2
import os
import dlib

IMAGE_SHOW = True

data = pickle.loads(open("./encodings.pickle", "rb").read())

detector = dlib.cnn_face_detection_model_v1("resources/mmod_human_face_detector.dat")

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])

	# return our bounding box coordinates
	return (startY, endX, endY, startX)

font = cv2.FONT_HERSHEY_SIMPLEX

if __name__ == "__main__":
    print("face-recognition")
    # loop over frames from the video file stream
    while True:
        frame = cv2.imread("../stream/frame.jpg")

        # Start timer
        new_frame_time = time()

        # convert the input frame from BGR to RGB then resize it
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=600)
        r = frame.shape[1] / float(rgb.shape[1])

        start = time()
        results = detector(rgb, 0)
        end = time()
        boxes = [convert_and_trim_bb(frame, r.rect) for r in results]
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

        # Stop timer
        prev_frame_time = time()
        fps = 1/(prev_frame_time-new_frame_time)

        cv2.putText(frame, "FPS: {:.2f}".format(fps), (5, 25), font,
                    1, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imwrite("./frame_temp.jpg", frame)
        os.system("mv frame_temp.jpg frame.jpg")
