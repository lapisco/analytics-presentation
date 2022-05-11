import cv2
import imutils

IMG_FR = "../face-recognition/frame.jpg"
IMG_OD = "../object-detector/frame.jpg"
IMG_OT = "../object-tracking/frame.jpg"

IMG_OVERLAY = "./overlay.jpg"

BOXES = [[20, 357, 40, 640], [357, 694, 40, 640], [20, 357, 640, 1240], [357, 694, 640, 1240]]

if __name__ == "__main__":
    print("presentation")
    while True:
        frame_fr = cv2.imread(IMG_FR)
        frame_fr = imutils.resize(frame_fr, width=600)
        
        frame_od = cv2.imread(IMG_OD)
        frame_od = imutils.resize(frame_od, width=600)

        frame_ot = cv2.imread(IMG_OT)
        frame_ot = imutils.resize(frame_ot, width=600)

        frame_over = cv2.imread(IMG_OVERLAY)

        frames_ans = [frame_fr, frame_od, frame_ot, frame_od]

        for i, frame_an in enumerate(frames_ans):
            
            frame_over[BOXES[i][0]:BOXES[i][1],BOXES[i][2]:BOXES[i][3], :] = frame_an

        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("window", frame_over)
        cv2.waitKey(1)