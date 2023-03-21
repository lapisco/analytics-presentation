import cv2
from time import time

IMG_FR = "../face-recognition/frame.jpg"
IMG_OD = "../object-detector/frame.jpg"
IMG_ED = "../ppe-detection/frame.jpg"
IMG_OT = "../object-tracking/frame.jpg"
IMG_OA = "../ocr-analysis/frame.jpg"
IMG_PE = "../pose-estimation/frame.jpg"
IMG_PEY = "../pose-estimation-yolo/frame.jpg"
IMG_PS = "../person-stay/frame.jpg"

IMG_OVERLAY = "./overlay.png"

BOXES = [[160, 560, 20, 770], [630, 1030, 20, 770],
         [160, 560, 820, 1570], [630, 1030, 820, 1570]]

TITLES = [((250, 150), "FACE RECOGITION"), ((1050, 150), "PERSON TRACKING"),
          ((250, 620), "OBJECT DETECTION"), ((1050, 620), "PPE DETECTION")]

font = cv2.FONT_HERSHEY_SIMPLEX

if __name__ == "__main__":
    print("presentation")
    while True:
        # Start timer
        new_frame_time = time()

        frame_fr = cv2.imread(IMG_FR)

        frame_od = cv2.imread(IMG_OD)

        frame_ed = cv2.imread(IMG_ED)

        frame_ot = cv2.imread(IMG_OT)

        frame_oa = cv2.imread(IMG_OA)

        frame_pe = cv2.imread(IMG_PE)

        frame_pey = cv2.imread(IMG_PEY)

        frame_ps = cv2.imread(IMG_PS)

        frame_over = cv2.imread(IMG_OVERLAY)

        frames_ans = [frame_fr, frame_od, frame_ps, frame_ed]

        for i, frame_an in enumerate(frames_ans):
            frame_an = cv2.resize(frame_an, (750, 400))
            frame_over[BOXES[i][0]:BOXES[i][1],
                       BOXES[i][2]:BOXES[i][3], :] = frame_an

            (w, h), _ = cv2.getTextSize(
                TITLES[i][1], cv2.FONT_HERSHEY_SIMPLEX, 1, 3)

            # Prints the text.
            img = cv2.rectangle(frame_over, (TITLES[i][0][0], TITLES[i][0][1] - h - 10),
                                (TITLES[i][0][0] + w, TITLES[i][0][1]), (255, 255, 255), -1)
            img = cv2.putText(frame_over, TITLES[i][1], (TITLES[i][0][0], TITLES[i][0][1] - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            "window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Stop timer
        prev_frame_time = time()
        fps = 1/(prev_frame_time-new_frame_time)
        print("FPS: ", "FPS: {:.2f}".format(fps))

        cv2.imshow("window", frame_over)

        if cv2.waitKey(22) & 0xFF == ord('q'):
            break
