import cv2
from time import time

IMG_AGE = "../age-emotion-detector/frame.jpg"
# IMG_FR = "../face-recognition/frame.jpg"
# IMG_OD = "../object-detector/frame.jpg"
# IMG_ED = "../ppe-detection/frame.jpg"
# IMG_OT = "../object-tracking/frame.jpg"
# IMG_OA = "../ocr-analysis/frame.jpg"
# IMG_PE = "../pose-estimation/frame.jpg"
# IMG_PEY = "../pose-estimation-yolo/frame.jpg"
# IMG_PS = "../person-stay/frame.jpg"

IMG_OVERLAY = "./overlay.png"

TITLES = [("FACE RECOGITION", (200, 150)), ("PERSON TRACKING", (1100, 150)),
          ("OBJECT DETECTION", (200, 650)), ("PPE DETECTION", (1100, 650))]

font = cv2.FONT_HERSHEY_SIMPLEX

if __name__ == "__main__":
    print("presentation")
    while True:
        # Start timer
        new_frame_time = time()

        frame_age = cv2.imread(IMG_AGE)

        # frame_fr = cv2.imread(IMG_FR)

        # frame_od = cv2.imread(IMG_OD)

        # frame_ed = cv2.imread(IMG_ED)

        # frame_ot = cv2.imread(IMG_OT)

        # frame_oa = cv2.imread(IMG_OA)

        # frame_pe = cv2.imread(IMG_PE)

        # frame_pey = cv2.imread(IMG_PEY)

        # frame_ps = cv2.imread(IMG_PS)

        frame_over = cv2.imread(IMG_OVERLAY)

        # frames_ans = [frame_fr, frame_od, frame_ps, frame_ed]
        frames_ans = [frame_age]

        for i, frame_an in enumerate(frames_ans):
            h, w, _ = frame_over.shape
            square_height = h // 2 + 300  # Define a altura do retângulo aumentada em 100 pixels
            square_width = int(w * 0.575)  # Define a largura do retângulo
            x = (w - square_width) // 2 + 250  # Desloca o retângulo para a direita em 50 pixels
            y = (h - square_height) // 2 - 100  # Calcula a coordenada y do canto superior esquerdo do retângulo

            frame_an = cv2.resize(frame_an, (square_width, square_height))
            frame_over[y:y + square_height, x:x + square_width, :] = frame_an

            (text_w, text_h), _ = cv2.getTextSize(
                TITLES[i][0], font, 1, 3)

            # Prints the text.
            img = cv2.rectangle(frame_over, (TITLES[i][1][0], TITLES[i][1][1] - text_h - 10),
                                (TITLES[i][1][0] + text_w, TITLES[i][1][1]), (255, 255, 255), -1)
            img = cv2.putText(frame_over, TITLES[i][0], (TITLES[i][1][0], TITLES[i][1][1] - 5),
                              font, 1, (0, 0, 0), 3)

        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            "window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Stop timer
        prev_frame_time = time()
        fps = 1 / (prev_frame_time - new_frame_time)
        print("FPS: ", "FPS: {:.2f}".format(fps))

        cv2.imshow("window", frame_over)

        if cv2.waitKey(22) & 0xFF == ord('q'):
            break
