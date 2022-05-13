from lib.visualization import draw_prediction_on_image
from lib.pose_estimation import movenet
import tensorflow as tf
from time import time
import numpy as np
import cv2
import os

IMAGE_SHOW = False

input_size = 192

if __name__ == '__main__':
    print("pose-estimation")

    while True:
        # Load the input image.
        image_path = '../stream/frame.jpg'
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image)

        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        input_image = tf.expand_dims(image, axis=0)
        input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

        # Run model inference.
        start = time()
        keypoints_with_scores = movenet(input_image)

        # Visualize the predictions with image.
        display_image = tf.expand_dims(image, axis=0)
        display_image = tf.cast(tf.image.resize_with_pad(
            display_image, 1280, 1280), dtype=tf.int32)

        output_overlay = draw_prediction_on_image(
            np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores, output_image_height=1080)

        output_overlay = output_overlay[240:825, 30:-10, ::-1]
        cv2.imwrite("./frame_temp.jpg", output_overlay)
        os.system("mv frame_temp.jpg frame.jpg")

        if IMAGE_SHOW:
            cv2.imshow('frame',output_overlay)
            if cv2.waitKey(22) & 0xFF == ord('q'):
                break