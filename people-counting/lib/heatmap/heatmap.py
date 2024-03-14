import cv2
import numpy as np


class HeatMap:
    def __init__(self) -> None:
        pass

    def plot_heatmap(self, accumulated_image, black_image, background_heatmap, alpha=0.1, color_map=cv2.COLORMAP_HOT, kernel=(15, 15), apply_blur=True):
        norm_image = (accumulated_image - accumulated_image.min()) / (
            accumulated_image.max() - accumulated_image.min() + 1e-8) * 255
        norm_image = norm_image.astype('uint8')

        if apply_blur:
            norm_image = cv2.GaussianBlur(norm_image, kernel, 0)

        black_image = cv2.applyColorMap(norm_image, color_map)
        black_image = black_image.astype(background_heatmap.dtype)  # Convert to the same dtype

        black_image[:, :, 0][black_image[:, :, 1] == 0] = 0
        black_image[:, :, 0][black_image[:, :, 2] == 0] = 0

        alpha_frame_gray = cv2.cvtColor(background_heatmap, cv2.COLOR_BGR2GRAY)

        accumulated_image = alpha * alpha_frame_gray + (1 - alpha) * accumulated_image

        finalimg = cv2.addWeighted(black_image, 0.7, background_heatmap, 0.3, 0)

        return np.asarray(finalimg, np.uint8)

    
    def iterative_heatmap(self, accumulated_image, image, alpha=0.1, color_map=cv2.COLORMAP_HOT, kernel=(15, 15), apply_blur=True):

        # Normalize and blur the heatmap
        norm_image = (accumulated_image - accumulated_image.min()) / (
                accumulated_image.max() - accumulated_image.min() + 1e-8) * 255
        norm_image = norm_image.astype('uint8')
        norm_image = cv2.GaussianBlur(norm_image, (25, 25), 0)

        # Apply color map to create heatmap visualization
        heatmap = cv2.applyColorMap(norm_image, color_map)

        # Resize video frame for heatmap averaging
        alpha_frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform frame averaging for heatmap smoothing
        accumulated_image = alpha * alpha_frame_gray + (1 - alpha) * accumulated_image

        # Create a final image by blending the heatmap and original video frame
        finalimg = cv2.addWeighted(heatmap, 0.7, image, 0.3, 0)

        return np.asarray(finalimg, np.uint8)
