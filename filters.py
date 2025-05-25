import cv2
import numpy as np

def cartoonify(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 9, 10)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def pencil_sketch(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray_img
    blur = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blur = 255 - blur
    sketch = cv2.divide(gray_img, inverted_blur, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def sepia_filter(img):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)

def stylize_filter(img):
    return cv2.stylization(img, sigma_s=60, sigma_r=0.07)

def list_styles():
    return {
        "Cartoon": cartoonify,
        "Pencil Sketch": pencil_sketch,
        "Sepia": sepia_filter,
        "Stylized": stylize_filter
    }
