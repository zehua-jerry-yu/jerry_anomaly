import cv2
import os
import numpy as np


def align_image(template_path, target_path, output_path):  # generated by gpt
    template = cv2.imread(template_path)
    target = cv2.imread(target_path)

    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)

    kp1, des1 = orb.detectAndCompute(gray_template, None)  #mask
    kp2, des2 = orb.detectAndCompute(gray_target, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    h, w = gray_template.shape
    aligned_target = cv2.warpPerspective(target, H, (w, h))

    gray_aligned = cv2.cvtColor(aligned_target, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray_aligned, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = aligned_target[y:y+h, x:x+w]

    cv2.imwrite(output_path, cropped_image)


def align_pcb():
    # PATH_DATA = "/root/autodl-tmp/kaggle/anomaly/pcb2/mixed"
    # PATH_TEMPLATE = "/root/autodl-tmp/kaggle/anomaly/pcb2/template/template.JPG"
    # PATH_OUT = "/root/autodl-tmp/kaggle/anomaly/pcb2/aligned"
    PATH_DATA = "/root/autodl-tmp/kaggle/anomaly/Jerry_PCB2_FullTest/mixed"
    PATH_TEMPLATE = "/root/autodl-tmp/kaggle/anomaly/Jerry_PCB2_FullTest/template/template.JPG"
    PATH_OUT = "/root/autodl-tmp/kaggle/anomaly/Jerry_PCB2_FullTest/aligned"

    filenames = os.listdir(PATH_DATA)
    filenames = [f for f in filenames if f.endswith(".JPG")]
    for filename in filenames:
        path_image = os.path.join(PATH_DATA, filename)
        path_out = os.path.join(PATH_OUT, filename)
        align_image(PATH_TEMPLATE, path_image, path_out)


if __name__ == "__main__":
    align_pcb()