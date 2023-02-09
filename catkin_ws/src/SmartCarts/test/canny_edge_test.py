## from https://docs.opencv.org/3.4/da/d5c/tutorial_canny_detector.html

from __future__ import print_function
import cv2 as cv
import argparse
import os

max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3
dir = os.path.dirname(__file__)
# img_filename = os.path.join(dir, 'ball_overlap.jpg')
img_filename = os.path.join(dir, 'ball_overlap_fx_onlymask.png')

def CannyThreshold(val):
    low_threshold = val
    img_blur = cv.blur(src_gray, (3,3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))
    cv.imshow(window_name, dst)

# parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
# parser.add_argument('--input', help='Path to input image.', default=img_filename)
# args = parser.parse_args()
src = cv.imread(img_filename)
if src is None:
    print('Could not open or find the image: ', args.input)
    exit(0)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.namedWindow(window_name)
cv.createTrackbar(title_trackbar, window_name , 0, max_lowThreshold, CannyThreshold)
CannyThreshold(0)
cv.waitKey()