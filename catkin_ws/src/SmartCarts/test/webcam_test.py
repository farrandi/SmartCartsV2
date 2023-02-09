import cv2
import os

dir = os.path.dirname(__file__)
vid_filename = os.path.join(dir, 'zigzag_1.mov')

cam = cv2.VideoCapture(vid_filename)
w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cam.get(cv2.CAP_PROP_FPS)
frame_count = cam.get(cv2.CAP_PROP_FRAME_COUNT)
print w,h,fps, frame_count