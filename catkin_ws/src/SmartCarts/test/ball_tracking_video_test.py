## PURPOSE: To optimize the ball tracking algorithm from PySense blog
## Written by Svena Yu, Nov 2020

from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
import cv2
import time
import os
import sys
import time
import csv

redLower = (130, 130, 0)
redUpper = (255, 255, 255)
# redLower = (154,161,0)
# redUpper = (255,255,255)
dir = os.path.dirname(__file__)
vid_filename = os.path.join(dir, 'front_2_back.mp4')
np.set_printoptions(threshold=sys.maxsize)

x_list = []
y_list = []
radius_list = []
timestamp_list = []

def contour_filter(img):
	blurred = cv2.GaussianBlur(img, (11, 11), 0)
	# blurred = cv2.medianBlur(img,9) ## need to use odd number 	
	# blurred = cv2.bilateralFilter(img,12,125,125)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, redLower, redUpper)
	return mask

def find_hough_lines(gray_img, img):
	edges = cv2.Canny(gray_img,50,150,apertureSize = 3)
	minLineLength = 5
	maxLineGap = 0
	minIntersectingPts = 50
	lines = cv2.HoughLinesP(edges,1,np.pi/180,minIntersectingPts,minLineLength,maxLineGap)
	## ILLUSTRATION PURPOSES
	# for line in lines:
	# 	for x1,y1,x2,y2 in line:
	# 		cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
	# cv2.imshow("houghlines", img)
	return lines

def ball_tracker():
	vs = cv2.VideoCapture(vid_filename)
	# allow vid file to warm up
	time.sleep(2.0)
	while True:
		flag,frame = vs.read()

		if frame is None:
			break
		# frame = frame.astype('uint8')
		timestamp_list.append(vs.get(cv2.CAP_PROP_POS_MSEC))
		print(vs.get(cv2.CAP_PROP_POS_FRAMES))

		# Filtering
		mask = contour_filter(frame)
		
		# # Finding Contours, Finding Hough Lines, intersecting contours are ignored
		# # Method 1
		image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Find max contour area
		i = 0
		maxContour = 0
		maxContourArea = 0
		for contour in contours:
			contourArea = cv2.contourArea(contour)
			if contourArea > maxContourArea:
				maxContourArea = contourArea
				maxContour = i
			i+=1

		((x, y), radius) = cv2.minEnclosingCircle(contours[maxContour])

		if radius > 0:
			cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
			cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

			x_list.append(x)
			y_list.append(y)
			radius_list.append(radius)
		
		# Display Image
		# cv2.imshow('image', frame)  
		# Maintain output window utill user presses a key 
		key = cv2.waitKey(1) & 0XFF
		if key == ord('q'):
			break  
	
	# If not using vid file, stop the camera video stream
	vs.release()

	# Destroying present windows on screen 
	cv2.destroyAllWindows()
	
	# Plot Graph
	plt.figure(figsize=(10, 5))
	plt.plot(timestamp_list, x_list)
	plt.title("x")
	plt.figure(figsize=(10, 5))
	plt.plot(timestamp_list, y_list)
	plt.title("y")

	# Save Data
	data = np.asarray([x_list, y_list, radius_list, timestamp_list])
	# print(data)
	header = ['x','y','radius','time']

	data_filename = os.path.join(dir, 'front2back.csv')
	np.savetxt(data_filename, np.transpose(data), delimiter=",") 


if __name__ == '__main__': 
	ball_tracker();

