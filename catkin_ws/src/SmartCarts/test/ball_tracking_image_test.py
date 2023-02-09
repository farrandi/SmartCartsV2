## PURPOSE: To optimize the ball tracking algorithm from PySense blog
## Written by Svena Yu, Nov 2020

from collections import deque
import numpy as np
import math
import argparse
import cv2
import time
import os
import sys

redLower = (130, 130, 0)
redUpper = (255, 255, 255)
# redLower = (154,161,0)
# redUpper = (255,255,255)
dir = os.path.dirname(__file__)
img_filename = os.path.join(dir, 'ball_faraway.jpg')
np.set_printoptions(threshold=sys.maxsize)

def contour_filter(img):
	# blurred = cv2.GaussianBlur(img, (11, 11), 0)
	# blurred = cv2.medianBlur(img,9) ## need to use odd number 	
	blurred = cv2.bilateralFilter(img,12,125,125)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, redLower, redUpper)
	return mask

def houghcircles_filter(img):
	blurred = cv2.bilateralFilter(img,12,100,100)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, redLower, redUpper)

	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	black_img = np.zeros((img.shape[1],img.shape[0]))
	cimg = cv2.bitwise_not(gray_img, black_img, mask)
	return cimg 

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
	# Save image in set directory, read RGB image 
	img = cv2.imread(img_filename) 
	img = img.astype('uint8')

	# Filtering
	mask = contour_filter(img)
	# cv2.imshow("mask",mask)

	# gray_img = houghcircles_filter(img)
	
	# # Finding Contours, Finding Hough Lines, intersecting contours are ignored
	# # Method 1
	# image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# lines = find_hough_lines(gray_img, img)
	# breakflag = 0 
	# i = 0
	# rej_mask = []
	# for contour in contours:
	# 	for cnt in contour:
	# 		for x,y in cnt:
	# 			for line in lines:
	# 				for x1,y1,x2,y2 in line:
	# 					if (x==x1 and y==y1) or (x==x2 and y==y2):
	# 						rej_mask.append(i)
	# 						breakflag = 1
	# 						break
	# 				if breakflag == 1:
	# 					break
	# 			if breakflag == 1:
	# 				break
	# 		if breakflag==1:
	# 			break
	# 	i += 1
	# 	breakflag = 0
	# print(rej_mask)
	# contours = np.delete(contours,rej_mask,0) # get rid of contours that are lines

	# i = 0
	# while i < len(contours):
	# 	img = cv2.drawContours(img, contours, i, (50*i,50*i,50*i), 3)
	# 	i += 1

	# # Find max contour area
	# i = 0
	# maxContour = 0
	# maxContourArea = 0
	# for contour in contours:
	# 	contourArea = cv2.contourArea(contour)
	# 	if contourArea > maxContourArea:
	# 		maxContourArea = contourArea
	# 		maxContour = i
	# 	i+=1

	# ((x, y), radius) = cv2.minEnclosingCircle(contours[maxContour])
	# cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
	# cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)


	## Method 2
	# circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,20,
 #                            param1=10,param2=100,minRadius=0,maxRadius=0)
	# circles = np.uint16(np.around(circles))
	# for i in circles[0,:]:
	#     # draw the outer circle
	#     cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
	#     # draw the center of the circle
	#     cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

	# Display Image
	cv2.imshow('image', img)  
	# Maintain output window utill user presses a key 
	cv2.waitKey(0)  
	# Destroying present windows on screen 
	cv2.destroyAllWindows()


if __name__ == '__main__':
	ball_tracker();

