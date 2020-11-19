#!/usr/bin/env python2

## BALL RECOGNITION CODE TAKEN FROM https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
## Written by Adrian Rosebrock, 2015

##### ROS #####
import rospy
from std_msgs.msg import Int32, Int32MultiArray

##### BALL RECOGNITION #####
# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import math
import argparse
import cv2
import imutils
import time

greenLower = (33, 94, 96)
greenUpper = (67, 255, 255)
redLower = (140, 154, 65)
redUpper = (207, 255, 195)
center_x = 250 # center of screen, x coord
center_y = 250 # center of screen, y coord

def init_webcam():
	'''Returns videostream'''
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video",
		help="path to the (optional) video file")
	ap.add_argument("-b", "--buffer", type=int, default=64,
		help="max buffer size")
	args = vars(ap.parse_args())

	# define the lower and upper boundaries of the "green"
	# ball in the HSV color space, then initialize the
	# list of tracked points
	pts = deque(maxlen=args["buffer"])
	# if a video path was not supplied, grab the reference
	# to the webcam
	if not args.get("video", False):
		vs = VideoStream(src=0).start()
	# otherwise, grab a reference to the video file
	else:
		vs = cv2.VideoCapture(args["video"])
	# allow the camera or video file to warm up
	time.sleep(2.0)

	return vs

def target_distance_calc(radius):
	if radius > 1:
		return 1/radius * 100
	else:
		return 100;

def target_rel_angle_calc(distance, center):
	return [degrees(atan(np.abs(center_x - center[0])/distance)), degrees(atan(np.abs(center_y - center[1])/distance))]

def ball_tracker():
	target_dist_pub = rospy.Publisher('target_distance', Int32)
	target_angle_pub = rospy.Publisher('target_relative_angle', Int32MultiArray)
	rospy.init_node('ball_tracker')
	r = rospy.Rate(10)
	vs = init_webcam()
	while not rospy.is_shutdown():
		# grab the current frame
		frame = vs.read()
		# handle the frame from VideoCapture or VideoStream
		frame = frame[1] if args.get("video", False) else frame
		# if we are viewing a video and we did not grab a frame,
		# then we have reached the end of the video
		if frame is None:
			break
		# resize the frame, blur it, and convert it to the HSV
		# color space
		frame = imutils.resize(frame, width=600)
		blurred = cv2.GaussianBlur(frame, (11, 11), 0)
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
		# construct a mask for the color "green", then perform
		# a series of dilations and erosions to remove any small
		# blobs left in the mask
		mask = cv2.inRange(hsv, redLower, redUpper)
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)

		# find contours in the mask and initialize the current
		# (x, y) center of the ball
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		center = None 
		radius = 0
		# only proceed if at least one contour was found
		if len(cnts) > 0:
			# find the largest contour in the mask, then use
			# it to compute the minimum enclosing circle and
			# centroid
			c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
			# only proceed if the radius meets a minimum size
			if radius > 10:
				# draw the circle and centroid on the frame,
				# then update the list of tracked points
				cv2.circle(frame, (int(x), int(y)), int(radius),
					(0, 255, 255), 2)
				cv2.circle(frame, center, 5, (0, 0, 255), -1)
		# update the points queue
		pts.appendleft(center)

		# Publish Results on ROS Node
		target_distance = target_distance_calc(radius)
		target_relative_angle = target_rel_angle_calc(target_distance, center)
 
		target_dist_pub.publish(target_distance)
		target_angle_pub.publish = target_relative_angle
		# loop over the set of tracked points
		for i in range(1, len(pts)):
			# if either of the tracked points are None, ignore
			# them
			if pts[i - 1] is None or pts[i] is None:
				continue
			# otherwise, compute the thickness of the line and
			# draw the connecting lines
			thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
			cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

		# record to parse min and max contour of ball
		if (startflag == 1):
		    minMaxRadius.append(radius);

		# show the frame to our screen
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		# if the 'q' key is pressed, stop the loop
		if key == ord("q"):
			break

		rate.sleep()

	# if we are not using a video file, stop the camera video stream
	if not args.get("video", False): 
		vs.stop()
	# otherwise, release the camera
	else:
		vs.release()
	# close all windows
	cv2.destroyAllWindows()

if __name__ == '__main__':
	try:
		ball_tracker()
	except rospy.ROSInterruptException:
		pass

