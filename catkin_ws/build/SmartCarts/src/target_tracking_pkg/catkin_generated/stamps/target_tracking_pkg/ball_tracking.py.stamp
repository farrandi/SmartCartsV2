#!/usr/bin/env python

## BALL RECOGNITION CODE TAKEN FROM https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
## Written by Adrian Rosebrock, 2015

##### ROS #####
import rospy
from std_msgs.msg import Int32, Int32MultiArray, UInt16MultiArray
from sensor_msgs.msg import Image
##### BALL RECOGNITION #####
# import the necessary packages
from collections import deque
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
import time

#import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
#sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

##### GLOBAL VARIABLES #####
redLower = (130, 130, 0)
redUpper = (255, 255, 255)
x_list = []
y_list = []
radius_list = []
distance_list = []
timestamp_list = []
# RGBD data 
depth_frame_data = [] # uint8
color_frame_data = [] # uint8
# args = {}
# pts = deque()

## Takes in an img given by cv2.imread and applies the following filters
## Returns the filtered image
def contour_filter(img):
    blurred = cv2.bilateralFilter(img,12,125,125)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, redLower, redUpper)
    return mask

## Takes in [mask] a binary image (1/0 for each pixel) and [img] the 2D color image
##  Finds the contours and appends the x,y,radius,timestamp lists from results of each image
##  MODIFIES [img] by drawing circle and centroid on image
## Returns [circle] a list for information about enclosing circle (x,y,radius)
def parse_color_image(mask, img):
    ## Finding Contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find max contour area
    i = 0
    maxContour = 0
    maxContourArea = 0
    for contour in contours:
        contourArea = cv2.contourArea(contour)
        if contourArea > maxContourArea:
            maxContourArea = contourArea
            maxContour = i
        i += 1
    # Find coordinates + radius of min enclosing circle of blob in mask
    ((x, y), radius) = cv2.minEnclosingCircle(contours[maxContour])
    # Drawing on image the circle & corresponding centroid calculated above if circle is detected
    # Also populating x,y,radius lists
    if radius > 0:
        img = cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        img = cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        x_list.append(x)
        y_list.append(y)
        radius_list.append(radius)
        # Define list to return
        circle = [x,y,radius]
        return circle
    else:
        return None

## Callback functions for depth frame
# From camera stream
def camera_depth_callback(image):
	depth_frame_data = image.data
# From .bag file
def bag_depth_callback(image):
	depth_frame_data = image.data

## Callback functions for color frame
# From camera stream
def camera_color_callback(image):
	color_frame_data = image.data
# From .bag file
def bag_color_callback(image):
	color_frame_data = image.data

## Main ball_tracking node.
##  Opens stream/video and executes the logic to parse each frame
## Returns [circle] tuple of (x,y,radius,distance(from depth))
def ball_tracker():
	### ROS INTERFACE ###
	# Subscribe to the following topics:
	# Video Stream:
		##<------TODO------>##
	# 	1. Realsense Camera Node Package > Color Frame Topic 
	# 	2. Realsense Camera Node Package > Depth Frame Topic 
	# Playing from .bag file:
	#	1. rosbag > Depth image Topic
	#	2. rosbag > Color image Topic
	# Publish the following topics:
	# 	1. Tuple (x in meters, y in meters, depth in meters)
	# 	2. Time of video frame

    # Init ball_tracker ROS node with annonymous node name 
    rospy.init_node('ball_tracker', anonymous= True)
    rospy.myargv(argv=sys.argv) 
 	# Subscribe to different topics based on parameters passed in. 
 	# Default is realsense camera node (input_type = 0)
    if (sys.argv[0] == 0): 
        rospy.Subscriber('/camera/depth/image_raw', Image, camera_color_callback) # THIS IS FROM GLYNN'S CODE AND NOT TESTED
        rospy.Subscriber('/camera/color/image_raw', Image, camera_color_callback)
    else:
        rospy.Subscriber("/device_0/sensor_0/Depth_0/image/data", Image, bag_depth_callback)
        rospy.Subscriber("/device_0/sensor_1/Color_0/image/data", Image, bag_color_callback)
 	# Publish the parsed x,y,radius,distance (uint8[]) array
    circle_pub = rospy.Publisher('target_distance', UInt16MultiArray, queue_size = 10)
    r = rospy.Rate(10)
    time.sleep(1.0)
    frame_num = 0
    flag = 0
    while not rospy.is_shutdown():
        if ((depth_frame_data is not [] or depth_frame_data is not None) and frame_num < 10):
            ##### BELOW IS ALL COMMENTED OUT BECAUSE IM NOT SURE HOW TO PARSE THIS NEW "IMAGE" MESSAGE TYPE #####
            depth_image = np.asanyarray(depth_frame_data)
            color_image = np.asanyarray(color_frame_data)
            if flag == 0:
                flag = 1
                first_frame = depth_frame_data
            # # parsing color image
            # mask = contour_filter(color_image)
            # # gets the x,y,radius reading from image
            # circle_list = parse_color_image(mask, color_image)

            # if circle_list is None:
            #     break

            # # get the depth reading from the depth frame at circle centroid (units: meters) and save into tuple
            # distance = depth_frame_data.get_distance(int(circle_list[0]), int(circle_list[1]))
            # distance_list.append(distance)
            # circle_list.append(distance)

            # # get the timestamp of the current color/depth frame
            # timestamp_list.append(color_frame.get_timestamp())

            circle_list = [frame_num, frame_num] #pseudo code, just stepping through frames
            frame_num += 1

            ### Publish Results on ROS Node ###
            circle_pub.publish(data=circle_list)

    np.savetxt("/home/miyesven/Documents/5_first_depth.txt", first_frame)
    r.sleep()

    
if __name__ == '__main__':
	try:
		ball_tracker()
	except rospy.ROSInterruptException:
		pass

