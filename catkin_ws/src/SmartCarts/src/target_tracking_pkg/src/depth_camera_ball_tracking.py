#!/usr/bin/env python

## BALL RECOGNITION CODE TAKEN FROM https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

##### ROS #####
import rospy
from std_msgs.msg import Int32, Int32MultiArray, UInt8MultiArray, Float32MultiArray, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

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


##### GLOBAL VARIABLES #####
# IN RGB, in simulation for RED ball
# redLower = (0, 0, 0)
# redUpper = (20, 60, 255)


#RED
# IN HSV, in reality ## USE THIS FOR PHYSICAL BALL TRACKING
#following is for RED BALL (svena's calibration):
# colourLower = (130, 130, 0)
# colourUpper = (255, 255, 255)

#following is for RED BALL in DAY lighting:
# colourLower = (0, 159, 82)
# colourUpper = (183, 249, 220)

#following is for RED BALL in NIGHT lighting:
colourLower = (0, 129, 127)
colourUpper = (181, 255, 255)


#YELLOW
#following is for YELLOW BALL in DAY lighting:
# colourLower = (14, 180, 105)
# colourUpper = (30, 255, 255)

#following is for YELLOW BALL in NIGHT lighting:
# colourLower = (12, 110, 123)
# colourUpper = (26, 255, 255)

# #following is for YELLOW BALL in OFFICE lighting:
# colourLower = (12, 107, 96)
# colourUpper = (25, 255, 255)


pts = deque()

# Init cv_bridge
bridge = CvBridge()


## Takes in an img given by cv2.imread and applies the following filters
## Returns the filtered image
def contour_filter(img):
    # image = cv2.bilateralFilter(img,12,125,125)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) ## USE THIS FOR PHYSICAL BALL TRACKING
    mask = cv2.inRange(image, colourLower, colourUpper)
    return mask

## Takes in [mask] a binary image (1/0 for each pixel) and [img] the 2D color image
##  Finds the contours and appends the x,y,radius,timestamp lists from results of each image
##  MODIFIES [img] by drawing circle and centroid on image
## Returns [circle] a list for information about enclosing circle (x,y,radius)
def parse_color_image(mask, img):
    ## Finding Contours
    image_, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    if len(contours) >= 1:
        ((x, y), radius) = cv2.minEnclosingCircle(contours[maxContour])
        # Drawing on image the circle & corresponding centroid calculated above if circle is detected
        # Also populating x,y,radius lists
        if radius > 10:
            circle = [int(x),int(y),int(radius)]
            return circle
    return None

class ball_tracker:

    def __init__(self):
        time.sleep(2.0) # Wait for camera in gazebo to start up
        self.bridge = CvBridge()
        self.namespace = rospy.get_namespace()
        
        # Fetch topic_name from the ~private namespace
        #self.camera_param = rospy.get_param('/camera')
        self.camera_param = 'depth'

        # Initialize Publishers and Subscribers
        if self.camera_param == 'depth':
            ## EDIT THIS FOR PHYSICAL BALL TRACKING
            self.color_sub = rospy.Subscriber('/camera/color/image_raw'.format(self.namespace), Image, self.color_callback, queue_size = 2)
            #self.depth_sub = rospy.Subscriber('/camera/depth/image_rect_raw'.format(self.namespace), Image, self.depth_callback, queue_size = 2)
            self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw'.format(self.namespace), Image, self.depth_callback, queue_size = 2)
        else:
            self.image_sub = rospy.Subscriber('/camera/color/image_raw'.format(self.namespace), Image, self.color_callback)
        
        self.distance_pub = rospy.Publisher('target_distance', Float32, queue_size = 2)
        # This publishes the coordinates of the ball centroid in image frame.
        self.position_pub = rospy.Publisher('target_position', Int32MultiArray, queue_size = 2)
        # Initialize Class Variables
        self.depth_image_raw = None
        self.color_image_raw = None
        self.circle_list = None

    def color_callback(self, image):
        try:
            self.color_image_raw = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        except CvBridgeError as e:
            print(e)
        # print("color image header = {}, type = {}".format(image.header.stamp, type(image.header.stamp)))
        color_image = np.asanyarray(self.color_image_raw)
        mask = contour_filter(color_image)
        self.circle_list = parse_color_image(mask, color_image)
        # print("self.circle_list in COLOR callback = {}".format(self.circle_list))
        # print("color image header = {}, type = {}".format(image.header.stamp, type(image.header.stamp)))
        if self.circle_list is None:
            self.position_pub.publish(Int32MultiArray(data=[-1,-1]))
        else:
            x = self.circle_list[0]
            y = self.circle_list[1]
            radius = self.circle_list[2]
            color_image = cv2.circle(color_image, (x,y), radius, (0, 255, 255), 2)
            position_data = Int32MultiArray(data=[x,y])
            self.position_pub.publish(position_data)
        cv2.imshow("RGB", color_image)
        cv2.waitKey(3)
        return

    def depth_callback(self, image):
        try:
            # Converting an image message pointer to an OpenCV message only requires a call to the function
            # Takes in image message and encoding of destination Opencv image
            ## CHECK FOR PHYSICAL BALL TRACKING
            self.depth_image_raw = self.bridge.imgmsg_to_cv2(image, "passthrough") # 'passthrough' to convert to 32FP, each pixel rep depth in mm
            # print("type is {} and shape is {}".format(type(self.depth_image_raw), self.depth_image_raw.shape))
        except CvBridgeError as e:
            # Catch conversion errors
            print(e)
        # print("self.circle_list in DEPTH callback = {}".format(self.circle_list))
        # print("depth image header = {}, type = {}".format(image.header.stamp, type(image.header.stamp)))
        if self.circle_list is None:
            self.distance_pub.publish(float(-1.0))
        else:
            x = self.circle_list[0]
            y = self.circle_list[1]
            
            distance = self.depth_image_raw[y][x]
            self.distance_pub.publish(float(distance))


            # if(x >= 480):
            #     self.distance_pub.publish(float(-1.0))
            # else:
            #     # depth_image = cv2.circle(depth_image, (x,y), 5, (0, 255, 255), 2)
            #     distance = self.depth_image_raw[x][y]
            #     self.distance_pub.publish(float(distance))
        
        # cv2.imshow("Depth", self.depth_image_raw) ## NOT SURE WHY IT IS STUCK HERE WHEN THIS CODE IS ACTIVE
        # cv2.waitKey(3)
        return

def main(args):
    rospy.init_node('ball_tracker', anonymous=True)
    bt = ball_tracker()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("ball_tracker shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)