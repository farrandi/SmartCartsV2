#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32MultiArray, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import cv2
import sys

LOWER_RED1 = np.array([0,200,50])
UPPER_RED1 = np.array([10,255,255])

LOWER_RED2 = np.array([170,200,50])
UPPER_RED2 = np.array([180,255,255])

LOWER_GREEN = np.array([40, 40, 40])
UPPER_GREEN = np.array([70,255,255])

def contour_filter(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #mask1 = cv2.inRange(image, LOWER_RED1, UPPER_RED1)
    #mask2 = cv2.inRange(image, LOWER_RED2, UPPER_RED2)
    #mask = mask1 + mask2
    mask = cv2.inRange(image, LOWER_GREEN, UPPER_GREEN)

    return mask

def parse_color_image(mask, img):
    image_, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    maxContour = 0
    maxContourArea = 0
    for contour in contours:
        contourArea = cv2.contourArea(contour)
        if contourArea > maxContourArea:
            maxContourArea = contourArea
            maxContour = i
        i += 1
    if len(contours) >= 1:
        ((x, y), radius) = cv2.minEnclosingCircle(contours[maxContour])
        # Drawing on image the circle & corresponding centroid calculated above if circle is detected
        # Also populating x,y,radius lists
        if radius > 10:
            circle = [int(x),int(y),int(radius)]
            return circle
    return None

class ballTracker:

    def __init__(self):
        self.bridge = CvBridge()
        self.namespace = rospy.get_namespace()

        self.camera_param = 'depth'

        self.color_sub = rospy.Subscriber('/camera/color/image_raw'.format(self.namespace), Image, self.color_callback, queue_size = 2)
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback, queue_size = 2)

        self.position_pub = rospy.Publisher('target_position', Int32MultiArray, queue_size = 2)
        self.distance_pub = rospy.Publisher('target_distance', Float32, queue_size = 2)

        self.depth_image_raw = None
        self.color_image_raw = None
        self.circle_list = None

    def color_callback(self, image):
        try:
            self.color_image_raw = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        except CvBridgeError as e:
            print(e)

        color_image = np.asanyarray(self.color_image_raw)
        mask = contour_filter(color_image)
        self.circle_list = parse_color_image(mask, color_image)

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
            self.depth_image_raw = self.bridge.imgmsg_to_cv2(image, "passthrough")
        except CvBridgeError as e:
            print(e)

        depth_array = np.array(self.depth_image_raw, dtype=np.float32)

        if self.circle_list is None:
            self.distance_pub.publish(float(-1.0))
        else:
            x = self.circle_list[0]
            y = self.circle_list[1]
            distance = depth_array[y][x]
            self.distance_pub.publish(float(distance))

def main(args):
    rospy.init_node('ball_tracker', anonymous=True)
    bt = ballTracker()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("ball_tracker shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)