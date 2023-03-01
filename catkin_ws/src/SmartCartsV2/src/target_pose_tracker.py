#!/usr/bin/env python
'''
Published:
    /target_position
    /target_distance
Subscribed:
    /camera/color/image_raw
    /camera/aligned_depth_to_color/image_raw

Written by: Iain Copland, Feb 2023
Modified by: Farrandi Hernando, Feb 2023
'''


import rospy
from std_msgs.msg import Int32MultiArray, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer

import numpy as np
import cv2
import sys
import time

####### GLOBAL VARIABLES ########
LOWER_RED1 = np.array([0,200,50])
UPPER_RED1 = np.array([10,255,255])

LOWER_RED2 = np.array([170,200,50])
UPPER_RED2 = np.array([180,255,255])

LOWER_GREEN = np.array([40, 40, 40])
UPPER_GREEN = np.array([70,255,255])



class ballTracker:

    def __init__(self):
        time.sleep(2.0) # Wait for camera in gazebo to start up
        self.bridge = CvBridge()
        self.namespace = rospy.get_namespace()

        self.camera_param = 'depth'

        # Initialize Subscribers
        self.color_sub = rospy.Subscriber('/camera/color/image_raw'.format(self.namespace), Image, self.color_callback, queue_size = 2)
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback, queue_size = 2)
        # Initialize Publishers
        self.position_pub = rospy.Publisher('target_position', Int32MultiArray, queue_size = 2)
        self.distance_pub = rospy.Publisher('target_distance', Float32, queue_size = 2)
        # self.target_pose_pub = rospy.Publisher("/target_pose", PoseStamped, queue_size = 1)


        # Do we need the lines below?

        # self.ats = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], \
        #                                         queue_size=10, slop=0.1, allow_headerless=True)
        # self.ats.registerCallback(self.color_callback)
        # self.ats.registerCallback(self.depth_callback)

        self.depth_image_raw = None
        self.color_image_raw = None
        self.circle_list = None

    '''Takes in an img given by cv2.imread and applies the following filters, returns the filtered image'''
    def contour_filter(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #mask1 = cv2.inRange(image, LOWER_RED1, UPPER_RED1)
        #mask2 = cv2.inRange(image, LOWER_RED2, UPPER_RED2)
        #mask = mask1 + mask2
        mask = cv2.inRange(image, LOWER_GREEN, UPPER_GREEN)

        return mask

    '''
    Takes in [mask] a binary image (1/0 for each pixel) and [img] the 2D color image
    Finds the contours and appends the x,y,radius,timestamp lists from results of each image
    Returns [circle_list] a list for information about enclosing circle (x,y,radius)
    '''
    def parse_color_image(self, mask):
        # Finding Contours
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
    

    '''Calculate leader pose from postion/distance + modelstate pose, pos=List, dist,radius-Float32, returns Pose of target
    Retreived from target_pose_tracking.py (sim)
    '''
    # Unsure of self.ball_radius and self.camera_width
    # Also uses Pose, 
    def calc_leader_pose(self, pos, dist, radius):
        if self.camera_width is not None and \
                            pos is not [-1,-1] and \
                            dist != -1.0 and \
                            radius != -1.0 :
            # Below is calculation from the ball tracking inputs to "physical" dimensions
            # +x and +y designations based on gazebo sim defaults
            pixel_scale = radius/self.ball_radius # Pixels/unit Length
            y_dist_from_center = (self.camera_width/2 - pos[0]) / pixel_scale # Unit length
            if abs(y_dist_from_center/dist) > 1 :
                print("MATH ERROR: y_dist_from_center={}, dist_data={}".format(y_dist_from_center, dist))
                print("time = {}".format(rospy.Time.now()))
            else:
                theta = math.asin(y_dist_from_center / dist) # Radians
                x_dist_from_camera = dist * math.cos(theta) # Unit length, rel to Follower ref frame
                y_dist_from_camera = y_dist_from_center # Unit length, rel to Follower ref frame
                return Pose(position = Point(x = x_dist_from_camera, y = y_dist_from_camera, z = 0), orientation = Quaternion(w=1.0 ))
        return None


    ########## Callback Functions ###############

    '''Parse color frames from camera'''
    def color_callback(self, image):
        try:
            # time_start = rospy.Time.now()
            self.color_image_raw = self.bridge.imgmsg_to_cv2(image, 'bgr8')
        except CvBridgeError as e:
            print(e)

        # Find ball in color frame
        color_image = np.asanyarray(self.color_image_raw)
        mask = self.contour_filter(color_image)
        self.circle_list = self.parse_color_image(mask, color_image)

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

            # Condition from target_pose_tracking.py (Sim)
            # Still uses target pose...
            if distance < 0.01 or distance > 10 or np.isnan(distance):
                self.distance_pub.publish(float(-1.0))
            else:
                self.distance_pub.publish(distance)
                target_pose = self.calc_leader_pose([x,y], distance, radius)
                if self.twist.angular.z > 0:
                    #target_pose == target_position ??
                    self.target_pose_pub.publish(self.no_ball_poseStamped())
                else:
                    self.target_pose_pub.publish(PoseStamped(header = Header(stamp = rospy.Time.now(), frame_id = "follower"), \
                                                            pose = target_pose))
        return


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