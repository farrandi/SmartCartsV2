#!/usr/bin/env python

'''
Ball recognition algorithm taken from : https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
Uses cvbridge to parse the depth and color images from the simulated Gazebo Kinect RGBD Camera
(Can also be easily modified to use with realsense camera on physical robot)
Takes in robot ModelState pose information and ball tracking info at the same time
Published:
    /target_pose
Subscribed:
    /gazebo/model_states
    /{}/camera1/color/camera_info
    /{}/camera1/color/image_raw
    /{}/camera1/depth/image_raw
Modified by: Svena Yu, Feb 2021
'''
##### ROS #####
import rospy
from std_msgs.msg import Int32, Int32MultiArray, UInt8MultiArray, Float32MultiArray, Float32, Header
from geometry_msgs.msg import Twist, PoseStamped, Pose, Quaternion, Point, Transform, TransformStamped
from gazebo_msgs.msg import ModelStates
from smartcarts.msg import Float32Stamped, Float32MultiArrayStamped, BallPoseStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import *

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
# IN RGB, in simulation
redLower = (0, 0, 0)
redUpper = (20, 60, 255)
# IN HSV, in reality ## USE THIS FOR PHYSICAL BALL TRACKING
# redLower = (130, 130, 0)
# redUpper = (255, 255, 255)

# Init cv_bridge
bridge = CvBridge()

class BallTracker:

    def __init__(self):
        time.sleep(2.0) # Wait for camera in gazebo to start up
        self.bridge = CvBridge()
        self.robot_name = rospy.get_namespace().replace('/','')
        # Fetch topic_name from the ~private namespace
        self.camera_param = rospy.get_param('/camera_type')
        self.ball_radius = rospy.get_param('/ball_radius')
        self.debug = rospy.get_param('/debug_mode') # debug_mode = 1 means IN DEBUG MODE
        self.camera_width = None

        # Initialize Publishers and Subscribers
        self.camera_info_sub = rospy.Subscriber('/{}/camera1/color/camera_info'.format(self.robot_name), \
                                        CameraInfo, self.camera_info_callback, queue_size = 1)
        self.leader_help_sub = rospy.Subscriber('/{}/leader_help'.format(self.robot_name), Int32, self.leader_help_callback)
        self.cmd_vel_sub = rospy.Subscriber('/{}/cmd_vel'.format(self.robot_name), Twist, self.cmd_vel_callback, queue_size = 1)

        self.color_sub = Subscriber('/{}/camera1/color/image_raw'.format(self.robot_name), Image)
        self.depth_sub = Subscriber('/{}/camera1/depth/image_raw'.format(self.robot_name), Image)
        self.states_sub = Subscriber('/gazebo/model_states', ModelStates)
        self.ats = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub, self.states_sub], \
                                                queue_size=10, slop=0.1, allow_headerless=True)
        self.ats.registerCallback(self.target_pose_calc_callback)
        # Publish ball pose as calculated 
        self.target_pose_pub = rospy.Publisher("/target_pose", PoseStamped, queue_size = 1)
        self.ball_radius_pub = rospy.Publisher("/ball_radius", Float32, queue_size = 1) # for waypoint_listener
        self.ball_distance_pub = rospy.Publisher('/ball_distance', Float32, queue_size = 1)

        # Initialize Class Variables
        self.depth_image_raw = None
        self.color_image_raw = None
        self.circle_list = None
        self.twist = Twist()
        self.leader_help_state = 0 # = nominal, 1 = help!
        self.help_flag = 0;
        # tf2
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        print("Started ball tracking")

    '''Callback for leader help state, updates self.leader_help_state'''
    def leader_help_callback(self, state):
        self.leader_help_state = state.data

    '''Callback for robotname/cmd_vel updates self.twist'''
    def cmd_vel_callback(self, twist):
        self.twist = twist

    '''Takes in an img given by cv2.imread and applies the following filters, returns the filtered image'''
    def contour_filter(self, img):
        image = cv2.bilateralFilter(img,12,125,125)
        # using RGB here # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) ## USE THIS FOR PHYSICAL BALL TRACKING
        mask = cv2.inRange(image, redLower, redUpper)
        return mask

    '''
    Takes in [mask] a binary image (1/0 for each pixel) and [img] the 2D color image
    Finds the contours and appends the x,y,radius,timestamp lists from results of each image
    Returns [circle_list] a list for information about enclosing circle (x,y,radius)
    '''
    def parse_color_image(self, mask, img):
        # Finding Contours
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
        if len(contours) >= 1:
            ((x, y), radius) = cv2.minEnclosingCircle(contours[maxContour])
            # Drawing on image the circle & corresponding centroid calculated above if circle is detected
            # Also populating x,y,radius lists
            if radius > 5:
                circle = [int(x),int(y),radius]
                return circle
        return None

    # Get Camera Width
    def camera_info_callback(self, info):
        self.camera_width = info.width
        return

    # Calculate leader pose from postion/distance + modelstate pose, pos=List, dist,radius-Float32, returns Pose of target
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

    # Returns a 'negative' PoseStamped
    def no_ball_poseStamped(self):
        return PoseStamped( header = Header(stamp = rospy.Time.now(), frame_id = "odom"), \
            pose = Pose(position = Point(x = -1, y = -1, z = -1), orientation = Quaternion(w=1)))

    ''' Listener function and publishing to /target_waypoint topic '''
    def publish_abs_target_pose(self, time_past):
        try:
            abs_target_pose = self.tf_buffer.lookup_transform_full(target_frame='target', target_time=rospy.Time.now(), \
                source_frame='follower', source_time=time_past, fixed_frame='world', timeout=rospy.Duration(1.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return
        self.target_pose_pub.publish(abs_target_pose)

    ''' Parse color and depth frames from Kinect camera with ModelState of robot, publishes the newest waypoint to '/target_pose_pub'''
    def target_pose_calc_callback(self, color_frame, depth_frame, model_states):
        if self.help_flag == 0 and self.leader_help_state == 1:
            self.help_flag == 1
            rospy.sleep(rospy.Duration(0.5)) # wait for robot to stabilize
        if self.help_flag == 1 and self.leader_help_state == 0:
            self.help_flag = 0
        # Target Tracking Algorithm
        try:
            time_start = rospy.Time.now()
            self.color_image_raw = self.bridge.imgmsg_to_cv2(color_frame, 'bgr8')
            self.depth_image_raw = self.bridge.imgmsg_to_cv2(depth_frame, "passthrough") # 'passthrough' to convert to 32FP, each pixel rep depth in mm
        except CvBridgeError as e:
            print(e)
        # Find ball in color frame
        color_image = np.asanyarray(self.color_image_raw)
        mask = self.contour_filter(color_image)
        self.circle_list = self.parse_color_image(mask, color_image)
        if self.circle_list is None:
            self.target_pose_pub.publish(self.no_ball_poseStamped())
            self.ball_distance_pub.publish(-1.0)
        else:
            x = self.circle_list[0]
            y = self.circle_list[1]
            radius = self.circle_list[2]
            position_data = [x,y]
            distance = self.depth_image_raw[int(y)][int(x)] # ROWS ARE Y and COLS ARE X
            self.ball_radius_pub.publish(radius)
            if distance < 0.01 or distance > 10 or np.isnan(distance):
                self.target_pose_pub.publish(self.no_ball_poseStamped())
                self.ball_distance_pub.publish(-1.0)
            else:
                self.ball_distance_pub.publish(distance)
                target_pose = self.calc_leader_pose([x,y], distance, radius)
                if self.twist.angular.z > 0:
                    self.target_pose_pub.publish(self.no_ball_poseStamped())
                else:
                    self.target_pose_pub.publish(PoseStamped(header = Header(stamp = rospy.Time.now(), frame_id = "follower"), \
                                                            pose = target_pose))
            time_stop = rospy.Time.now()
        return

def main(args):
    rospy.init_node('ball_tracker', anonymous=True)
    bt = BallTracker()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("ball_tracker shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
