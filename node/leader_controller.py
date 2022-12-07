#!/usr/bin/env python

'''
Simple node to run dead-reckoning target tracking (proportional speed / angular)
Subscribed:
    /gazebo/model_states
    /move_base_simple/goal
Published:
    /<name>/waypoints_viz
    /<name>/cmd_vel
Written by Simon Zheng, Feb 2021
'''

import csv
import math
import numpy as np
import rospy
import random
import sys, time

from std_msgs.msg import Int32MultiArray, Float32, Int32
from geometry_msgs.msg import Twist, PoseStamped, Pose, PoseArray
from tf.transformations import euler_from_quaternion
from gazebo_msgs.msg import ModelStates
from visualization_msgs.msg import Marker, MarkerArray

class VehicleController():
    def __init__(self, use_waypointFile):
        time.sleep(1.0)
        self.name = rospy.get_namespace().replace('/','')
        # Publishers and Subscribers
        self.waypoint_pub = rospy.Publisher('/{}/waypoints_viz'.format(self.name), MarkerArray, queue_size = 1)
        self.vel_pub = rospy.Publisher('/{}/cmd_vel'.format(self.name), Twist, queue_size=1, latch=False)
        self.pose_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.pose_feedback_callback, queue_size=1)
        self.clicked_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.add_waypoint_callback, queue_size=1)
        self.leader_help_sub = rospy.Subscriber('/follower/leader_help', Int32, self.leader_help_callback, queue_size=1)
        # Class Variables
        self.pose = None
        self.heading_deadband = 0.10
        self.position_deadband = 0.015
        self.pose_goal = None
        self.pose_goal_buffer = []
        self.pose_goal_index = 0
        self.pose_goal_max = 30
        self.pose_goal_total = 0 #tracks total number of pose_goals visited (limited by UNSIGNED INT32 MAX)
        self.leader_help_state = 0 # 0 for nominal, 1 for need help

        ## Some variables for visualization
        self.marker_array = MarkerArray()
        ## if preset waypoints specified, import from file
        if(use_waypointFile):
            self.waypoints_file = rospy.get_param('~waypoints_file')

            with open(self.waypoints_file) as waypoints_csv:
                waypoints_reader = csv.reader(waypoints_csv, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
                for row in waypoints_reader:
                    new_pose = Pose()
                    # print(new_pose)
                    new_pose.position.x = row[0]
                    new_pose.position.y = row[1]
                    self.pose_goal_buffer.append(new_pose)
                    self.add_marker(new_pose)
        self.pose_goal = self.pose_goal_buffer[0]
        # Values
        self.max_angular_vel = 3
        self.max_linear_vel = 0.5
        self.at_rest = False
        self.last_reached_dest_time = rospy.Time.now()


    # Adds marker to marker array, define style here
    def add_marker(self, pose):
        marker = Marker();
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.5
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose = pose;
        marker.pose.orientation.w = 1.0 #pointing straight up
        self.marker_array.markers.append(marker)
        # renumber marker IDs to cap MAX
        self.renumber_markers()
        self.waypoint_pub.publish(self.marker_array)

    def renumber_markers(self):
        id = 0
        num_markers = len(self.marker_array.markers)
        n = float(num_markers)*3/4
        for m in self.marker_array.markers:
            m.id = id
            m.color.a = ((id/num_markers) * n + (num_markers-n))/n
            id += 1

    '''
    Adds the x-y position presented in the msg as a waypoint (pose + visual marker)
    Initializes pose_goal if none presents
    '''
    def add_waypoint_callback(self, msg):
        if(msg is not None):
            new_pose = Pose()
            new_pose.position.x = msg.pose.position.x
            new_pose.position.y = msg.pose.position.y
            self.pose_goal_buffer.append(new_pose)
            self.pose_goal_total += 1

            self.add_marker(new_pose)

            if(self.pose_goal is None):
                self.pose_goal = self.pose_goal_buffer[0]


    '''Callback function for /follower/leader_help. Populate self.leader_help_state state.'''
    def leader_help_callback(self, state):
        self.leader_help_state = state.data

    def pose_feedback_callback(self, msg):
        cmd_vel = Twist()
        if self.name in msg.name and self.leader_help_state == 0:
            index = msg.name.index(self.name)
            self.pose = msg.pose[index]

            if self.pose is not None and self.pose_goal is not None:
                current_rpy = euler_from_quaternion((self.pose.orientation.x,self.pose.orientation.y,self.pose.orientation.z,self.pose.orientation.w))
                dest_rpy = euler_from_quaternion((self.pose_goal.orientation.x, self.pose_goal.orientation.y, self.pose_goal.orientation.z, self.pose_goal.orientation.w))
                dx = self.pose_goal.position.x - self.pose.position.x
                dy = self.pose_goal.position.y - self.pose.position.y
                heading = math.atan2(dy, dx)
                angle = current_rpy[2]-heading
                while angle >= math.pi:
                    angle = angle - 2*math.pi
                while angle <= -math.pi:
                    angle = angle + 2*math.pi
                dist = math.hypot(dy,dx)

                # Rotate to the direct goal heading
                if abs(angle) > self.heading_deadband and dist > self.position_deadband and not self.at_rest:
                    # print("Aligning with heading")
                    if angle > self.heading_deadband:
                        cmd_vel.angular.z = -self.max_angular_vel
                    elif angle < -self.heading_deadband:
                        cmd_vel.angular.z = self.max_angular_vel

                # Drive forwards to goal position
                elif dist > self.position_deadband:
                    # print("Moving to destination position",dist)
                    cmd_vel.linear.x = self.max_linear_vel * random.uniform(0.4, 0.8)
                else:
                    # Reached deadband about goal position, remove waypoint from history
                    if(len(self.pose_goal_buffer) > 1):
                        self.pose_goal_buffer.pop(0)
                        self.pose_goal = self.pose_goal_buffer[0]  # Set the new goal
                    else:
                        self.pose_goal = None

        self.vel_pub.publish(cmd_vel)
        self.waypoint_pub.publish(self.marker_array)

if __name__ == '__main__':
    node_args = rospy.myargv(argv=sys.argv)  # takes in arguments
    rospy.init_node('leader_controller')

    if len(node_args) == 0:
        cw = VehicleController(False)
    else:
        cw = VehicleController(node_args[0])

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
