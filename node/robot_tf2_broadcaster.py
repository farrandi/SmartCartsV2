#!/usr/bin/env python

""" 
Modified from turtle_tf2 tutorial for use with fizzym/smartcarts package
http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20broadcaster%20%28Python%29
Simon Zheng
Feb 22 2020

This code converts the gazebo model state into a tf frame
"""

import rospy

import tf_conversions
import tf2_ros
import geometry_msgs.msg
import nav_msgs.msg
from gazebo_msgs.msg import ModelStates

robot_name = rospy.get_namespace().replace('/','') #enables node to programatically get robot name from launchfile

# Broadcasts the robot pose on the tfBuffer
def handle_robot_pose(msg, robot_name):
    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()

    t.header.stamp = rospy.Time.now() + rospy.Time(secs=0, nsecs=1)
    t.header.frame_id = "odom"
    t.child_frame_id = robot_name
    t.transform.translation.x = msg.position.x
    t.transform.translation.y = msg.position.y
    t.transform.translation.z = 0.0
    #q = tf_conversions.transformations.quaternion_from_euler(0, 0, msg.orientation) #gazebo pose is already a quaternion
    q = msg.orientation
    t.transform.rotation.x = q.x
    t.transform.rotation.y = q.y
    t.transform.rotation.z = q.z
    t.transform.rotation.w = q.w

    # TODO: This is causing warning outputs
    br.sendTransform(t)

# Finds pose corresponding to robot_name from ModelStates
def extract_pose_callback(msg):
    try:
        handle_robot_pose(msg.pose[msg.name.index(robot_name)], robot_name)
    except:
        pass

# Links the EKF robot pose with the gazebo simulation worlds
def ekf2world_callback(msg):
    try:
        handle_follower_pose(msg.pose, 'follower_ekf')
    except:
        pass


if __name__ == '__main__':
    rospy.init_node('tf2_robot_broadcaster')
    rospy.Subscriber('/gazebo/model_states', ModelStates, extract_pose_callback, queue_size=1)

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
