#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Imu


delay_time = 0.05	#number of seconds to delay the program

imu_msg = Imu()	#Twist message to prepare and publish to the cmd_vel topic to move the robot


#function called whenever the "position_subscriber" subscriber below subscribes a message from the "target_position" topic (published to by the "ball_tracking" node)
#this message is a python list (i.e.: array) of two 32 bit integers (Int32) representing the x and y positions in the camera's field of view & is stored in the variable "pData" 
def imuProcess(imuData):
	imu_msg.angular_velocity.x = imuData.angular_velocity.x
	imu_msg.angular_velocity.y = imuData.angular_velocity.y
	imu_msg.angular_velocity.z = imuData.angular_velocity.z

	imu_msg.linear_acceleration.x = imuData.linear_acceleration.x
	imu_msg.linear_acceleration.y = imuData.linear_acceleration.y
	imu_msg.linear_acceleration.z = imuData.linear_acceleration.z

	imu_msg.header = imuData.header

	imu_msg.orientation_covariance = imuData.orientation_covariance
	imu_msg.angular_velocity_covariance = imuData.angular_velocity_covariance
	imu_msg.linear_acceleration_covariance = imuData.linear_acceleration_covariance


rospy.init_node('imu_to_data_raw', anonymous=True)	#create/start new node called "imu_to_data_raw" unless this name is overridden in the launch file
imu_pub = rospy.Publisher('/imu/data_raw/', Imu, queue_size=10)	#publisher that publishes the updated Twist "vel_msg" messages to the "cmd_vel" topic; maintains a queue of 10 Twist messages 
imu_sub = rospy.Subscriber('/camera/imu/', Imu, imuProcess)	#subscriber that subscribes to the "target_position" topic (published to by the "ball_tracking" node) and calls the function "positionProcess"

# def imu_to_data_raw_init():
#     rospy.init_node('imu_to_data_raw', anonymous=True)	#create/start new node called "imu_to_data_raw" unless this name is overridden in the launch file
#     imu_pub = rospy.Publisher('/imu/data_raw/', Imu, queue_size=10)	#publisher that publishes the updated Twist "vel_msg" messages to the "cmd_vel" topic; maintains a queue of 10 Twist messages 

#     imu_sub = rospy.Subscriber('/camera/imu/', Imu, imuProcess)	#subscriber that subscribes to the "target_position" topic (published to by the "ball_tracking" node) and calls the function "positionProcess"


def imu_to_data_raw():
    while not rospy.is_shutdown():
        imu_pub.publish(imu_msg)
        rospy.sleep(delay_time)


if __name__ == '__main__':
    try:
        # imu_to_data_raw_init()       
        imu_to_data_raw()
    except rospy.ROSInterruptException: pass