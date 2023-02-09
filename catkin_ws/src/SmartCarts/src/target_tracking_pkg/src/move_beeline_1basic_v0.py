#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32MultiArray, Float32, Int32
from geometry_msgs.msg import Twist

#if the robot's camera does NOT see the ball (or anything it recognizes/identifies as the ball) then the robot will not move / do anything
#if the robot's camera DOES see the ball then it will do the following:

#robot moves linearly toward or away from ball to keep the ball between "distanceLimitFar" and "distanceLimitNear" in front of itself
distanceLimitFar = 1100	#far distance limit in mm
distanceLimitNear = 700	#near distance limit in mm

#robot rotates toward ball (left or right) to keep the ball between "xPositionLimitLeft" and "xPositionLimitRight"
#i.e.: keep the ball in the robot camera's (x-axis) center of view. The robot doesn't care if the ball is too high or low along the y-axis
#origin of pixel x and y positions is in the top left corner of the camera's screen/field of view
#left/right is the x-axis & up/down is the y-axis
#approximately (x,y) = ?(480,400)? is the center of the camera's field of view
xPositionLimitLeft = 450	#left position limit of x-axis in pixels
xPositionLimitRight = 850	#right position limit of x-axis in pixels

vel_msg = Twist()

temp_distance_publisher = rospy.Publisher('temp_dist', Float32, queue_size=10)
temp_xPosition_publisher = rospy.Publisher('temp_xPos', Int32, queue_size=10)

def distanceProcess(dData):
	if(dData.data < 0):
		vel_msg.linear.x = 0.0
	elif(dData.data > distanceLimitFar):
		vel_msg.linear.x = 0.5
	elif(0 < dData.data < distanceLimitNear):
		vel_msg.linear.x = -0.5
	else:
		vel_msg.linear.x = 0.0

	temp_distance_publisher.publish(dData.data)

def positionProcess(pData):
	xPosition = pData.data[0]

	if(xPosition < 0):
		vel_msg.angular.z = 0.0
	elif(xPosition > xPositionLimitRight):
		vel_msg.angular.z = -1.0
	elif(0 < xPosition < xPositionLimitLeft):
		vel_msg.angular.z = 1.0
	else:
		vel_msg.angular.z = 0.0

	temp_xPosition_publisher.publish(xPosition)


def move():
    # Starts a new node
    rospy.init_node('move_beeline', anonymous=True)
    velocity_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)

    #vel_msg.linear.x = 0	#comment out this line when def distancePosition(dData) is uncommented
    distance_subscriber = rospy.Subscriber('target_distance', Float32, distanceProcess)
    position_subscriber = rospy.Subscriber('target_position', Int32MultiArray, positionProcess)

    vel_msg.linear.y = 0
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0

    while not rospy.is_shutdown():
        velocity_publisher.publish(vel_msg)
        rospy.sleep(0.5)	#need this or there is a long delay for cmd_vel to update...don't know why


if __name__ == '__main__':
    try:
        #Testing our function
        move()
    except rospy.ROSInterruptException: pass
