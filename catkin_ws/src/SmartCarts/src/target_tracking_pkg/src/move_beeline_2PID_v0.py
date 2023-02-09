#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32MultiArray, Float32, Int32
from geometry_msgs.msg import Twist


delay_time = 0.1	#number of seconds to delay the program
linearX = 0.5	#linear x (forward/reverse) speed in m/s to use for keeping the target between "distanceLimitNear" and "distanceLimitFar"
angularZ = 1.0	#angular z (CW/CCW turning) speed in rad/s to use for keeping target centered in the camera's field of view between "xPositionLimitLeft" and "xPositionLimitRight"

#if the robot's camera does NOT see the ball (or anything it recognizes/identifies as the ball) then the robot will not move / do anything
#if the robot's camera DOES see the ball then it will do the following:
#robot moves linearly toward or away from ball to keep the ball between "distanceLimitFar" and "distanceLimitNear" in front of itself
linLimitFar = 1100.0	#far distance limit in mm
linLimitNear = 700.0	#near distance limit in mm
linSpeedMax = 1.5

#robot rotates toward ball (left or right) to keep the ball between "xPositionLimitLeft" and "xPositionLimitRight"
#i.e.: keep the ball in the robot camera's (horizontal x-axis) center of view. The robot doesn't care if the ball is vertically too high or low along the y-axis.
#origin of pixel x and y positions is in the top left corner of the camera's screen/field of view
#left/right is the x-axis & up/down is the y-axis
#approximately (x,y) = ?(480,400)? is the center of the camera's field of view
xPositionLimitLeft = 450.0	#left position limit of x-axis in pixels
xPositionLimitRight = 850.0	#right position limit of x-axis in pixels
angSpeedMax = 1.0


linKp = 0.0
linKi = 0.0
linKd = 0.0
linBias = 0.0

linTime_current = 0.0
linTime_last = rospy.Time.now()
linDesired = 900.0	#desired linear distance in mm of target in front of camera
linActual = 0.0
linError = 0.0
linError_last = 0.0
linIntegral = 0.0
linDerivative = 0.0
linOutput = 0.0


angKp = 0.0
angki = 0.0
angKd = 0.0
angBias = 0.0

angTime_current = 0.0
angTime_last = rospy.Time.now()
angDesired = 650.0	#desired xPosition in number of pixels to keep target centered (origin of pixels at top left corner of camera's field of view)
angActual = 0.0
angError = 0.0
angError_last = 0.0
angIntegral = 0.0
angDerivative = 0.0
angOutput = 0.0


vel_msg = Twist()	#Twist message to prepare and publish to the cmd_vel topic to move the robot


#function called whenever the "distance_subscriber" subscriber below subscribes a message from the "target_distance" topic (published to by the "ball_tracking" node)
#this message is a Float32 representing the distance of the target/ball from the camera & is stored in variable "dData"
def distanceProcess(dData):
	linActual = dData.data
	linTime_current = rospy.Time.now()

	if(linActual < 0):
		vel_msg.linear.x = 0.0
		linError_last = 0.0
		linIntegral = 0.0
	else:
		linError = linDesired - linActual
		linIntegral += linError * (linTime_current - linTime_last)
		linDerivative = (linError - linError_last) / (linTime_current - linTime_last)
		
		linOutput = (linKp * linError) + (linKi * linIntegral) + (linKd * linDerivative) + linBias
		
		if(linOutput > linSpeedMax):
			vel_msg.linear.x = linSpeedMax
		elif(linOutput < (-1*linSpeedMax)):
			vel_msg.linear.x = -1*linSpeedMax
		else:
			vel_msg.linear.x = linOutput

		linError_last = linError

	linTime_last = linTime_current
	temp_distance_publisher.publish(linActual)	#publish the target distance (Float32) to a "temp_dist" topic for troubleshooting purposes


#function called whenever the "position_subscriber" subscriber below subscribes a message from the "target_position" topic (published to by the "ball_tracking" node)
#this message is a python list (i.e.: array) of two 32 bit integers (Int32) representing the x and y positions in the camera's field of view & is stored in the variable "pData" 
def positionProcess(pData):
	xPosition = pData.data[0]	#pData.data[0] accesses the Int32 stored in the first element of the pData list; store this Int32 in the variable "xPosition"

	if(xPosition < 0):	#if xPosition is -1 (no target detected by the "ball tracking" node) then command the robot to stop moving angularly 
		vel_msg.angular.z = 0.0
	elif(xPosition > xPositionLimitRight):	#if target is too far to the right in the camera's field of view, command the robot to rotate CW (plan view)
		vel_msg.angular.z = -1 * angularZ
	elif(0 <= xPosition < xPositionLimitLeft):	#if target is too far to the left in the camera's field of view, command the robot to rotate CCW (plan view)
		vel_msg.angular.z = angularZ
	else:	#if xPositionLimitLeft <= xPosition <= xPositionLimitRight, command the robot to stop moving angularly
		vel_msg.angular.z = 0.0

	temp_xPosition_publisher.publish(xPosition)	#publish the target xPosition (Int32) to a "temp_xPos" topic for troubleshooting purposes


def move_beeline_init():
    rospy.init_node('move_beeline_2PID', anonymous=True)	#create/start new node called "move_beeline_2PID" unless this name is overridden in the launch file
    velocity_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)	#publisher that publishes the updated Twist "vel_msg" messages to the "cmd_vel" topic; maintains a queue of 10 Twist messages 

    temp_distance_publisher = rospy.Publisher('temp_dist', Float32, queue_size=10)	#publisher that publishes the target distance (Float32) to a "temp_dist" topic for troubleshooting purposes
	temp_xPosition_publisher = rospy.Publisher('temp_xPos', Int32, queue_size=10)	#publisher that publish the target xPosition (Int32) to a "temp_xPos" topic for troubleshooting purposes

    distance_subscriber = rospy.Subscriber('target_distance', Float32, distanceProcess)	#subscriber that subscribes to the "target_distance" topic (published to by the "ball_tracking" node) and calls the function "distanceProcess"
    position_subscriber = rospy.Subscriber('target_position', Int32MultiArray, positionProcess)	#subscriber that subscribes to the "target_position" topic (published to by the "ball_tracking" node) and calls the function "positionProcess"

    #initialize all values of the Twist message "vel_msg" to zero except for "vel_msg.linear.x" and "vel_msg.angular.z"
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0


def move_beeline():
    while not rospy.is_shutdown():	#run infinite loop
        velocity_publisher.publish(vel_msg)	#publish the updated "vel_msg" message modified by "distanceProcess" and "positionProcess" to the "cmd_vel" topic to move the robot
        rospy.sleep(delay_time)	#delay loop/program by "delay_time". This is needed or there is a long delay for cmd_vel to update...don't know why.


if __name__ == '__main__':
    try:
        move_beeline_init()       
        move_beeline()
    except rospy.ROSInterruptException: pass
