#!/usr/bin/env python
import rospy
#import roslib
from nav_msgs.msg import Odometry
import tf
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Pose, Point, Quaternion
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Int8, Int16, Int32, Float32, Float64, Int32MultiArray
import copy
import math
from math import pow, atan2, sqrt
import time
#import numpy as np


PI = 3.1415926535

WP1 = Pose(Point(0.5,0.5,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP2 = Pose(Point(-0.5,0.5,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP3 = Pose(Point(-0.5,-0.5,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP4 = Pose(Point(0.5,-0.5,0.0), Quaternion(0.0,0.0,0.0,1.0))


DISTANCE_LIMIT_FAR = 600  #far distance limit in mm
#DISTANCE_LIMIT_NEAR = 500 #near distance limit in mm

#robot rotates toward ball (left or right) to keep the ball between "xPositionLimitLeft" and "xPositionLimitRight"
#i.e.: keep the ball in the robot camera's (horizontal x-axis) center of view. The robot doesn't care if the ball is vertically too high or low along the y-axis.
#origin of pixel x and y positions is in the top left corner of the camera's screen/field of view
#left/right is the x-axis & up/down is the y-axis
#approximately (x,y) = ?(480,400)? is the center of the camera's field of view
XPOSITION_LIMIT_RIGHT = 400    #left position limit of x-axis in pixels
XPOSITION_LIMIT_LEFT = 800   #right position limit of x-axis in pixels
XPOSITION_CENTER = 650 #I estimated this center pixel position using the depth ball camera node
XPOSITION_THRESHOLD = 150

DELAY_TIME = 0.005

THRESHOLD_YAW_RADIANS = (1.5)*PI/180
DISTANCE_TOLERANCE = 0.05

LINEAR_VEL_X = 0.25
MAX_LINEAR_VEL_X = 0.25

ANGULAR_VEL_Z = 0.25
MAX_ANGULAR_VEL_Z = 0.25

VEL_PUBLISH_RATE = 3    #3Hz velocity message publish rate
TEST_PUBLISH_RATE = 1    #1Hz test messages publish rate
QUEUE_SIZE = 10

#NEXT_WP = Pose()
#test waypoint is: position (0.5=x,0.5=y,0.0=z) and quaternion (0.0,0.0,0.0=sin(theta/2),1.0=cos(theta/s)) so (yaw) theta = 0.0 radians (both in odom coordinate frame (fixed world frame))
#NEXT_WP = WP1

#All SmartCart STATES:self.lastDistance
STATE_AT_GOAL = 0
STATE_KEEP_TARGET_CENTERED = 0.5
STATE_GET_NEXT_GOAL = 1
STATE_TURN_TO_GOAL = 2
STATE_DRIVE_TO_GOAL = 3

Kp_ANG = 0.5


class SmartCart:
    #Class Constants:
    #DELAY_TIME = 0.1    #number of secondes to delay the program

    def __init__(self):
        self.goal_pose = Pose(Point(0.001,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
        self.current_pose = Pose(Point(0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
        self.starting_pose = Pose(Point(0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))

        self.vel = Twist()
        self.vel.linear.x = 0.0
        self.vel.linear.y = 0.0
        self.vel.linear.z = 0.0
        self.vel.angular.x = 0.0
        self.vel.angular.y = 0.0
        self.vel.angular.z = 0.0

        #self.zero_vel = copy.deepcopy(self.vel)

        self.goalReached = False  #TRUE if the robot's base_link (centeVEL_PUBLISH_RATE = 5    #5Hz velocity message publish rater of rotation) has reached the goal waypoint within the "goal_radius", FALSE otherwise
        #self.nextWPFacing = False   #TRUE if the robot is facing the next waypoint, FALSE otherwise

        self.currentDistance = 0.0
        self.startingDistance = 0.0

        self.currentYaw = 0.0
        self.deltaYaw = 0.0

        self.target_distance = 0.0
        self.target_xposition = XPOSITION_CENTER

        #create/start new node called "move_waypoints" unless this name is overwritten in the launch file
        #"anonymous = True" ensures that the node has a unique name by adding random numbers to the end of "move_waypoints"
        rospy.init_node('move_waypoints', anonymous=True)
        #rate = rospy.Rate(rate_value)   #creates a Rate object called "rate" so program loops at "rate_value" in Hz
        

        #NEEDED PUBLISHERS & SUBSCRIBERS
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size = QUEUE_SIZE)
        self.vel_rate = rospy.Rate(VEL_PUBLISH_RATE)
        
        self.temp_distance_publisher = rospy.Publisher('temp_dist', Float32, queue_size=10)  #publisher that publishes the target distance (Float32) to a "temp_dist" topic for troubleshooting purposes
        self.temp_xPosition_publisher = rospy.Publisher('temp_xPos', Int32, queue_size=10)   #publisher that publish the target xPosition (Int32) to a "temp_xPos" topic for troubleshooting purposes


        #subscriber that subscribes to the "Odom" topic and calls the function "odomProcess"
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odomProcess)
        self.distance_sub = rospy.Subscriber('target_distance', Float32, self.distanceProcess) #subscriber that subscribes to the "target_distance" topic (published to by the "ball_tracking" node) and calls the function "distanceProcess"
        self.position_sub = rospy.Subscriber('target_position', Int32MultiArray, self.positionProcess) #subscriber that subscribes to the "target_position" topic (published to by the "ball_tracking" node) and calls the function "positionProcess"


        #FOR TROUBLESHOOTING
        self.distance_pub = rospy.Publisher('currentDistance', Float32, queue_size = QUEUE_SIZE)
        self.distance_rate = rospy.Rate(TEST_PUBLISH_RATE)

        self.deltaYaw_pub = rospy.Publisher('deltaYaw', Float32, queue_size = QUEUE_SIZE)
        self.deltaYaw_rate = rospy.Rate(TEST_PUBLISH_RATE)

        self.currentYaw_pub = rospy.Publisher('currentYaw', Float32, queue_size = QUEUE_SIZE)
        self.currentYaw_rate = rospy.Rate(TEST_PUBLISH_RATE)

        #publisher that publishes TRUE/FALSE to the "WPgoal_reached" topic if the robot's base_link HAS/HASN'T reached the next Waypoint (within the WPgoal_radius)
        #mantains a queue of 10 bool messages
        #self.goalReached_pub = rospy.Publisher('Goal_Reached', Bool, queue_size = QUEUE_SIZE)
        #self.goalReached_rate = rospy.Rate(TEST_PUBLISH_RATE)
        
        self.state = STATE_AT_GOAL #Set state so that Initially, we get next goal from user

        #self.goalReached_pub.publish(self.goalReached)

        print("SmartCart Initialized")


    #function called whenever the "distance_subscriber" subscriber below subscribes a message from the "target_distance" topic (published to by the "ball_tracking" node)
    #this message is a Float32 representing the distance of the target/ball from the camera & is stored in variable "dData"
    def distanceProcess(self, dData):
        self.target_distance = dData.data

        # if(dData.data < 0): #dData.data accesses the Float32 stored in the message "dData"; dData.data is -1 if no target is seen by the "ball_tracking" node/python program   
        #     vel_msg.linear.x = 0.0  #if no target detected, command the robot to stop moving linearly
        # elif(dData.data > DISTANCE_LIMIT_FAR):    #if target farther than "distanceLimitFar", command robot to move forward, toward the target
        #     vel_msg.linear.x = MAX_LINEAR_VEL_X
        # elif(0 <= dData.data < DISTANCE_LIMIT_NEAR):  #if target closer than "distanceLimitNear" (and positive), command robot to move backwards, away from the target
        #     vel_msg.linear.x = -1 * MAX_LINEAR_VEL_X
        # else:   #if distanceLimitNear <= dData.data <= distanceLimitFar, command the robot to stop moving linearly
        #     vel_msg.linear.x = 0.0
        
        # self.velocity_publisher.publish(vel_msg)
        self.temp_distance_publisher.publish(self.target_distance) #publish the target distance (Float32) to a "temp_dist" topic for troubleshooting purposes

    #function called whenever the "position_subscriber" subscriber below subscribes a message from the "target_position" topic (published to by the "ball_tracking" node)
    #this message is a python list (i.e.: array) of two 32 bit integers (Int32) representing the x and y positions in the camera's field of view & is stored in the variable "pData" 
    def positionProcess(self, pData):
        self.target_xposition = pData.data[0]   #pData.data[0] accesses the Int32 stored in the first element of the pData list; store this Int32 in the variable "xPosition"

        # if(self.target_xposition < 0):  #if xPosition is -1 (no target detected by the "ball tracking" node) then command the robot to stop moving angularly 
        #     vel_msg.angular.z = 0.0
        # elif(self.target_xposition > XPOSITION_LIMIT_RIGHT):  #if target is too far to the right in the camera's field of view, command the robot to rotate CW (plan view)
        #     vel_msg.angular.z = -1 * MAX_ANGULAR_VEL_Z
        # elif(0 <= self.target_xposition < XPOSITION_LIMIT_LEFT):  #if target is too far to the left in the camera's field of view, command the robot to rotate CCW (plan view)
        #     vel_msg.angular.z = MAX_ANGULAR_VEL_Z
        # else:   #if xPositionLimitLeft <= xPosition <= xPositionLimitRight, command the robot to stop moving angularly
        #     vel_msg.angular.z = 0.0

        # self.velocity_publisher.publish(vel_msg)
        self.temp_xPosition_publisher.publish(self.target_xposition) #publish the target xPosition (Int32) to a "temp_xPos" topic for troubleshooting purposes


    def odomProcess(self, odomData):
        self.current_pose.position.x = odomData.pose.pose.position.x
        self.current_pose.position.y = odomData.pose.pose.position.y
        self.currentDistance = self.euclidean_distance()

        self.currentYaw = euler_from_quaternion([odomData.pose.pose.orientation.x, odomData.pose.pose.orientation.y, odomData.pose.pose.orientation.z, odomData.pose.pose.orientation.w])[2]
        self.deltaYaw = self.get_deltaYaw()

        self.distance_pub.publish(self.currentDistance)
        #self.distance_rate.sleep()

        # self.currentYaw_pub.publish( self.currentYaw * 180 / PI )
        # self.deltaYaw_pub.publish( self.deltaYaw * 180 / PI )
        rospy.sleep(DELAY_TIME)
        
        #self.deltaYaw_rate.sleep()
        
        # self.goalReached_pub.publish(cart.goalReached)
        # self.goalReached_rate.sleep()

    def get_deltaYaw(self):
        return ( ( atan2(self.goal_pose.position.y - self.current_pose.position.y, self.goal_pose.position.x - self.current_pose.position.x) ) - self.currentYaw )


#0) GOAL HAS BEEN REACHED; STOPPING ROBOT
    def atGoal(self):
        self.set_vel(0.0, 0.0)
        self.goalReached = True
        print("Goal Reached!")
        print("")
        print("")

        self.state = STATE_KEEP_TARGET_CENTERED

    def set_vel(self, forward, turn):
        self.vel.linear.x = forward
        
        if(turn > MAX_ANGULAR_VEL_Z):
            self.vel.angular.z = MAX_ANGULAR_VEL_Z
        elif(turn < (-1 * MAX_ANGULAR_VEL_Z) ):
            self.vel.angular.z = -1 * MAX_ANGULAR_VEL_Z
        else:
            self.vel.angular.z = turn

        self.vel_pub.publish(self.vel)
        #self.vel_rate.sleep()
        rospy.sleep(DELAY_TIME)


#0.5) ROTATE ROBOT TO KEEP TARGET CENTERED IN ITS CAMERA'S FIELD OF VIEW
    def keepTargetCentered(self):
        while abs(self.target_xposition) > XPOSITION_THRESHOLD and self.target_distance < DISTANCE_LIMIT_FAR:
            self.set_vel(0.0, (Kp_ANG * (XPOSITION_CENTER - self.target_xposition)) )            

        self.set_vel(0.0, 0.0)

        if self.target_xposition < 0:
            self.state = STATE_KEEP_TARGET_CENTERED
        else:
            self.state = STATE_DRIVE_TO_GOAL
        
        #self.startingDistance = self.euclidean_distance()
        self.starting_pose = self.current_pose
        # print("waiting 1 second")
        # rospy.sleep(1)


#1) GETTING NEXT GOAL FROM USER
    def getNextGoal(self):
        # Get the input from the user.
        # self.goal_pose.position.x = float(input("Set your x goal in metres: "))
        # self.goal_pose.position.y = float(input("Set your y goal in metres: "))


        #self.goal_pose.position.x = 
        #self.goal_pose.position.y = 

        if self.euclidean_distance() > DISTANCE_TOLERANCE:
            self.state = STATE_TURN_TO_GOAL
            #self.state = STATE_DRIVE_TO_GOAL
            self.goalReached = False
        else:
            self.state = STATE_GET_NEXT_GOAL

    #returns distance from current_pose to goal_pose
    def euclidean_distance(self):
        return sqrt( pow((self.goal_pose.position.x - self.current_pose.position.x), 2) + pow((self.goal_pose.position.y - self.current_pose.position.x), 2) )

    #returns distance from starting_pose to current_pose
    def distance_travelled(self):
        return sqrt( pow((self.starting_pose.position.x - self.current_pose.position.x), 2) + pow((self.starting_pose.position.y - self.current_pose.position.x), 2) )


#2) TURNING TO GOAL
    def turnToGoal(self):
        self.deltaYaw = self.get_deltaYaw()
        print("deltaYaw is:", (self.deltaYaw * 180 / PI))
        print("currentYaw is:", (self.currentYaw * 180 / PI))

        self.startingDistance = self.euclidean_distance()
        self.starting_pose = self.current_pose

        while abs(self.deltaYaw) > THRESHOLD_YAW_RADIANS:
            # if self.deltaYaw > THRESHOLD_YAW_RADIANS:
            #     self.set_vel(0.0, ANGULAR_VEL_Z)
            # elif self.deltaYaw < (-1*THRESHOLD_YAW_RADIANS):
            #     self.set_vel(0.0, (-1*ANGULAR_VEL_Z))
            self.set_vel(0.0, (Kp_ANG * self.deltaYaw) )

            self.deltaYaw = self.get_deltaYaw()

        self.set_vel(0.0, 0.0)

        self.state = STATE_DRIVE_TO_GOAL
        
        #self.state = STATE_AT_GOAL
        print("waiting 1 second")
        rospy.sleep(1)



#3) DRIVING TO GOAL
    def driveToGoal(self):
        # self.startingDistance = self.euclidean_distance()
        # self.starting_pose = self.current_pose

        #while (self.currentDistance > DISTANCE_TOLERANCE) and (self.distance_travelled() < self.startingDistance):
            #self.set_vel(LINEAR_VEL_X, (Kp_ANG * self.deltaYaw) )
        while self.distance_travelled() < self.target_distance:
            self.set_vel(LINEAR_VEL_X, (Kp_ANG * (XPOSITION_CENTER - self.target_xposition)) ) 

        self.state = STATE_AT_GOAL


if __name__ == '__main__':
    try:
        #rospy.sleep(5)  #delay start of control program by 5 seconds

        cart = SmartCart()

        while not rospy.is_shutdown():  #run infinite loop 
            if cart.state == STATE_AT_GOAL:        #STATE_AT_GOAL = 0
                print("current state is: 0 (STATE_AT_GOAL)")
                cart.atGoal()
            elif cart.state == STATE_KEEP_TARGET_CENTERED:
                print("current state is: 0.5 (KEEP_TARGET_CENTERED)") #KEEP_TARGET_CENTERED = 0.5
                cart.keepTargetCentered()
            elif cart.state == STATE_GET_NEXT_GOAL:  #STATE_GET_NEXT_GOAL = 1
                print("current state is: 1 (STATE_GET_NEXT_GOAL)")
                cart.getNextGoal()
            elif cart.state == STATE_TURN_TO_GOAL:   #STATE_TURN_TO_GOAL = 2
                print("current state is: 2 (STATE_TURN_TO_GOAL)")
                cart.turnToGoal()
            elif cart.state == STATE_DRIVE_TO_GOAL:  #STATE_DRIVE_TO_GOAL = 3
                print("current state is: 3 (STATE_DRIVE_TO_GOAL)")
                cart.driveToGoal()
            else:
                print("ERROR: NOT in any state")
                cart.set_vel(0.0, 0.0)

            #rospy.sleep(delay_time) #delay loop/program by "delay_time". This is needed or there is a long delay for cmd_vel to update...don't know why.

    except rospy.ROSInterruptException: pass
