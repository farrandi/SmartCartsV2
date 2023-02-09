#!/usr/bin/env python
import rospy
#import roslib
from nav_msgs.msg import Odometry
import tf
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Pose, Point, Quaternion
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Int8, Int16, Float32, Float64
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
        self.distanceTravelled = 0.0
        self.lastDistanceTravelled = 0.0

        self.currentYaw = 0.0
        self.deltaYaw = 0.0

        #create/start new node called "move_waypoints" unless this name is overwritten in the launch file
        #"anonymous = True" ensures that the node has a unique name by adding random numbers to the end of "move_waypoints"
        rospy.init_node('move_waypoints', anonymous=True)
        #rate = rospy.Rate(rate_value)   #creates a Rate object called "rate" so program loops at "rate_value" in Hz
        

        #NEEDED PUBLISHERS & SUBSCRIBERS
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size = QUEUE_SIZE)
        self.vel_rate = rospy.Rate(VEL_PUBLISH_RATE)
        
        #subscriber that subscribes to the "Odom" topic and calls the function "odomProcess"
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odomProcess)
        

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
        self.distanceTravelled = self.distance_travelled()

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

        self.state = STATE_GET_NEXT_GOAL

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


#1) GETTING NEXT GOAL FROM USER
    def getNextGoal(self):
        # Get the input from the user.
        self.goal_pose.position.x = float(input("Set your x goal in metres: "))
        self.goal_pose.position.y = float(input("Set your y goal in metres: "))

        # Please, insert a number slightly greater tha#self.state = self.STATE_TURN_TO_GOALn 0 (e.g. 0.01).
        #self.distance_tolerance = input("Set your tolerance in metres: ")

        if self.euclidean_distance() > DISTANCE_TOLERANCE:
            self.state = STATE_TURN_TO_GOAL
            #self.state = self.STATE_DRIVE_TO_GOAL
            self.goalReached = False
        else:
            self.state = STATE_GET_NEXT_GOAL

        self.startingDistance = self.euclidean_distance()
        self.starting_pose = self.current_pose


    #returns distance from current_pose to goal_pose
    def euclidean_distance(self):
        return sqrt( pow((self.goal_pose.position.x - self.current_pose.position.x), 2) + pow((self.goal_pose.position.y - self.current_pose.position.y), 2) )

    #returns distance from starting_pose to current_pose
    def distance_travelled(self):
        return sqrt( pow((self.starting_pose.position.x - self.current_pose.position.x), 2) + pow((self.starting_pose.position.y - self.current_pose.position.y), 2) )


#2) TURNING TO GOAL
    def turnToGoal(self):
        self.deltaYaw = self.get_deltaYaw()
        print("deltaYaw is:", (self.deltaYaw * 180 / PI))
        print("currentYaw is:", (self.currentYaw * 180 / PI))

        #self.startingDistance = self.euclidean_distance()
        #self.starting_pose = self.current_pose

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
        print("waiting 0.5 second")
        rospy.sleep(0.5)

        self.currentDistance = self.euclidean_distance()

        self.startingDistance = self.currentDistance     
        self.lastDistanceTravelled = self.startingDistance - self.euclidean_distance()



#3) DRIVING TO GOAL
    def driveToGoal(self):
        # self.startingDistance = self.euclidean_distance()
        # self.starting_pose = self.current_pose
        # self.currentDistance = self.euclidean_distance()

        # self.startingDistance = self.currentDistance       
        #self.distanceTravelled = 0.0

        # self.starting_pose.position.x = self.current_pose.position.x
        # self.starting_pose.position.y = self.current_pose.position.y

        # print("")
        # print("BEFORE")
        # print("currentDistance BEFORE travel is:", self.currentDistance)
        # print("distanceTravelled BEFORE travel is:", self.distanceTravelled)
        # print("startingDistance BEFORE travel is: ", self.startingDistance)
        # print("")
        # print("self.starting_pose.position.x is:", self.starting_pose.position.x)
        # print("self.starting_pose.position.y is:", self.starting_pose.position.y)
        # print("self.current_pose.position.x is:", self.current_pose.position.x)
        # print("self.current_pose.position.y is:", self.current_pose.position.y)

        #rospy.sleep(0.01)

        #self.starting_pose = self.current_pose

        if (self.currentDistance > DISTANCE_TOLERANCE) and (self.distanceTravelled < self.startingDistance) and (self.lastDistanceTravelled > self.distanceTravelled):
        #while (self.currentDistance > DISTANCE_TOLERANCE) and (self.distanceTravelled < self.startingDistance):

        #while self.distance_travelled() < self.startingDistance:
            self.set_vel(MAX_LINEAR_VEL_X, (Kp_ANG * self.deltaYaw) )
            #self.deltaYaw = self.get_deltaYaw()

            #self.odomProcess(self.odomData)
            
            #self.distanceTravelled = self.distance_travelled()
            
            #print("")
            #print("in between distanceTravelled:", self.distanceTravelled)
            #print("")

            #rospy.spin_some()
            #rospy.spin()
            #rospy.spin_once()
            
            self.state = STATE_DRIVE_TO_GOAL

        else:
            self.state = STATE_AT_GOAL

        rospy.sleep(DELAY_TIME)
        #self.distanceTravelled = self.distance_travelled()
        
        self.distanceTravelled = self.startingDistance - self.euclidean_distance()
        self.lastDistanceTravelled = self.distanceTravelled


        # print("AFTER")
        # print("currentDistance AFTER travel is: ", self.currentDistance)
        print("distanceTravelled AFTER travel is:", self.distanceTravelled)
        # print("startingDistance AFTER travel is: ", self.startingDistance)
        # print("")
        # print("self.starting_pose.position.x is:", self.starting_pose.position.x)
        # print("self.starting_pose.position.y is:", self.starting_pose.position.y)
        # print("self.current_pose.position.x is:", self.current_pose.position.x)
        # print("self.current_pose.position.y is:", self.current_pose.position.y)



        # self.starting_pose.position.x = self.current_pose.position.x
        # self.starting_pose.position.y = self.current_pose.position.y

        #self.starting_pose = self.current_pose

        #self.state = STATE_AT_GOAL


if __name__ == '__main__':
    try:
        #rospy.sleep(5)  #delay start of control program by 5 seconds

        cart = SmartCart()

        while not rospy.is_shutdown():  #run infinite loop 
            if cart.state == STATE_AT_GOAL:        #STATE_AT_GOAL = 0
                print("current state is: 0 (STATE_AT_GOAL)")
                cart.atGoal()
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
