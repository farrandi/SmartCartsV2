#!/usr/bin/env python
from math import pow, atan2, sqrt, pi
from tf.transformations import euler_from_quaternion

import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
#Layout Preset Path in Waypoint Poses

WP0 = Pose(Point(1.5,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP1 = Pose(Point(3.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP2 = Pose(Point(3.0,3.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP3 = Pose(Point(1.5,3.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP4 = Pose(Point(0.0,3.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP5 = Pose(Point(0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
LOOP = True

DELAY_TIME = 0.005

THRESHOLD_YAW_RADIANS = (1.5)*pi/180    #when turning to goal, this is the tolerance +/- in radians 
DISTANCE_TOLERANCE = 0.01   #robot will travel to this value in metres around the goal position

MAX_LINEAR_VEL_X = 0.25     #maximum linear x velocity to use in m/s
MAX_ANGULAR_VEL_Z = 0.25    #maximum angular z velocity to use in rad/s
MIN_ANGULAR_VEL_Z = 0.00    #minimum angular z velocity to use in rad/s

VEL_PUBLISH_RATE = 20    #5Hz velocity message publish rate
LED_PUBLISH_RATE = 3    #3Hz LED message publish rate
TEST_PUBLISH_RATE = 1    #1Hz test messages publish rate
QUEUE_SIZE = 10

KP_ANG = 0.8

#Possible SmartCart STATES:
STATE_AT_GOAL = 0
STATE_GET_NEXT_GOAL = 1
STATE_TURN_TO_GOAL = 2
STATE_DRIVE_TO_GOAL = 3
STATE_COMPLETE = 4

class SmartCart:
    def __init__(self):
        self.goal_pose = Pose(Point(0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
        self.current_pose = Pose(Point(0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
        self.starting_pose = Pose(Point(0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
        self.startingDistance = 0.0 #Set current distance between current_pose and goal_pose

        self.vel = Twist()
        self.vel.linear.x = 0.0
        self.vel.linear.y = 0.0
        self.vel.linear.z = 0.0
        self.vel.angular.x = 0.0
        self.vel.angular.y = 0.0
        self.vel.angular.z = 0.0

        self.goalIndex = 0
        self.currentYaw = 0.0   #robot's current angle in radians relative to odom frame's positive x-axis; CCW is positive
        self.deltaYaw_unfiltered = 0.0
        self.deltaYaw = 0.0 #difference between goalYaw and currentYaw

        self.LED = Bool()
        self.LED.data = 0

        self.waypoints = [WP0, WP1, WP2, WP3, WP4, WP5]

        rospy.init_node('preset_path', anonymous=True)

        #NEEDED PUBLISHERS & SUBSCRIBERS
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size = QUEUE_SIZE)
        self.vel_rate = rospy.Rate(VEL_PUBLISH_RATE)

        self.LED_pub = rospy.Publisher('LEDsignal', Bool, queue_size = QUEUE_SIZE)
        self.LED_rate = rospy.Rate(LED_PUBLISH_RATE)
        
        #subscriber that subscribes to the "Odom" topic and calls the function "odomProcess"
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odomProcess)

        self.state = STATE_AT_GOAL #Set state so that Initially, we get next goal from user
        print("SmartCart Initialized")


    #Publishing Functions
    def set_vel(self, forward, turn):
        self.vel.linear.x = forward
        
        if turn >= MAX_ANGULAR_VEL_Z:
            self.vel.angular.z = MAX_ANGULAR_VEL_Z
        elif 0 <= turn <= MIN_ANGULAR_VEL_Z:
            self.vel.angular.z = MIN_ANGULAR_VEL_Z
        
        elif turn < (-1 * MAX_ANGULAR_VEL_Z):
            self.vel.angular.z = -1 * MAX_ANGULAR_VEL_Z
        elif (-1 * MIN_ANGULAR_VEL_Z) <= turn < 0:
            self.vel.angular.z = -1 * MIN_ANGULAR_VEL_Z
        
        else:
            self.vel.angular.z = turn

        self.vel_pub.publish(self.vel)
        self.vel_rate.sleep()

    def set_LED(self, LED_state):
        self.LED.data = LED_state
        self.LED_pub.publish(self.LED)
        

    #Helper Functions
    def odomProcess(self, odomData):
        self.current_pose.position.x = odomData.pose.pose.position.x
        self.current_pose.position.y = odomData.pose.pose.position.y
        self.currentYaw = euler_from_quaternion([odomData.pose.pose.orientation.x, odomData.pose.pose.orientation.y, odomData.pose.pose.orientation.z, odomData.pose.pose.orientation.w])[2]

    def euclidean_distance(self):
        return sqrt( pow((self.goal_pose.position.x - self.current_pose.position.x), 2) + pow((self.goal_pose.position.y - self.current_pose.position.y), 2) )
    
    def distance_travelled(self):
        return sqrt( pow((self.starting_pose.position.x - self.current_pose.position.x), 2) + pow((self.starting_pose.position.y - self.current_pose.position.y), 2) )
    
    def get_deltaYaw(self):
        # atan2(y, x) returns value of atan(y/x) in radians. The atan2() method returns a numeric value between -pi and pi representing the angle theta of a (x, y) point and positive x-axis.
        self.deltaYaw_unfiltered = atan2(self.goal_pose.position.y - self.current_pose.position.y, self.goal_pose.position.x - self.current_pose.position.x) - self.currentYaw

        if self.deltaYaw_unfiltered > pi:
            return (self.deltaYaw_unfiltered - 2*pi)
        elif self.deltaYaw_unfiltered < (-1*pi):
            return (2*pi + self.deltaYaw_unfiltered)
        else:
            return self.deltaYaw_unfiltered


    #Goal Setting/Getting Functions
    #STATE 0
    def atGoal(self):
        self.set_vel(0.0, 0.0)
        self.set_LED(1)
        print("Goal Reached!")
        print("")
        print("")

        self.state = STATE_GET_NEXT_GOAL
        print("Current state is: 1 (STATE_GET_NEXT_GOAL)")

    #STATE 1
    def getNextGoal(self):
        if self.goalIndex == len(self.waypoints):
            if LOOP:
                self.goalIndex = 0
            else:
                self.state = STATE_COMPLETE
                return

        self.goal_pose.position.x = self.waypoints[self.goalIndex].position.x
        self.goal_pose.position.y = self.waypoints[self.goalIndex].position.y
        self.goalIndex += 1
        rospy.sleep(1.0)

        if self.euclidean_distance() > DISTANCE_TOLERANCE:
            self.startingDistance = self.euclidean_distance()
            self.starting_pose.position.x = self.current_pose.position.x
            self.starting_pose.position.y = self.current_pose.position.y

            self.set_LED(0)
            self.state = STATE_TURN_TO_GOAL
            print("Current state is: 2 (STATE_TURN_TO_GOAL)")
        else:
            print("Goal position you input is too close to the current position, please change your waypoints so they are greater than ", DISTANCE_TOLERANCE, "metres from each other")
            self.state = STATE_COMPLETE

    #STATE 2
    def turnToGoal(self):
        self.deltaYaw = self.get_deltaYaw()

        if abs(self.deltaYaw) > THRESHOLD_YAW_RADIANS:
            self.set_vel(0.0, (KP_ANG * self.deltaYaw) )
        else:
            self.set_vel(0.0, 0.0)
            print("NOW FACING goal_pose")
            rospy.sleep(1.0)

            self.state = STATE_DRIVE_TO_GOAL
            print("current state is: 3 (STATE_DRIVE_TO_GOAL)")

    #STATE 3
    def driveToGoal(self):
        if (self.euclidean_distance() > DISTANCE_TOLERANCE) and (self.distance_travelled() < self.startingDistance):
            self.set_vel(MAX_LINEAR_VEL_X, (KP_ANG * self.deltaYaw) )
            # print("distance travelled is:", self.distance_travelled())
        else:
            self.state = STATE_AT_GOAL
            print("current state is: 0 (STATE_AT_GOAL)")

if __name__ == "__main__":
    try:
        cart = SmartCart()

        while not rospy.is_shutdown():  #run infinite loop 
            if cart.state == STATE_AT_GOAL:        #STATE_AT_GOAL = 0
                # print("current state is: 0 (STATE_AT_GOAL)")
                cart.atGoal()
            elif cart.state == STATE_GET_NEXT_GOAL:  #STATE_GET_NEXT_GOAL = 1
                # print("current state is: 1 (STATE_GET_NEXT_GOAL)")
                cart.getNextGoal()
            elif cart.state == STATE_TURN_TO_GOAL:   #STATE_TURN_TO_GOAL = 2
                # print("current state is: 2 (STATE_TURN_TO_GOAL)")
                cart.turnToGoal()
            elif cart.state == STATE_DRIVE_TO_GOAL:  #STATE_DRIVE_TO_GOAL = 3
                # print("current state is: 3 (STATE_DRIVE_TO_GOAL)")
                cart.driveToGoal()
            elif cart.state == STATE_COMPLETE:
                print("Path is complete!  Shutting Down...")
                cart.set_vel(0.0, 0.0)
                break
            else:
                print("ERROR: NOT in any state")
                cart.set_vel(0.0, 0.0)

    except rospy.ROSInterruptException: pass

