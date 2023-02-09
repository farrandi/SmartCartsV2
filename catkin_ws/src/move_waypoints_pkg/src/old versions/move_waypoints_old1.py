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
import time
#import numpy as np


PI = 3.1415926535

WP1 = Pose(Point(0.5,0.5,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP2 = Pose(Point(-0.5,0.5,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP3 = Pose(Point(-0.5,-0.5,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP4 = Pose(Point(0.5,-0.5,0.0), Quaternion(0.0,0.0,0.0,1.0))


class SmartCart:
    #Class Constants:
    #DELAY_TIME = 0.1    #number of secondes to delay the program


    WP_GOAL_RADIUS = 0.15    #robot has succesfully reached the goal waypoint if its base_link (center of rotation) is within the "goal_radius" (in metres) around the goal waypoint    
    THRESHOLD_YAW_RADIANS = (1.0)*PI/180

    LINEAR_VEL_X = 0.5
    ANGULAR_VEL_Z = 0.9
    MAX_ANGULAR_VEL_Z = 1.5

    VEL_PUBLISH_RATE = 5    #5Hz velocity message publish rate
    TEST_PUBLISH_RATE = 5    #5Hz test messages publish rate
    QUEUE_SIZE = 10

    #NEXT_WP = Pose()
    #test waypoint is: position (0.5=x,0.5=y,0.0=z) and quaternion (0.0,0.0,0.0=sin(theta/2),1.0=cos(theta/s)) so (yaw) theta = 0.0 radians (both in odom coordinate frame (fixed world frame))
    NEXT_WP = WP1

    #All SmartCart STATES:
    STATE_AT_GOAL = 0
    STATE_GET_NEXT_WP = 1
    STATE_TURN_TO_WP = 2
    STATE_DRIVE_TO_WP = 3


    KP_ANG = 0.0
    KI_ANG = 0.0
    KD_ANG = 0.0
    BIAS_ANG = 0.0


    def __init__(self):
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

        self.currentX = 0.0
        self.currentY = 0.0
        #self.currentOrientation = Quaternion()
        #self.currentEuler = [0.0, 0.0, 0.0]
        #self.currentRoll = 0.0
        #self.currentPitch = 0.0
        self.currentYaw = 0.0

        self.nextX = 0.0
        self.nextY = 0.0
        #self.nextYaw = 0.0

        self.deltaYaw = 0.0


        self.time_current_ang = 0.0
        #self.time_last_ang = 0.0
        # time_last_ang = rospy.Time.now()
        self.delta_time_ang = 0.0
        #angDesired = 650.0  #desired xPosition in number of pixels to keep target centered (origin of pixels at top left corner of camera's field of view)
        self.actual_ang = 0.0
        self.error_ang = 0.0
        self.error_last_ang = 0.0
        self.integral_ang = 0.0
        self.derivative_ang = 0.0
        self.output_ang = 0.0


        #create/start new node called "move_waypoints" unless this name is overwritten in the launch file
        #"anonymous = True" ensures that the node has a unique name by adding random numbers to the end of "move_waypoints"
        rospy.init_node('move_waypoints', anonymous=True)
        #rate = rospy.Rate(rate_value)   #creates a Rate object called "rate" so program loops at "rate_value" in Hz
        
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size = self.QUEUE_SIZE)
        self.vel_rate = rospy.Rate(self.VEL_PUBLISH_RATE)

        self.deltaYaw_pub = rospy.Publisher('deltaYaw', Float64, queue_size = self.QUEUE_SIZE)
        self.deltaYaw_rate = rospy.Rate(self.TEST_PUBLISH_RATE)

        self.state_pub = rospy.Publisher('SmartCart_State', Int8, queue_size = self.QUEUE_SIZE)
        self.state_rate = rospy.Rate(self.TEST_PUBLISH_RATE)
        #publisher that publishes TRUE/FALSE to the "WPgoal_reached" topic if the robot's base_link HAS/HASN'T reached the next Waypoint (within the WPgoal_radius)
        #mantains a queue of 10 bool messages
        self.goalReached_pub = rospy.Publisher('Goal_Reached', Bool, queue_size = self.QUEUE_SIZE)
        self.goalReached_rate = rospy.Rate(self.TEST_PUBLISH_RATE)
        #subscriber that subscribes to the "Odom" topic (published to by the "odom_enc_node" node) and calls the function "odomProcess"
        self.odom_sub = rospy.Subscriber('odom_enc_node', Odometry, self.odomProcess)
        #nextWP_SUB = rospy.Subscriber('nextWP',Pose, nextWPProcess)
        self.KP_ANG_sub = rospy.Subscriber('KP_ANG', Float32, self.KP_ANGProcess)
        self.KI_ANG_sub = rospy.Subscriber('KI_ANG', Float32, self.KI_ANGProcess)
        self.KD_ANG_sub = rospy.Subscriber('KD_ANG', Float32, self.KD_ANGProcess)


        #Initially get next waypoint
        self.state = self.STATE_GET_NEXT_WP

        self.vel_pub.publish(self.vel)
        self.vel_rate.sleep()

        self.goalReached_pub.publish(self.goalReached)

        print("SmartCart Initialized")

        self.time_last_ang = rospy.get_time()
        #self.time_last_ang = rospy.get_rostime().to_sec()


    def odomProcess(self, odomData):
        self.currentX = odomData.pose.pose.position.x
        self.currentY = odomData.pose.pose.position.y
        #self.currentOrientation = odomData.pose.pose.orientation
        #self.currentEuler = euler_from_quaternion(self.currentOrientation)    #currentEuler is a list of 3 elements in the order: roll, pitch, yaw
        #self.currentYaw = self.currentEuler[2]   #currentYaw is the 3rd element of the list "currentEuler" so euler[2]
        #self.currentRoll, self.currentPitch, self.currentYaw = euler_from_quaternion(self.currentOrientation)
        self.currentYaw = euler_from_quaternion([odomData.pose.pose.orientation.x, odomData.pose.pose.orientation.y, odomData.pose.pose.orientation.z, odomData.pose.pose.orientation.w])[2]

    def KP_ANGProcess(self, KP_ANGData):
        self.KP_ANG = KP_ANGData.data

    def KI_ANGProcess(self, KI_ANGData):
        self.KI_ANG = KI_ANGData.data

    def KD_ANGProcess(self, KD_ANGData):
        self.KD_ANG = KD_ANGData.data

    def set_vel(self, forward, turn):
        self.vel.linear.x = forward
        self.vel.angular.z = turn
        self.vel_pub.publish(self.vel)
        self.vel_rate.sleep()

    def withinRadius(self):
        if ( (self.nextY - self.currentY)**2 + (self.nextX - self.currentX)**2 ) > ( self.WP_GOAL_RADIUS**2 ):
            return False
        else:
            return True

    def facingNextWP(self):
        #self.deltaYaw = math.atan((self.nextY - self.currentY) / (self.nextX - self.currentX))   #math.atan() returns an angle in radians between -PI/2 and PI/2
        self.deltaYaw = math.atan2( (self.nextY - self.currentY), (self.nextX - self.currentX) )  #math.atan2() returns an angle in radians between -PI and PI
        
        if ((self.deltaYaw - self.THRESHOLD_YAW_RADIANS) <= self.currentYaw) and (self.currentYaw <= (self.deltaYaw + self.THRESHOLD_YAW_RADIANS)):
            return True
        else:
            return False


    def atGoal(self):
        self.set_vel(0.0, 0.0)
        self.goalReached = True
        # while self.state is self.STATE_STOP:
        #     self.vel_pub.publish(self.vel)
        #     self.vel_rate.sleep()

    def getNextWP(self):
        self.nextX = self.NEXT_WP.position.x
        self.nextY = self.NEXT_WP.position.y
        #self.nextEuler = tf.transformations.euler_from_quaternion(NEXT_WP.pose.pose.orientation)
        #self.nextYaw = self.nextEuler[2]
        
        if self.withinRadius():
            self.state = self.STATE_GET_NEXT_WP
        else:
            self.state = self.STATE_TURN_TO_WP
            self.goalReached = False


    # def turnToWP(self):
    #     if self.facingNextWP():
    #         self.state = self.STATE_DRIVE_TO_WP
    #     else:
    #         if self.currentYaw < ( self.deltaYaw - self.THRESHOLD_YAW_RADIANS ):
    #             self.set_vel(0.0, self.ANGULAR_VEL_Z)
    #         elif self.currentYaw > ( self.deltaYaw + self.THRESHOLD_YAW_RADIANS ):
    #             self.set_vel(0.0, ( -1 * self.ANGULAR_VEL_Z ) )
    #         else:
    #             self.set_vel(0.0, 0.0)

    def turnToWP(self):
        if self.facingNextWP():
            self.state = self.STATE_DRIVE_TO_WP
        else:
            # time_current_ang = rospy.Time.now()
            #self.time_current_ang = rospy.get_rostime().to_sec()
            self.time_current_ang = rospy.get_time()
            # delta_time_ang = time_current_ang.to_sec() - time_last_ang.to_sec()
            self.delta_time_ang = self.time_current_ang - self.time_last_ang

            self.error_ang = self.deltaYaw - self.currentYaw
            self.integral_ang += self.error_ang * self.delta_time_ang
            self.derivative_ang = (self.error_ang - self.error_last_ang) / self.delta_time_ang

            self.output_ang = ( self.KP_ANG * self.error_ang ) + ( self.KI_ANG * self.integral_ang ) + ( self.KD_ANG * self.derivative_ang ) + self.BIAS_ANG

            if self.output_ang > self.MAX_ANGULAR_VEL_Z:
                self.set_vel(0.0, self.MAX_ANGULAR_VEL_Z)
            elif self.output_ang < (-1 * self.MAX_ANGULAR_VEL_Z):
                self.set_vel(0.0, (-1 * self.MAX_ANGULAR_VEL_Z))
            else:
                self.set_vel(0.0, self.output_ang)

            self.error_last_ang = self.error_ang

        self.time_last_ang = self.time_current_ang


    def driveToWP(self):
        if self.withinRadius():
            self.state = self.STATE_AT_GOAL
        # elif not(self.facingNextWP()):
        #     self.state = self.STATE_TURN_TO_WP
        else:
            self.set_vel(self.LINEAR_VEL_X, 0.0)



if __name__ == '__main__':
    try:
        #rospy.sleep(5)  #delay start of control program by 5 seconds

        cart = SmartCart()

        while not rospy.is_shutdown():  #run infinite loop 
            cart.state_pub.publish(cart.state)
            cart.state_rate.sleep()

            cart.deltaYaw_pub.publish( cart.deltaYaw * 180 / PI )
            cart.deltaYaw_rate.sleep()

            cart.goalReached_pub.publish(cart.goalReached)
            cart.goalReached_rate.sleep()

            if cart.state == cart.STATE_AT_GOAL:        #STATE_AT_GOAL = 0
                cart.atGoal()
            elif cart.state == cart.STATE_GET_NEXT_WP:  #STATE_GET_NEXT_WP = 1
                cart.getNextWP()
            elif cart.state == cart.STATE_TURN_TO_WP:   #STATE_TURN_TO_WP = 2
                cart.turnToWP()
            elif cart.state == cart.STATE_DRIVE_TO_WP:  #STATE_DRIVE_TO_WP = 3
                cart.driveToWP()
            else:
                cart.stop()

            #rospy.sleep(delay_time) #delay loop/program by "delay_time". This is needed or there is a long delay for cmd_vel to update...don't know why.

    except rospy.ROSInterruptException: pass
