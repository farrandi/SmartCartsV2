#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
import tf
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from std_msgs.msg import Bool, Int8, Int16, Int32, Float32, Float64, Int32MultiArray
from math import pow, atan2, sqrt, cos, sin, tan, pi


PI = 3.1415926535

# to move CCW around rectangle
WP0 = Pose(Point(1.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP1 = Pose(Point(2.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP2 = Pose(Point(2.0,1.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP3 = Pose(Point(1.0,1.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP4 = Pose(Point(0.0,1.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP5 = Pose(Point(0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))

# # to move CW around rectangle
# WP0 = Pose(Point(0.0,1.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
# WP1 = Pose(Point(1.0,1.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
# WP2 = Pose(Point(2.0,1.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
# WP3 = Pose(Point(2.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
# WP4 = Pose(Point(1.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
# WP5 = Pose(Point(0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))

DELAY_TIME = 0.005

THRESHOLD_YAW_RADIANS = (1.5)*PI/180    #when turning to goal, this is the tolerance +/- in radians 
DISTANCE_TOLERANCE = 0.01   #robot will travel to this value in metres around the goal position

MAX_LINEAR_VEL_X = 0.25     #maximum linear x velocity to use in m/s
MAX_ANGULAR_VEL_Z = 0.25    #maximum angular z velocity to use in rad/s
MIN_ANGULAR_VEL_Z = 0.00    #minimum angular z velocity to use in rad/s

VEL_PUBLISH_RATE = 20    #3Hz velocity message publish rate
LED_PUBLISH_RATE = 3    #3Hz LED message publish rate
TEST_PUBLISH_RATE = 1    #1Hz test messages publish rate
QUEUE_SIZE = 10

Kp_ANG = 0.8    #was 0.5

#All SmartCart STATES:
STATE_AT_GOAL = 0
STATE_KEEP_TARGET_CENTERED = 1
STATE_GET_NEXT_GOAL = 2
STATE_TURN_TO_GOAL = 3
STATE_DRIVE_TO_GOAL = 4


DISTANCE_LIMIT_FAR = 2000  #far distance limit in mm
#DISTANCE_LIMIT_NEAR = 500 #near distance limit in mm

#robot rotates toward ball (left or right) to keep the ball between "xPositionLimitLeft" and "xPositionLimitRight"
#i.e.: keep the ball in the robot camera's (horizontal x-axis) center of view. The robot doesn't care if the ball is vertically too high or low along the y-axis.
#origin of pixel x and y positions is in the TOP LEFT CORNER of the camera's screen/field of view
#left/right is the x-axis & up/down is the y-axis
#the robot screen/field of view is 1280x720 pixels (HD)
#approximately (x,y) = (640,360) is the center of the camera's field of view
XPOSITION_CENTER = 640 #this is the center pixel position; found using the display window that pops up when running the target tracking node
XPOSITION_THRESHOLD = 190
XPOSITION_LIMIT_RIGHT = XPOSITION_CENTER + XPOSITION_THRESHOLD  #left position limit of x-axis in pixels wrt camera's view
XPOSITION_LIMIT_LEFT = XPOSITION_CENTER - XPOSITION_THRESHOLD   #right position limit of x-axis in pixels wrt camera's view

CAMERA_XOFFSET = 0.0    #was 0.38
# TARGET_ANGLE_BASIC = 0
TARGET_ANGLE_BASIC = (20)*PI/180
# TARGET_ANGLE_BASIC = atan2(0.3, 0.6)    #for DISTANCE_LIMIT_FAR = 600mm = 0.6m, XPOSITION_THRESHOLD = 190 is approximately 30cm = 0.3m


class SmartCart:
    def __init__(self):
        self.goal_pose = Pose(Point(0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
        self.current_pose = Pose(Point(0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
        self.starting_pose = Pose(Point(0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
        self.next_pose = Pose(Point(0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))

        self.vel = Twist()
        self.vel.linear.x = 0.0
        self.vel.linear.y = 0.0
        self.vel.linear.z = 0.0
        self.vel.angular.x = 0.0
        self.vel.angular.y = 0.0
        self.vel.angular.z = 0.0

        # self.currentDistance = 0.0  #distance between current_pose and goal_pose; always updating
        self.startingDistance = 0.0 #distance between current_pose and goal_pose when the goal_pose was obtained; fixed until the next goal_pose is obtained
        # self.distanceTravelled = 0.0    #distance travelled by robot between starting_pose and current_pose; always updating
        # self.lastDistanceTravelled = 0.0    #previous distance travelled by robot between starting_pose and current_pose; always updating

        self.goalIndex = 0
        self.currentYaw = 0.0   #robot's current angle in radians relative to odom frame's positive x-axis; CCW is positive
        self.deltaYaw_unfiltered = 0.0
        self.deltaYaw = 0.0 #difference between goalYaw and currentYaw

        self.LED = Bool()
        self.LED.data = 0

        self.target_distance = 0.0
        self.target_xposition = 0.0
        self.target_angle = 0.0
        self.target_last = 0.0

        #create/start new node called "move_waypoints" unless overwritten in the launch file. "anonymous = True" ensures the node has a unique name by adding random numbers to the end of "move_waypoints"
        rospy.init_node('move_waypoints', anonymous=True)
        #rate = rospy.Rate(rate_value)   #creates a Rate object called "rate" so program loops at "rate_value" in Hz
        
        #NEEDED PUBLISHERS & SUBSCRIBERS
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size = QUEUE_SIZE)
        self.vel_rate = rospy.Rate(VEL_PUBLISH_RATE)

        self.LED_pub = rospy.Publisher('LEDsignal', Bool, queue_size = QUEUE_SIZE)
        self.LED_rate = rospy.Rate(LED_PUBLISH_RATE)
        
        #subscriber that subscribes to the "Odom" topic and calls the function "odomProcess"
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odomProcess)
        self.distance_sub = rospy.Subscriber('target_distance', Float32, self.distanceProcess) #subscriber that subscribes to the "target_distance" topic (published to by the "ball_tracking" node) and calls the function "distanceProcess"
        self.position_sub = rospy.Subscriber('target_position', Int32MultiArray, self.positionProcess) #subscriber that subscribes to the "target_position" topic (published to by the "ball_tracking" node) and calls the function "positionProcess"


        self.state = STATE_AT_GOAL #Set state so that Initially, we get next goal from user
        print("SmartCart Initialized")
        # rospy.sleep(1.0)


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
        # self.temp_distance_publisher.publish(self.target_distance) #publish the target distance (Float32) to a "temp_dist" topic for troubleshooting purposes

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
        # self.temp_xPosition_publisher.publish(self.target_xposition) #publish the target xPosition (Int32) to a "temp_xPos" topic for troubleshooting purposes


    def odomProcess(self, odomData):
        self.current_pose.position.x = odomData.pose.pose.position.x
        self.current_pose.position.y = odomData.pose.pose.position.y
        self.currentYaw = euler_from_quaternion([odomData.pose.pose.orientation.x, odomData.pose.pose.orientation.y, odomData.pose.pose.orientation.z, odomData.pose.pose.orientation.w])[2]


#0) GOAL HAS BEEN REACHED; STOPPING ROBOT
    def atGoal(self):
        self.set_vel(0.0, 0.0)
        self.set_LED(1)
        print("Goal Reached!")
        print("")
        print("")

        self.state = STATE_KEEP_TARGET_CENTERED
        print("current state is: 1 (STATE_KEEP_TARGET_CENTERED)")

        # self.state = STATE_GET_NEXT_GOAL
        # print("current state is: 2 (STATE_GET_NEXT_GOAL)")

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



#1) ROTATE ROBOT TO KEEP TARGET CENTERED IN ITS CAMERA'S FIELD OF VIEW
    def keepTargetCentered(self):
        
        if self.target_xposition < 0:  #if xPosition is -1 (no target detected by the "ball tracking" node) then command the robot to stop moving angularly 
            # self.set_vel(0.0, 0.0)
            if self.target_last == 1.0:
                self.set_vel(0.0, -1 * MAX_ANGULAR_VEL_Z)
            elif self.target_last == -1.0:
                self.set_vel(0.0, MAX_ANGULAR_VEL_Z)
            elif self.target_last == 0.0:
                self.set_vel(0.0, 0.0)
            else:
                self.set_vel(0.0, 0.0)
        elif self.target_xposition > XPOSITION_LIMIT_RIGHT:  #if target is too far to the right in the camera's field of view, command the robot to rotate CW (plan view)
            self.set_vel(0.0, (-1 * MAX_ANGULAR_VEL_Z))
            self.target_last = 1.0
        elif 0 <= self.target_xposition < XPOSITION_LIMIT_LEFT:  #if target is too far to the left in the camera's field of view, command the robot to rotate CCW (plan view)
            self.set_vel(0.0, MAX_ANGULAR_VEL_Z)
            self.target_last = -1.0
        else:   
            self.set_vel(0.0, 0.0)
            self.target_last = 0.0

        if self.target_distance > DISTANCE_LIMIT_FAR:
            if 0 <= self.target_xposition < XPOSITION_CENTER:
                self.target_angle = TARGET_ANGLE_BASIC
            elif self.target_xposition > XPOSITION_CENTER:
                self.target_angle = -1 * TARGET_ANGLE_BASIC
            else:
                self.target_angle = 0.0

            self.next_pose.position.x = self.current_pose.position.x + (CAMERA_XOFFSET * cos(self.currentYaw)) + ((self.target_distance/1000) * cos(self.currentYaw + self.target_angle))
            self.next_pose.position.y = self.current_pose.position.y + (CAMERA_XOFFSET * sin(self.currentYaw)) + ((self.target_distance/1000) * sin(self.currentYaw + self.target_angle))

            self.state = STATE_GET_NEXT_GOAL
            print("current state is: 2 (STATE_GET_NEXT_GOAL)")


#2) GETTING NEXT GOAL FROM USER
    def getNextGoal(self):
        # to command robot to move to a position (x,y) inputted from the user
        # self.goal_pose.position.x = float(input("Set your x goal in metres: "))
        # self.goal_pose.position.y = float(input("Set your y goal in metres: "))


        # to command robot to move along the perimeter of a 2m x 1m rectangle
        # if(self.goalIndex == 0):
        #     self.goal_pose.position.x = WP0.position.x
        #     self.goal_pose.position.y = WP0.position.y
        #     self.goalIndex += 1
        #     rospy.sleep(1.0)
        # elif(self.goalIndex == 1):
        #     self.goal_pose.position.x = WP1.position.x
        #     self.goal_pose.position.y = WP1.position.y
        #     self.goalIndex += 1
        #     rospy.sleep(1.0)
        # elif(self.goalIndex == 2):
        #     self.goal_pose.position.x = WP2.position.x
        #     self.goal_pose.position.y = WP2.position.y
        #     self.goalIndex += 1
        #     rospy.sleep(1.0)
        # elif(self.goalIndex == 3):
        #     self.goal_pose.position.x = WP3.position.x
        #     self.goal_pose.position.y = WP3.position.y
        #     self.goalIndex += 1
        #     rospy.sleep(1.0)
        # elif(self.goalIndex == 4):
        #     self.goal_pose.position.x = WP4.position.x
        #     self.goal_pose.position.y = WP4.position.y
        #     self.goalIndex += 1
        #     rospy.sleep(1.0)
        # elif(self.goalIndex == 5):
        #     self.goal_pose.position.x = WP5.position.x
        #     self.goal_pose.position.y = WP5.position.y
        #     self.goalIndex = 0
        #     rospy.sleep(1.0)


        self.goal_pose.position.x = self.next_pose.position.x
        self.goal_pose.position.y = self.next_pose.position.y


        if self.euclidean_distance() > DISTANCE_TOLERANCE:
            self.startingDistance = self.euclidean_distance()
            print(self.startingDistance)
            
            # self.starting_pose = self.current_pose    #<-- don't use this; it doesn't work. It's almost as if this line makes starting_pose the same as current_pose (i.e.: whenever 
            # current_pose is updated, starting_pose is also updated with current_pose's data...weird...investigate further when you have time)
            self.starting_pose.position.x = self.current_pose.position.x
            self.starting_pose.position.y = self.current_pose.position.y

            self.set_LED(0)
            self.state = STATE_TURN_TO_GOAL
            print("current state is: 3 (STATE_TURN_TO_GOAL)")
        else:
            print("goal position you input is too close to the current position")
            print("input a goal position that is greater than ", DISTANCE_TOLERANCE, "metres away from the current position")
            self.state = STATE_GET_NEXT_GOAL

    #returns distance still to travel from current_pose to goal_pose
    def euclidean_distance(self):
        return sqrt( pow((self.goal_pose.position.x - self.current_pose.position.x), 2) + pow((self.goal_pose.position.y - self.current_pose.position.y), 2) )

    #returns distance travelled from starting_pose to current_pose
    def distance_travelled(self):
        # print('self.starting_pose.position.x: ', self.starting_pose.position.x, '   &   self.starting_pose.position.y: ', self.starting_pose.position.y)
        # print('self.current_pose.position.x: ', self.current_pose.position.x, '   &   self.current_pose.position.y: ', self.currrent_pose.position.y)
        return sqrt( pow((self.starting_pose.position.x - self.current_pose.position.x), 2) + pow((self.starting_pose.position.y - self.current_pose.position.y), 2) )


#3) TURNING TO GOAL
    def turnToGoal(self):
        self.deltaYaw = self.get_deltaYaw()
        # print("deltaYaw is:", "{:.3f}".format((self.deltaYaw * 180 / PI)), " degrees   &   currentYaw is:", "{:.3f}".format((self.currentYaw * 180 / PI)), " degrees")

        if abs(self.deltaYaw) > THRESHOLD_YAW_RADIANS:
            self.set_vel(0.0, (Kp_ANG * self.deltaYaw) )
        else:
            self.set_vel(0.0, 0.0)
            print("NOW FACING goal_pose")
            rospy.sleep(1.0)

            self.state = STATE_DRIVE_TO_GOAL
            print("current state is: 4 (STATE_DRIVE_TO_GOAL)")

    def get_deltaYaw(self):
        # atan2(y, x) returns value of atan(y/x) in radians. The atan2() method returns a numeric value between -pi and pi representing the angle theta of a (x, y) point and positive x-axis.
        self.deltaYaw_unfiltered = atan2(self.goal_pose.position.y - self.current_pose.position.y, self.goal_pose.position.x - self.current_pose.position.x) - self.currentYaw

        if self.deltaYaw_unfiltered > PI:
            return (self.deltaYaw_unfiltered - 2*PI)
        elif self.deltaYaw_unfiltered < (-1*PI):
            return (2*PI + self.deltaYaw_unfiltered)
        else:
            return self.deltaYaw_unfiltered


#4) DRIVING TO GOAL
    def driveToGoal(self):
        if (self.euclidean_distance() > DISTANCE_TOLERANCE) and (self.distance_travelled() < self.startingDistance):
            self.set_vel(MAX_LINEAR_VEL_X, (Kp_ANG * self.deltaYaw) )
            # print("distance travelled is:", self.distance_travelled())
        else:
            self.state = STATE_AT_GOAL
            print("current state is: 0 (STATE_AT_GOAL)")

        # if self.target_xposition < 0:
        #     self.target_last = -2.0
        if self.target_xposition > XPOSITION_LIMIT_RIGHT:  #if target is too far to the right in the camera's field of view, command the robot to rotate CW (plan view)
            self.target_last = 1.0
        elif 0 <= self.target_xposition < XPOSITION_LIMIT_LEFT:  #if target is too far to the left in the camera's field of view, command the robot to rotate CCW (plan view)
            self.target_last = -1.0
        elif XPOSITION_LIMIT_LEFT <= self.target_xposition <= XPOSITION_LIMIT_RIGHT:
            self.target_last = 0.0

        

if __name__ == '__main__':
    try:
        cart = SmartCart()

        while not rospy.is_shutdown():  #run infinite loop 
            if cart.state == STATE_AT_GOAL:        #STATE_AT_GOAL = 0
                # print("current state is: 0 (STATE_AT_GOAL)")
                cart.atGoal()
            elif cart.state == STATE_KEEP_TARGET_CENTERED:
                # print("current state is: 1 (KEEP_TARGET_CENTERED)") #KEEP_TARGET_CENTERED = 1
                cart.keepTargetCentered()
            elif cart.state == STATE_GET_NEXT_GOAL:  #STATE_GET_NEXT_GOAL = 2
                # print("current state is: 2 (STATE_GET_NEXT_GOAL)")
                cart.getNextGoal()
            elif cart.state == STATE_TURN_TO_GOAL:   #STATE_TURN_TO_GOAL = 3
                # print("current state is: 3 (STATE_TURN_TO_GOAL)")
                cart.turnToGoal()
            elif cart.state == STATE_DRIVE_TO_GOAL:  #STATE_DRIVE_TO_GOAL = 4
                # print("current state is: 4 (STATE_DRIVE_TO_GOAL)")
                cart.driveToGoal()
            else:
                print("ERROR: NOT in any state")
                cart.set_vel(0.0, 0.0)

    except rospy.ROSInterruptException: pass
