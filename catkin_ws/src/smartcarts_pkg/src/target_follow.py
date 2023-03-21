#!/usr/bin/env python
from math import pow, atan2, sqrt, pi, asin, cos
import numpy as np
from tf.transformations import euler_from_quaternion

import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from std_msgs.msg import Bool, Int32MultiArray, Float32
from nav_msgs.msg import Odometry

DELAY_TIME = 0.005

THRESHOLD_YAW_RADIANS = (1.5)*pi/180    #when turning to goal, this is the tolerance +/- in radians 
DISTANCE_TOLERANCE = 0.01   #robot will travel to this value in metres around the goal position

MAX_LINEAR_VEL_X = 0.25     #maximum linear x velocity to use in m/s
MAX_ANGULAR_VEL_Z = 0.1    #maximum angular z velocity to use in rad/s
MIN_ANGULAR_VEL_Z = 0.00    #minimum angular z velocity to use in rad/s

VEL_PUBLISH_RATE = 20    #5Hz velocity message publish rate
LED_PUBLISH_RATE = 3    #3Hz LED message publish rate
TEST_PUBLISH_RATE = 1    #1Hz test messages publish rate

HORIZONTAL_FOV = 86 # (degrees) for D435, src: https://www.intel.com/content/www/us/en/support/articles/000030385/emerging-technologies/intel-realsense-technology.html
BALL_RADIUS = 0.096 #m (actual ball radius measured by meter rule)

QUEUE_SIZE = 10

KP_ANG = 0.8

MAX_WAYPOINTS = 3 # max number of waypoints to save

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

        self.target_pos = Int32MultiArray(data=[-1,-1])
        self.target_dist = float(-1.0)
        self.target_radius = float(-1.0)

        self.camera_width = 1280
        self.ball_radius = 0.1

        self.waypoints = []

        rospy.init_node('preset_path', anonymous=True)

        #NEEDED PUBLISHERS & SUBSCRIBERS
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size = QUEUE_SIZE)
        self.vel_rate = rospy.Rate(VEL_PUBLISH_RATE)

        self.LED_pub = rospy.Publisher('LEDsignal', Bool, queue_size = QUEUE_SIZE)
        self.LED_rate = rospy.Rate(LED_PUBLISH_RATE)
        
        #subscriber that subscribes to the "Odom" topic and calls the function "odomProcess"
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odomProcess)

        self.target_pos_sub = rospy.Subscriber('target_position', Int32MultiArray, self.target_pos_process)
        self.target_dist_sub = rospy.Subscriber('target_distance', Float32, self.target_dist_process)
        self.target_radius_sub = rospy.Subscriber('ball_radius', Float32, self.target_radius_process)
        self.last_waypoint_time = rospy.Time.now()

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
        self.current_pose.position.x = odomData.pose.pose.position.x + 0.375
        self.current_pose.position.y = odomData.pose.pose.position.y
        self.currentYaw = euler_from_quaternion([odomData.pose.pose.orientation.x, odomData.pose.pose.orientation.y, odomData.pose.pose.orientation.z, odomData.pose.pose.orientation.w])[2]

    def target_pos_process(self, posData):
        self.target_pos = posData.data

    def target_dist_process(self, distData):
        self.target_dist = distData.data / 1000.0

    def target_radius_process(self, radData):
        self.target_radius = radData.data

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
        
    def get_next_waypoint(self):
        #Process the next pose using target_pos, target_dist
        print(self.target_dist) # in mm
        if self.target_dist < 0.01 or self.target_dist > 10 or np.isnan(self.target_dist):
            target_pose = Pose(position = Point(x = -1, y = -1, z = -1), orientation = Quaternion(w=1))
        else:
            pixel_scale = self.target_radius/self.ball_radius # Pixels/unit Length
            y_dist_from_center = (self.camera_width/2 - self.target_pos[0]) / pixel_scale # Unit length
            if abs(y_dist_from_center/self.target_dist) > 1 :
                print("MATH ERROR: y_dist_from_center={}, dist_data={}".format(y_dist_from_center, self.target_dist))
                print("time = {}".format(rospy.Time.now()))
            else:
                theta = asin(y_dist_from_center / self.target_dist) # Radians
                x_dist_from_camera = self.target_dist * cos(theta) # Unit length, rel to Follower ref frame
                y_dist_from_camera = y_dist_from_center # Unit length, rel to Follower ref frame
                #target_pose = Pose(position = Point(x = x_dist_from_camera, y =  y_dist_from_camera, z = 0), orientation = Quaternion(w=1.0 ))
                print("x: ", x_dist_from_camera, " y: ", y_dist_from_camera)
                target_pose = Pose(position = Point(x = self.current_pose.position.x + x_dist_from_camera, y = self.current_pose.position.y + y_dist_from_camera, z = 0), orientation = Quaternion(w=1.0 ))
        
        self.waypoints.append(target_pose)

    '''Gets the next waypoint from target_dist (subsribed package)'''
    def get_next_waypoint_scaled(self):
        print("Getting next Waypoint..... \n Target distance: {} mm".format(self.target_dist))

        if self.target_dist < 0.01 or self.target_dist > 10 or np.isnan(self.target_dist):
            # If target_dist is a "garbage value", return (-1,-1,-1)
            target_pose = Pose(position = Point(x = -1, y = -1, z = -1), orientation = Quaternion(w=1))
        else:
            # Valid Target Distance Value
            y_disp = self.get_horizontal_displacement()

            if abs(y_disp/self.target_dist) > 1 :
                print("MATH ERROR: y_disp={}, dist_data={}".format(y_disp, self.target_dist))
                print("time = {}".format(rospy.Time.now()))
            else:
                theta = asin(y_disp / self.target_dist) # Radians
                x_dist_from_camera = self.target_dist * cos(theta) # Unit length, rel to Follower ref frame
                y_dist_from_camera = y_disp # Unit length, rel to Follower ref frame
                #target_pose = Pose(position = Point(x = x_dist_from_camera, y =  y_dist_from_camera, z = 0), orientation = Quaternion(w=1.0 ))
                print("x: ", x_dist_from_camera, " y: ", y_dist_from_camera)
                target_pose = Pose(position = Point(x = self.current_pose.position.x + x_dist_from_camera, y = self.current_pose.position.y + y_dist_from_camera, z = 0), orientation = Quaternion(w=1.0 ))
        
        self.waypoints.append(target_pose)


    '''Gets the y displacement from the center using graph tested
    src: 
    '''
    def get_horizontal_displacement(self):
        ball_diam_pixels = -120.54 * self.target_dist + 338.4
        pixel_scale = ball_diam_pixels / BALL_RADIUS

        return (self.camera_width/2 - self.target_pos[0]) / pixel_scale # returns in m
        


    #Goal Setting/Getting Functions
    #STATE 0
    def atGoal(self):
        self.set_vel(0.0, 0.0)
        self.set_LED(1)
        if len(self.waypoints) != 0:
            self.waypoints.pop(0)
        print("Goal Reached!")
        print("")
        print("")

        self.state = STATE_GET_NEXT_GOAL
        print("Current state is: 1 (STATE_GET_NEXT_GOAL)")

    #STATE 1
    def getNextGoal(self):
        if len(self.waypoints) == 0:
            #Wait for next waypoint
            raw_input("Press Enter to register next waypoint")
            self.get_next_waypoint_scaled()
            print("Got Next waypoint : ", self.waypoints[0])

        self.goal_pose.position.x = self.waypoints[0].position.x
        self.goal_pose.position.y = self.waypoints[0].position.y
        self.goalIndex += 1
        rospy.sleep(1.0)
        print("CURRENT : ", self.current_pose)
        print("GOAL : ", self.goal_pose)

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
            # Append waypoints if there are fewer than the buffer and more than 5 seconds have passed
            if len(cart.waypoints) < MAX_WAYPOINTS and (rospy.Time.now() - cart.last_waypoint_time).to_sec() > 5:
                cart.get_next_waypoint_scaled()

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
