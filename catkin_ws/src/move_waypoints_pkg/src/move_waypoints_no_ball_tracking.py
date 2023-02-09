#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
import tf
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from std_msgs.msg import Bool, Int8, Int16, Float32, Float64
from math import pow, atan2, sqrt


PI = 3.1415926535

# to move CCW around rectangle (office 3m x 3m)
# WP0 = Pose(Point(1.5,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
# WP1 = Pose(Point(3.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
# WP2 = Pose(Point(3.0,3.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
# WP3 = Pose(Point(1.5,3.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
# WP4 = Pose(Point(0.0,3.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
# WP5 = Pose(Point(0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))

# to move straight then 30 degrees to left (office)
# WP0 = Pose(Point(1.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
# WP1 = Pose(Point(2.7321,1.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
# WP2 = Pose(Point(2.7321,3.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
# WP3 = Pose(Point(1.5,3.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
# WP4 = Pose(Point(0.0,3.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
# WP5 = Pose(Point(0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))

# to move straight then 45 degrees to left (office)
WP0 = Pose(Point(1.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP1 = Pose(Point(2.0,1.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP2 = Pose(Point(2.0,3.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP3 = Pose(Point(1.5,3.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP4 = Pose(Point(0.0,3.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
WP5 = Pose(Point(0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))


# to move CCW around rectangle
# WP0 = Pose(Point(1.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
# WP1 = Pose(Point(2.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
# WP2 = Pose(Point(2.0,1.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
# WP3 = Pose(Point(1.0,1.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
# WP4 = Pose(Point(0.0,1.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
# WP5 = Pose(Point(0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))

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

VEL_PUBLISH_RATE = 20    #5Hz velocity message publish rate
LED_PUBLISH_RATE = 3    #3Hz LED message publish rate
TEST_PUBLISH_RATE = 1    #1Hz test messages publish rate
QUEUE_SIZE = 10

Kp_ANG = 0.8    #was 0.5

#All SmartCart STATES:
STATE_AT_GOAL = 0
STATE_GET_NEXT_GOAL = 1
STATE_TURN_TO_GOAL = 2
STATE_DRIVE_TO_GOAL = 3


class SmartCart:
    def __init__(self):
        self.goal_pose = Pose(Point(0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
        self.current_pose = Pose(Point(0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))
        self.starting_pose = Pose(Point(0.0,0.0,0.0), Quaternion(0.0,0.0,0.0,1.0))

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

        self.state = STATE_AT_GOAL #Set state so that Initially, we get next goal from user
        print("SmartCart Initialized")
        # rospy.sleep(1.0)


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

        self.state = STATE_GET_NEXT_GOAL
        print("current state is: 1 (STATE_GET_NEXT_GOAL)")

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


#1) GETTING NEXT GOAL FROM USER
    def getNextGoal(self):
        # to command robot to move to a position (x,y) inputted from the user
        # self.goal_pose.position.x = float(input("Set your x goal in metres: "))
        # self.goal_pose.position.y = float(input("Set your y goal in metres: "))


        # to command robot to move along the perimeter of a 2m x 1m rectangle
        if(self.goalIndex == 0):
            self.goal_pose.position.x = WP0.position.x
            self.goal_pose.position.y = WP0.position.y
            self.goalIndex += 1
            rospy.sleep(1.0)
        elif(self.goalIndex == 1):
            self.goal_pose.position.x = WP1.position.x
            self.goal_pose.position.y = WP1.position.y
            self.goalIndex += 1
            rospy.sleep(1.0)
        elif(self.goalIndex == 2):
            self.goal_pose.position.x = WP2.position.x
            self.goal_pose.position.y = WP2.position.y
            self.goalIndex += 1
            rospy.sleep(1.0)
        elif(self.goalIndex == 3):
            self.goal_pose.position.x = WP3.position.x
            self.goal_pose.position.y = WP3.position.y
            self.goalIndex += 1
            rospy.sleep(1.0)
        elif(self.goalIndex == 4):
            self.goal_pose.position.x = WP4.position.x
            self.goal_pose.position.y = WP4.position.y
            self.goalIndex += 1
            rospy.sleep(1.0)
        elif(self.goalIndex == 5):
            self.goal_pose.position.x = WP5.position.x
            self.goal_pose.position.y = WP5.position.y
            self.goalIndex = 0
            rospy.sleep(1.0)


        if self.euclidean_distance() > DISTANCE_TOLERANCE:
            self.startingDistance = self.euclidean_distance()
            
            # self.starting_pose = self.current_pose    #<-- don't use this; it doesn't work. It's almost as if this line makes starting_pose the same as current_pose (i.e.: whenever 
            # current_pose is updated, starting_pose is also updated with current_pose's data...weird...investigate further when you have time)
            self.starting_pose.position.x = self.current_pose.position.x
            self.starting_pose.position.y = self.current_pose.position.y

            self.set_LED(0)
            self.state = STATE_TURN_TO_GOAL
            print("current state is: 2 (STATE_TURN_TO_GOAL)")
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


#2) TURNING TO GOAL
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
            print("current state is: 3 (STATE_DRIVE_TO_GOAL)")

    def get_deltaYaw(self):
        # atan2(y, x) returns value of atan(y/x) in radians. The atan2() method returns a numeric value between -pi and pi representing the angle theta of a (x, y) point and positive x-axis.
        self.deltaYaw_unfiltered = atan2(self.goal_pose.position.y - self.current_pose.position.y, self.goal_pose.position.x - self.current_pose.position.x) - self.currentYaw

        if self.deltaYaw_unfiltered > PI:
            return (self.deltaYaw_unfiltered - 2*PI)
        elif self.deltaYaw_unfiltered < (-1*PI):
            return (2*PI + self.deltaYaw_unfiltered)
        else:
            return self.deltaYaw_unfiltered


#3) DRIVING TO GOAL
    def driveToGoal(self):
        if (self.euclidean_distance() > DISTANCE_TOLERANCE) and (self.distance_travelled() < self.startingDistance):
            self.set_vel(MAX_LINEAR_VEL_X, (Kp_ANG * self.deltaYaw) )
            # print("distance travelled is:", self.distance_travelled())
        else:
            self.state = STATE_AT_GOAL
            print("current state is: 0 (STATE_AT_GOAL)")
        

if __name__ == '__main__':
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
            else:
                print("ERROR: NOT in any state")
                cart.set_vel(0.0, 0.0)

    except rospy.ROSInterruptException: pass
