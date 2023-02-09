//to do list: modify code to account for integer overflow of encoder counts, create tf node to account for camera position wrt to robot position, etc?

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Int16.h>
#include <std_msgs/Float64.h>

//wheel_rad is the wheel radius in meters, wheel_sep is separation distance between the wheels in meters
//I measured wheel_rad=4.25"=0.10795m~0.108m (wheel diameter is approx. 8.5"), wheel_sep=24.75"=0.62865m~0.629m
#define WHEEL_RAD 0.108
#define WHEEL_SEP 0.629

#define ENC_CPR 480 //with quadrature and the gear ratio, there are 480 cpr (counts per revolution)
#define PI 3.1415926535 

double distancePerCount = (2 * PI * WHEEL_RAD)/ENC_CPR;

double delta_time;

int lCount_current = 0;
int rCount_current = 0;
int lCount_last = 0;
int rCount_last = 0;

double delta_lDist;
double delta_rDist;
double delta_cDist;
double delta_th;

//x,y,th are all zero which implies whenever the robot starts up, it is located at the origin of 
//the odom coordinate frame (0,0) and the front of the robot is pointed along the positive x-axis
double x = 0;
double y = 0;
double th = 0;

double vx;
double vy;
double vth;


//function called whenever "lCount_sub" subscriber below subscribes to "lCount" topic
void process_lCount(const std_msgs::Int16::ConstPtr& lCount){
  lCount_current = lCount->data;
}

//function called whenever "rCount_sub" subscriber below subscribes to "rCount" topic
void process_rCount(const std_msgs::Int16::ConstPtr& rCount){
  rCount_current = rCount->data;
}


int main(int argc, char** argv){
  ros::init(argc, argv, "odom_enc_node");

  ros::NodeHandle n;  //create node variable n
  //create a publisher called "odom_enc_pub" that publishes a message of type "nav_msgs::Odometry" to the topic "odom" 
  //with a queue size of 50 (50 outgoing messages in the queue)
  ros::Publisher odom_enc_pub = n.advertise<nav_msgs::Odometry>("odom_enc_node", 50); 
  //create publisher called "yaw_pub" that publishes a message of type "std_msgs::float64" to the topic "yaw_enc"
  //with a queue size of 50
  ros::Publisher yaw_pub = n.advertise<std_msgs::Float64>("yaw_enc", 50);
  //create a transform broadcaster called "odom_enc_broadcaster"
  tf::TransformBroadcaster odom_enc_broadcaster;

  //create subscriber "lCount_sub" that subscribes to the "lCount" topic, has a queue size of 50 messages, 
  //and calls the function "process_lCount" whenever a new message arrives
  ros::Subscriber lCount_sub = n.subscribe("lCount", 50, process_lCount);
  
  //create subscriber "rCount_sub" that subscribes to the "rCount" topic, has a queue size of 50 messages, 
  //and calls the function "process_rCount" whenever a new message arrives
  ros::Subscriber rCount_sub = n.subscribe("rCount", 50, process_rCount);


  ros::Time current_time, last_time;
  current_time = ros::Time::now();  //needed here?
  last_time = ros::Time::now();

  ros::Rate r(5.0); //publish odometry info at 5.0Hz


  while(n.ok()){

    ros::spinOnce();               // check for incoming messages
    current_time = ros::Time::now();
    

    //an Odometry message consists of a "pose" message and a "twist" message. Will now get the left and right wheel encoder counts to 
    //compute x,y,odom_enc_quat values for the "pose" message, and vx,vy,vth values for the "twist" message
    //***pose is estimated position of robot in odometric frame (i.e.: fixed world frame)
    //***twist is robot's velocity in child frame, normally coordinate frame of mobile base (base_link)
    delta_lDist = (lCount_current - lCount_last) * distancePerCount; //delta_lDist is the incremental distance moved by the left wheel in metres
    delta_rDist = (rCount_current - rCount_last) * distancePerCount; //delta_rDist is the incremental distance moved by the right wheel in metres
    delta_cDist = (delta_rDist + delta_lDist) / 2;  //delta_cDist is the incremental distance moved by the robot in metres (center of rotation of the robot, i.e.: base_link)
    delta_th = (delta_rDist - delta_lDist) / WHEEL_SEP; //delta_th is the incremental angle moved by the robot about its center of rotation (z-axis) in radians

    //compute x,y,odom_enc_quat for the "pose" message
    th += delta_th; //add to th the incremental th angle moved by the robot in the ?base-link? coordinate frame
    x += delta_cDist * cos(th); //add to x the incremental x distance moved by the robot relative to the odom coordinate frame
    y += delta_cDist * sin(th); //add to y the incremental y distance moved by the robot relative to the odom coordinate frame
    geometry_msgs::Quaternion odom_enc_quat = tf::createQuaternionMsgFromYaw(th); //since all odometry is 6DOF we'll need a quaternion created from yaw

    //compute vx,vy,vth for the "twist" message
    delta_time = (current_time - last_time).toSec();
    vx = delta_cDist / delta_time;  //velocity of the robot directly ahead of it; vx always points directly ahead of the robot
    vy = 0; //vy always zero for a differential drive robot because it can't move directly left or right
    vth = delta_th / delta_time;  //angular velocity about the z-axis of the robot; z-axis moves with robot


    //publish the transform over tf
    geometry_msgs::TransformStamped odom_enc_trans;
    odom_enc_trans.header.stamp = current_time;
    odom_enc_trans.header.frame_id = "odom_enc";
    odom_enc_trans.child_frame_id = "base_link";  //the base_link frame is the child of the odom frame

    odom_enc_trans.transform.translation.x = x; //the base_link frame (the robot) is now x metres in the x direction relative to the odom frame
    odom_enc_trans.transform.translation.y = y; //the base_link frame (the robot) is now y metres in the y direction relative to the odom frame
    odom_enc_trans.transform.translation.z = 0.0; //the base_link frame (the robot) is now 0 metres in the z direction relative to the odom frame
    odom_enc_trans.transform.rotation = odom_enc_quat;  //the base_link frame (the robot) is now rotated th radians (expressed in a quaternion)

    odom_enc_broadcaster.sendTransform(odom_enc_trans); //send the transform


    //publish the odometry message over ROS
    nav_msgs::Odometry odom_enc;
    odom_enc.header.stamp = current_time;
    odom_enc.header.frame_id = "odom_enc";

    //set the pose message (i.e.: the positions)
    odom_enc.pose.pose.position.x = x;
    odom_enc.pose.pose.position.y = y;
    odom_enc.pose.pose.position.z = 0.0;
    odom_enc.pose.pose.orientation = odom_enc_quat;

    //set the twist message (i.e.: the velocities)
    odom_enc.child_frame_id = "base_link";
    odom_enc.twist.twist.linear.x = vx;
    odom_enc.twist.twist.linear.y = vy;
    odom_enc.twist.twist.angular.z = vth;

    odom_enc_pub.publish(odom_enc); //publish the odom message ("odom_enc" variable) on the "odom_enc" topic

    yaw_pub.publish(th*180/PI); //publish the yaw message in DEGREES ("th*180/PI" variable) to the "yaw_enc" topic

    last_time = current_time;
    lCount_last = lCount_current;
    rCount_last = rCount_current;

    r.sleep();
  }
}