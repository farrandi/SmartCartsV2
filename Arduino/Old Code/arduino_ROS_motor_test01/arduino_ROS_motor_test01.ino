//Original Code Authors:
//* Ahmed A. Radwan (author)
//* Maisa Jazba

//Modified by:
//Shymon Sumiyoshi

//#include <ArduinoHardware.h>
#include "Arduino.h"
#include <ros.h>
#include <geometry_msgs/Twist.h>

#define MULTIPLIER_STRAIGHT_LEFTWHEEL 20.00
#define MULTIPLIER_STRAIGHT_RIGHTWHEEL 23.50
#define MULTIPLIER_TURN_RIGHTWHEEL 15.00
#define MUTLIPLIER_TURN_LEFTWHEEL 12.00

#define EN_LEFT_PIN 8
#define PWM_LEFT_PIN 10

#define EN_RIGHT_PIN 9
#define PWM_RIGHT_PIN 11

//speed_left and speed_right are the speed values for the left and right wheels respectively
double speed_left = 0, speed_right=0;

//wheel_rad is the wheel radius in meters, wheel_sep is separation distance between the wheels in meters
//wheel_rad=8"=0.2032m, wheel_sep=24.75"=0.62865m
double wheel_rad = 0.2032, wheel_sep = 0.62865;

ros::NodeHandle nh; //create ROS node called "nh"
int lowSpeedLimitAbs = 50;
int highSpeedLimitAbs = 150;
double speed_ang=0, speed_lin=0;  //declare variables for linear speed and angular speed and initialize to zero 

//function called by motor_subscriber
void process_speeds( const geometry_msgs::Twist& cmd_vel_msg ){
  speed_ang = cmd_vel_msg.angular.z;
  speed_lin = cmd_vel_msg.linear.x;
  speed_left = (speed_lin/wheel_rad) - ((speed_ang*wheel_sep)/(2.0*wheel_rad));
  speed_right = (speed_lin/wheel_rad) + ((speed_ang*wheel_sep)/(2.0*wheel_rad));
}

ros::Subscriber<geometry_msgs::Twist> motor_subscriber( "cmd_vel", &process_speeds );

//arduino function declarations
void motors_init();
//void motors(int speed_left, int speed_right);
void motorLeft(int speed_left);
void motorRight(int speed_right);

//arduino function definitions
void motors_init(){
 pinMode(EN_LEFT_PIN, OUTPUT);
 pinMode(PWM_LEFT_PIN, OUTPUT);
 pinMode(EN_RIGHT_PIN, OUTPUT);
 pinMode(PWM_RIGHT_PIN, OUTPUT);
 
 digitalWrite(EN_LEFT_PIN, LOW);
 digitalWrite(PWM_LEFT_PIN, LOW);
 digitalWrite(EN_RIGHT_PIN, LOW);
 digitalWrite(PWM_RIGHT_PIN, LOW);
}

/*void motors(int speed_left, int speed_right){
 //robot forwards
 if (speed_left >= 0 && speed_right >= 0){
     digitalWrite(EN_LEFT_PIN, HIGH);
     analogWrite(PWM_LEFT_PIN, speed_left*MULTIPLIER_STRAIGHT_LEFTWHEEL);
     digitalWrite(EN_RIGHT_PIN, HIGH);
     analogWrite(PWM_RIGHT_PIN, speed_right*MULTIPLIER_STRAIGHT_RIGHTWHEEL);
 }
 //robot backwards
 if (speed_left < 0 && speed_right < 0){
     digitalWrite(EN_LEFT_PIN, LOW);
     analogWrite(PWM_LEFT_PIN, speed_left*MULTIPLIER_STRAIGHT_LEFTWHEEL);
     digitalWrite(EN_RIGHT_PIN, LOW);
     analogWrite(PWM_RIGHT_PIN, speed_right*MULTIPLIER_STRAIGHT_RIGHTWHEEL);
 }
 //robot turn left
 if (speed_left < 0 && speed_right > 0){
     digitalWrite(EN_LEFT_PIN, LOW);
     analogWrite(PWM_LEFT_PIN, speed_left*MULTIPLIER_TURN_LEFTWHEEL);
     digitalWrite(EN_RIGHT_PIN, HIGH);
     analogWrite(PWM_RIGHT_PIN, speed_right*MULTIPLIER_TURN_RIGHTWHEEL);
 }
//robot turn right
 if (speed_left > 0 && speed_right < 0){
     digitalWrite(EN_LEFT_PIN, HIGH);
     analogWrite(PWM_LEFT_PIN, speed_left*MULTIPLIER_TURN_LEFTWHEEL);
     digitalWrite(EN_RIGHT_PIN, LOW);
     analogWrite(PWM_RIGHT_PIN, speed_right*MULTIPLIER_TURN_RIGHTWHEEL);
 }
 //robot stop
 if (speed_left == 0 && speed_right == 0){
     digitalWrite(EN_LEFT_PIN, HIGH);
     analogWrite(PWM_LEFT_PIN, speed_left*MULTIPLIER_STRAIGHT_LEFTWHEEL);
     digitalWrite(EN_RIGHT_PIN, HIGH);
     analogWrite(PWM_RIGHT_PIN, speed_right*MULTIPLIER_STRAIGHT_RIGHTWHEEL);
 }
}*/

void motorLeft(int speed_left){
 if (speed_left > 0){
     digitalWrite(EN_LEFT_PIN, HIGH);
     analogWrite(PWM_LEFT_PIN, speed_left);
 }
 if (speed_left < 0){
     speed_left=abs(speed_left);
     digitalWrite(EN_LEFT_PIN, LOW);
     analogWrite(PWM_LEFT_PIN, speed_left);
 }
 if (speed_left == 0){
     digitalWrite(EN_LEFT_PIN, HIGH);
     analogWrite(PWM_LEFT_PIN, speed_left);
 }
}

void motorRight(int speed_right){
 if (speed_right > 0){
     digitalWrite(EN_RIGHT_PIN, HIGH);
     analogWrite(PWM_RIGHT_PIN, speed_right);
 }
 if (speed_right < 0){
     speed_right=abs(speed_right);
     digitalWrite(EN_RIGHT_PIN, LOW);
     analogWrite(PWM_RIGHT_PIN, speed_right);
 }
 if (speed_right == 0){
     digitalWrite(EN_RIGHT_PIN, HIGH);
     analogWrite(PWM_RIGHT_PIN, speed_right);
 }
}

void setup(){
 motors_init();
 nh.initNode();
 nh.subscribe(motor_subscriber);
}

void loop(){
 //motors(speed_left, speed_right);
 motorLeft(speed_left*MULTIPLIER_STRAIGHT_LEFTWHEEL);
 motorRight(speed_right*MULTIPLIER_STRAIGHT_RIGHTWHEEL);
 nh.spinOnce();
}
