//the following code is to setup an arduino node in ROS called /serial.node
//that subscribes to the cmd_vel topic and writes PWM signals to the motor drivers
//& publishes to a new topic "encoderCounts" the encoder counts from the left and right
//wheel encoders with quadrature. (Arduino Board: Arduino Uno R3)

//NOTES: This code does NOT use Interrupts.Ensure tires are properly inflated to minimize slippage 
//between tire and hub. Arduino Uno only has 2 external interrupt pins: digital pins 2 and 3. 
//Maybe can't use interrupts properly since I would need 4 pins: leftA, leftB, rightA, rightB
//?upgrade this code with ISR (Interrupt Service Routine)?

//?upgrade this code with ISR (Interrupt Service Routine)?

//parts of code from:
//https://playground.arduino.cc/Main/RotaryEncoders/ &
//https://howtomechatronics.com/tutorials/arduino/rotary-encoder-works-use-arduino/

//Created/Modified by:
//Author: Shymon Sumiyoshi
//Date: 2021-02-07

//encoder attachd to motor IG42 24VDC Geared Motor, item# TD-044-240 
//with gear head (1:24): rated torque 8kg-cm, rated speed 246rpm, rated current < 2.1A
//stall torque ?103.2kg-cm?, stall current 13A, no load current < 0.5A  

//encoder is 2 Channel Hall Effect Magnetic Encoder (incremental rotary encoder) 
//5cpr without quadrature & gear ratio / with quadrature & gear ratio, 480cpr (= 5x4x24)
//6 wires coming from geared motor:
//1) Black: -Motor
//2) Red: +Motor
//3) Brown: Hall Sensor Vcc (use 5VDC but can be anywhere between 3.5-20VDC)
//4) Green: Hall Sensor GND
//5) Blue: Hall Sensor A Vout   //need 1K ohm pull up resistor here, but I'll use the arduino's internal pull-up resistor
//6) Purple: Hall Sensor B Vout //need 1K ohm pull up resistor here, but I'll use the arduino's internal pull-up resistor


#include "Arduino.h"
#include <ros.h>
#include <geometry_msgs/Twist.h>  //for subscribing to messages from the "cmd_vel" topic
#include <std_msgs/Int16.h> //for publishing encoder count messages to the "encoders" topic


//motor constants
#define MULTIPLIER_STRAIGHT_LEFTWHEEL 20.00
#define MULTIPLIER_STRAIGHT_RIGHTWHEEL 23.50
//#define MULTIPLIER_TURN_RIGHTWHEEL 15.00
//#define MUTLIPLIER_TURN_LEFTWHEEL 12.00

//wheel_rad is the wheel radius in meters, wheel_sep is separation distance between the wheels in meters
//I measured wheel_rad=8"=0.2032m, wheel_sep=24.75"=0.62865m
#define WHEEL_RAD 0.2032
#define WHEEL_SEP 0.62865

//motor pins
#define EN_LEFT_PIN 8 //left motor forward enable pin; will output HIGH when left wheel is rotating forward and LOW when left wheel is rotating backward
#define PWM_LEFT_PIN 10 //left PWM pin
#define EN_RIGHT_PIN 9  //right motor forward enable pin; will output HIGH when right wheel is rotating forward and LOW when right wheel is rotating backward
#define PWM_RIGHT_PIN 11  //right PWM pin

//encoder pins
#define ENCODER_LEFT_A_PIN 2  //use blue wire from Left Encoder Hall Sensor A Vout
#define ENCODER_LEFT_B_PIN 3  //use purple wire from Left Encoder Hall Sensor B Vout
#define ENCODER_RIGHT_A_PIN 4 //use blue wire from Right Encoder Hall Sensor A Vout
#define ENCODER_RIGHT_B_PIN 5 //use purple wire from Right Encoder Hall Sensor B Vout


//set to HIGH for testing/debugging. set to LOW under normal operation to speed up program 
//(any serial communication really slows down the program) 
//bool serialEnable = LOW;  


//motor variables
//speed_left and speed_right are the speed values for the left and right wheels respectively
float speed_left = 0;
float speed_right=0;
//float lowSpeedLimitAbs = 50;    //?needed?
//float highSpeedLimitAbs = 150;  //?needed?
//declare variables for linear speed and angular speed and initialize to zero 
float speed_ang=0;
float speed_lin=0;  


//encoder variables
//note to self: could use "long" instead of "int" for lEncPos and rEncPos below; "long" stores a 32-bit or 4-byte value, 
//which yields -2,147,483,648 to 2,147,483,647 (-2^31 to (2^31 - 1)). For 480cpr, this is approx 4,473,924.3 revolutions = 2,147,483,647/480
int lEncPos = 0;  //note: "int" here stores a 16-bit or 2-byte value, which yields -32,768 to 32,767 (-2^15 to (2^15 - 1)). For 480cpr, this is approx 68.3 revolutions = 32,767/480
bool lEncAStateCurrent;
bool lEncAStateLast;
bool lEncBStateCurrent;
bool lEncBStateLast;

int rEncPos = 0;  //note: "int" here stores a 16-bit or 2-byte value, which yields -32768 to 32,767 (-2^15 to (2^15 - 1)). For 480cpr, this is approx 68.26 revolutions = 32,767/480
bool rEncAStateCurrent;
bool rEncAStateLast;
bool rEncBStateCurrent;
bool rEncBStateLast;



std_msgs::Int16 lCount_msg;
std_msgs::Int16 rCount_msg;

//create ROS node called "nh"; although in rqt_graph, it's listed as "/serial_node"
ros::NodeHandle nh; 

//create a publisher called "lCount_publisher" that publishes a "lCount_msg" message of type Int16 variable to the "/lCount" topic
ros::Publisher lCount_publisher("lCount", &lCount_msg); 
//create a publisher called "rCount_publisher" that publishes a "rCount_msg" message of type Int16 variable to the "rlCount" topic
ros::Publisher rCount_publisher("rCount", &rCount_msg); 

//function called by motor_subscriber
void process_speeds( const geometry_msgs::Twist& cmd_vel_msg ){
  speed_ang = cmd_vel_msg.angular.z;
  speed_lin = cmd_vel_msg.linear.x;
  speed_left = (speed_lin/WHEEL_RAD) - ((speed_ang*WHEEL_SEP)/(2.0*WHEEL_RAD));
  speed_right = (speed_lin/WHEEL_RAD) + ((speed_ang*WHEEL_SEP)/(2.0*WHEEL_RAD));
}
//create a subscriber called "motor_subscriber" for the node nh, which subscribes to the 
//"cmd_vel" topic and calls the function above "process_speeds" to process the Twist message 
//read on the "cmd_vel" topic 
ros::Subscriber<geometry_msgs::Twist> motor_subscriber( "cmd_vel", &process_speeds );



//arduino function declarations for motors
void motors_init();
//void motors(int speed_left, int speed_right);
void motorLeft(float speed_left);
void motorRight(float speed_right);

//arduino function declarations for encoders
/*void serialReset();  //for testing/debugging
void serialPrintBothPos(int lEncPos, int rEncPos);  //for testing/debugging */
void encoders_init();


//arduino function definitions for motors
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
     analogWrite(PWM_LEFT_PIN, speed_left*MULTIPLIER_TURN_
void serialReset();  //for testing/debugging
void serialPrintBothPos(int lEncPos, int rEncPos);  //for testing/debuggingLEFTWHEEL);
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

void motorLeft(float speed_left){
 if (speed_left >= 0){
     digitalWrite(EN_LEFT_PIN, HIGH);
     analogWrite(PWM_LEFT_PIN, speed_left);
 }
 if (speed_left < 0){
     digitalWrite(EN_LEFT_PIN, LOW);
     analogWrite(PWM_LEFT_PIN, abs(speed_left));
 }
}

void motorRight(float speed_right){
 if (speed_right >= 0){
     digitalWrite(EN_RIGHT_PIN, HIGH);
     analogWrite(PWM_RIGHT_PIN, speed_right);
 }
 if (speed_right < 0){
     digitalWrite(EN_RIGHT_PIN, LOW);
     analogWrite(PWM_RIGHT_PIN, abs(speed_right));
 }
}
//create a publisher called "lCount_publisher" that publishes a "lCount_msg" message of type Int16 variable to the "/lCount" topic

//arduino function definitions for encoders
/*void serialReset(){//create a publisher called "lCount_publisher" that publishes a "lCount_msg" message of type Int16 variable to the "/lCount" topic
  if(Serial.read() == 'r') {
    lEncPos = 0;
    rEncPos = 0;
    Serial.println("Left and Right Positions Reset to Zero");
  }
}

//***NOTE TO SELF: minimize using Serial.print() since it takes many compute cycles & slows down everything
void serialPrintBothPos(int lEncPos, int rEncPos){
  Serial.print("Left: ");
  Serial.print(lEncPos);
  Serial.print(",   Right: ");
  Serial.println(rEncPos);
}*/

void encoders_init(){
  pinMode(ENCODER_LEFT_A_PIN, INPUT_PULLUP);  //set pin as input and turn on pull-up resistor (Signal needs pull-up resistor)
  pinMode(ENCODER_LEFT_B_PIN, INPUT_PULLUP);  //set pin as input and turn on pull-up resistor (Signal needs pull-up resistor)
  pinMode(ENCODER_RIGHT_A_PIN, INPUT_PULLUP); //set pin as input and turn on pull-up resistor (Signal needs pull-up resistor)
  pinMode(ENCODER_RIGHT_B_PIN, INPUT_PULLUP); //set pin as input and turn on pull-up resistor (Signal needs pull-up resistor)
/*
  digitalWrite(ENCODER_LEFT_A_PIN, HIGH); //turn on pull-up resistor (Signal needs pull-up resistor)
  digitalWrite(ENCODER_LEFT_B_PIN, HIGH); //turn on pull-up resistor (Signal needs pull-up resistor)
  digitalWrite(ENCODER_RIGHT_A_PIN, HIGH);  //turn on pull-up resistor (Signal needs pull-up resistor)
  digitalWrite(ENCODER_RIGHT_B_PIN, HIGH);  //turn on pull-up resistor (Signal needs pull-up resistor)
*/
  lEncAStateLast = digitalRead(ENCODER_LEFT_A_PIN);
  lEncBStateLast = digitalRead(ENCODER_LEFT_B_PIN);
  rEncAStateLast = digitalRead(ENCODER_RIGHT_A_PIN);
  rEncBStateLast = digitalRead(ENCODER_RIGHT_B_PIN);
}


void setup(){
  motors_init();  //initialize the needed motor pins
  
  nh.initNode();  //initialize node nh created way above
  nh.advertise(lCount_publisher);
  nh.advertise(rCount_publisher);
  nh.subscribe(motor_subscriber); //get the node nh to start subscribing to the "cmd_vel" topic
  
  encoders_init();  //initialize the needed encoder pins and variables
  
/*  if(serialEnable == HIGH){
    Serial.begin(9600);
  }*/
}


void loop(){
/*  if(serialEnable == HIGH){
    if (Serial.available() > 0) {
      serialReset();  
    }
  }*/
  
  
  //LEFT wheel
  lEncAStateCurrent = digitalRead(ENCODER_LEFT_A_PIN);
  lEncBStateCurrent = digitalRead(ENCODER_LEFT_B_PIN);

  //check if there has been a HIGH-->LOW or LOW-->HIGH transition of LEFT wheel A signal
  if (lEncAStateCurrent != lEncAStateLast) {
    //compare LEFT wheel A and B signals; if they are DIFFERENT then LEFT wheel has moved FORWARD (CCW looking at left wheel shaft)
    if (lEncAStateCurrent != lEncBStateCurrent) {
      lEncPos++;
    } 
    else {
      lEncPos--;
    }
    lCount_msg.data = lEncPos;
    lCount_publisher.publish(&lCount_msg);
    
    /*if(serialEnable == HIGH){
      //only print left wheel encoder count if it is a multiple of 120 (1 wheel revolution should be 480)
      //(I want to check how much the Serial.print() function slows down the program)
      if(lEncPos%120 == 0){serialPrintBothPos(lEncPos, rEncPos);} 
      
      //serialPrintBothPos(lEncPos, rEncPos);
    }*/
  }  
  lEncAStateLast = lEncAStateCurrent;

  //check if there has been a HIGH-->LOW or LOW-->HIGH transition of LEFT wheel B signal
  if (lEncBStateCurrent != lEncBStateLast) {
    //compare LEFT wheel A and B signals; if they are the SAME then LEFT wheel has moved FORWARD (CCW looking at left wheel shaft)
    if (lEncAStateCurrent == lEncBStateCurrent) {
      lEncPos++;
    } 
    else {
      lEncPos--;
    }
    lCount_msg.data = lEncPos;
    lCount_publisher.publish(&lCount_msg);

    /*if(serialEnable == HIGH){
      //only print left wheel encoder count if it is a multiple of 120 (1 wheel revolution should be 480)
      //(I want to check how much the Serial.print() function slows down the program)
      if(lEncPos%120 == 0){serialPrintBothPos(lEncPos, rEncPos);} 
      
      //serialPrintBothPos(lEncPos, rEncPos);
    }*/
  }
  lEncBStateLast = lEncBStateCurrent;



  //RIGHT wheel
  rEncAStateCurrent = digitalRead(ENCODER_RIGHT_A_PIN);
  rEncBStateCurrent = digitalRead(ENCODER_RIGHT_B_PIN);
  
  //check if there has been a HIGH-->LOW or LOW-->HIGH transition of RIGHT wheel A signal
  if (rEncAStateCurrent != rEncAStateLast) {
    //compare RIGHT wheel A and B signals; if they are DIFFERENT then RIGHT wheel has moved BACKWARD (CCW looking at left wheel shaft)
    if (rEncAStateCurrent != rEncBStateCurrent) {
      rEncPos--;
    } 
    else {
      rEncPos++;
    }
    rCount_msg.data = rEncPos;
    rCount_publisher.publish(&rCount_msg);
    
    /*if(serialEnable == HIGH){
      //only print right wheel encoder count if it is a multiple of 120 (1 wheel revolution should be 480)
      //(I want to check how much the Serial.print() function slows down the program)
      if(rEncPos%120 == 0){serialPrintBothPos(lEncPos, rEncPos);} 
      
      //serialPrintBothPos(lEncPos, rEncPos);
    }*/

  }  
  rEncAStateLast = rEncAStateCurrent;

  //check if there has been a HIGH-->LOW or LOW-->HIGH transition of RIGHT wheel B signal
  if (rEncBStateCurrent != rEncBStateLast) {
    //compare RIGHT wheel A and B signals; if they are the SAME then RIGHT wheel has moved BACKWARD (CCW looking at left wheel shaft)
    if (rEncAStateCurrent == rEncBStateCurrent) {
      rEncPos--;
    } 
    else {
      rEncPos++;
    }
    rCount_msg.data = rEncPos;
    rCount_publisher.publish(&rCount_msg);

    /*if(serialEnable == HIGH){
      //only print right wheel encoder count if it is a multiple of 120 (1 wheel revolution should be 480)
      //(I want to check how much the Serial.print() function slows down the program)
      if(rEncPos%120 == 0){serialPrintBothPos(lEncPos, rEncPos);} 
      
      //serialPrintBothPos(lEncPos, rEncPos);
    }*/
  }
  rEncBStateLast = rEncBStateCurrent;

  
  //motors(speed_left, speed_right);
  motorLeft(speed_left*MULTIPLIER_STRAIGHT_LEFTWHEEL);
  motorRight(speed_right*MULTIPLIER_STRAIGHT_RIGHTWHEEL);

  nh.spinOnce(); //?needed?
}
