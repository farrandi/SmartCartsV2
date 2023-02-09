//WITH PID for wheel velocity closed-loop control

//the following code is to setup an arduino node in ROS called /serial.node
//that subscribes to the cmd_vel topic and writes PWM signals to the motor drivers
//& publishes to new topics "lCount" and "rCount" the encoder counts from the left and right
//wheel encoders with quadrature, respectively.
//Arduino Board: Arduino Mega 2560 Rev3

//NOTES: Ensure tires are properly inflated to minimize slippage between tire and hub

//parts of code from:
//https://playground.arduino.cc/Main/RotaryEncoders/ &
//https://howtomechatronics.com/tutorials/arduino/rotary-encoder-works-use-arduino/

//Created/Modified by:
//Author: Shymon Sumiyoshi
//Date Created: 2020-12-21
//Date Modified: 2021-02-20

//MOTOR: IG42 24VDC Geared Motor, item# TD-044-240 
//with gear head (1:24): rated torque 8kg-cm, rated speed 246rpm, rated current < 2.1A
//stall torque ?103.2kg-cm?, stall current 13A, no load current < 0.5A  

//MOTOR DRIVER: Cytron MD10C
//for brushed DC motor, motor voltage ranges from 5V to 30V (power), 3.3V and 5V logic level input 
//maximum current up to 13A continuous and 30A peak (10 seconds)
//fully NMOS H-Bridge (no heat sink required), speed control PWM frequency up to 20KHz
//supports both locked-antiphase and sign-magnitude PWM operation
//6 connections for motor driver:
//Signal 1) Black: GND
//Signal 2) Red: PWM input for speed control
//Signal 3) Yellow: DIR direction control
//Power 4) POWER+ from battery
//Power 5) POWER- from battery
//Power 6) Motor Output A
//Power 7) Motor Output B

//ENCODER: (attached to motor) is 2 Channel Hall Effect Magnetic Encoder (incremental rotary encoder) 
//5cpr without quadrature & gear ratio / with quadrature & gear ratio, 480cpr (= 5x4x24)
//6 wires coming from geared motor:
//Power 1) Black: -Motor
//Power 2) Red: +Motor
//Signal 3) Brown: Hall Sensor Vcc (use 5VDC but can be anywhere between 3.5-20VDC)
//Signal 4) Green: Hall Sensor GND
//Signal 5) Blue: Hall Sensor A Vout   //need 1K ohm pull up resistor here, but I'll use the arduino's internal pull-up resistor
//Signal 6) Purple: Hall Sensor B Vout //need 1K ohm pull up resistor here, but I'll use the arduino's internal pull-up resistor


#include "Arduino.h"
#include <ros.h>
#include <ros/time.h>
#include <geometry_msgs/Twist.h>  //for subscribing to messages from the "cmd_vel" topic
#include <std_msgs/Int16.h> //for publishing encoder count messages to the "lCount" & "rCount" topics
#include <std_msgs/Float32.h>
//#include <std_msgs/Int32MultiArray>


//Following Constants are use for CLOSED LOOP control of left and right wheel speeds (WITH PID Control):
float tKp = 1.0; 
float tKi = 0.0;
float tKd = 0.0;

float Kp = 0.0; 
float Ki = 0.0;
float Kd = 0.0;

//PID constants for the left wheel speed:
float leftEncPos_Last = 0.0;
float leftBias = 0.0;
float leftTime_Current = 0.0;
float leftTime_Last = 0.0;
float left_dt = 0.0;
//float leftDesired = 0.0;
//float leftSpeedActual = 0.0;
float leftActual = 0.0;
float leftError = 0.0;
float leftError_Last =0.0;
float leftIntegral = 0.0;
float leftDerivative = 0.0;
float leftOutput = 0.0;


//MOTOR CONSTANTS
//Following Constants were used for OPEN LOOP control of left and right wheel speeds (NO PID Control):
//ROS program "teleop_twist_keyboard.py" default linear speed is 0.5m/s=19.685"/sec; wheel diameter is approx. 8.5" so 1 wheel revolution = 26.704"
//so (1sec/19.685")x(26.704"/rev)=1.357sec/rev 
//and this is what I'll use to estimate what the MULTIPLIER_STRAIGHT_LEFTWHEEL and MULTIPLIER_STRAIGHT_RIGHTWHEEL values below should be
#define MULTIPLIER_STRAIGHT_LEFTWHEEL 10.4  //to change the value in "speed_left" from rad/s to the corresponding PWM value (0-255) so that the left wheel rotates at 1.357sec/rev 
#define MULTIPLIER_STRAIGHT_RIGHTWHEEL 10.6 //to change the value in "speed_right" from rad/s to the corresponding PWM value (0-255) so that the right wheel rotates at 1.357sec/rev 
//#define MULTIPLIER_TURN_RIGHTWHEEL 7.50
//#define MUTLIPLIER_TURN_LEFTWHEEL 6.00

//#define _PI 3.14159

//#define LOW_SPEED_LIMIT 50 //PWM low speed limit (possible values: 0-255)
#define HIGH_SPEED_LIMIT 255 //PWM high speed limit (possible values: 0-255)  
//if HIGH_SPEED_LIMIT is a high value (e.g.: 255), ensure you use fuses or motor starter so possible motor stall current doesn't damage motor drivers

//wheel_rad is the wheel radius in meters, wheel_sep is separation distance between the wheels in meters
//I measured wheel_rad=4.25"=0.10795m~0.108m (wheel diameter is approx. 8.5"), wheel_sep=24.75"=0.62865m~0.629m
#define WHEEL_RAD 0.108
#define WHEEL_SEP 0.625 //0.625 after I measured again

//MOTOR PINS
//for PWM pins, use the function analogWrite(pin,value) where "value" is the duty cycle: 0-255
//for Arduino Mega, PWM Pins are 2-13,44-46 / pwm frequency is 490Hz (except for pins 4 and 13, which are 980Hz)
#define EN_LEFT_PIN 8 //left motor forward enable pin; will output HIGH when left wheel is rotating forward and LOW when left wheel is rotating backward
#define PWM_LEFT_PIN 10 //left PWM pin
#define EN_RIGHT_PIN 9  //right motor forward enable pin; will output HIGH when right wheel is rotating forward and LOW when right wheel is rotating backward
#define PWM_RIGHT_PIN 11  //right PWM pin

//ENCODER PINS
//will use external interrupts and the function attachInterrupt(interrupt,ISR,mode) where "interrupt" is the number of the interrupt, "ISR" is the function to call, "mode" is the mode  
//for Arduino Mega, the total6 external interrupts are:interrupt0 (pin2), interrupt1 (pin3), interrupt2 (pin21), interrupt3 (pin20), interrupt4 (pin19), interrupt5 (pin18)
#define ENCODER_LEFT_A_PIN 2  //use blue wire from Left Encoder Hall Sensor A Vout (interrupt 0 (pin2) of Arduino Mega)
#define ENCODER_LEFT_B_PIN 3  //use purple wire from Left Encoder Hall Sensor B Vout (interrupt 1 (pin3) of Arduino Mega)
#define ENCODER_RIGHT_A_PIN 21 //use blue wire from Right Encoder Hall Sensor A Vout (interrupt 2 (pin21) of Arduino Mega)
#define ENCODER_RIGHT_B_PIN 20 //use purple wire from Right Encoder Hall Sensor B Vout (interrupt 3 (pin20) of Arduino Mega)


//set to HIGH for testing/debugging. set to LOW under normal operation to speed up program 
//(any serial communication really slows down the program) 
//bool serialEnable = LOW;  


//MOTOR VARIABLES
float speed_ang=0;  //speed_ang is the yaw speed (rotational speed about the z-axis) of the robot about its center of rotation in rad/s
float speed_lin=0;  //speed_lin is the linear speed of the robot in m/s; it is along an axis that always points straight ahead of the robot and moves with the robot
float speed_left = 0; //speed_left is the rotational speed of the left wheel/motor in rad/s 
float speed_right=0;  //speed_right is the rotational speed of the right wheel/motor in rad/s
//fyi: the value "(speed_lin -/+ (speed_ang*WHEEL_SEP/2))" represents the linear speed of the center of the left/right wheel, respectively, in m/s


//ENCODER VARIABLES
//note to self: could use "long" instead of "int" for lEncPos and rEncPos below; "long" stores a 32-bit or 4-byte value, 
//which yields -2,147,483,648 to 2,147,483,647 (-2^31 to (2^31 - 1)). For 480cpr, this is approx 4,473,924.3 wheel revolutions = 2,147,483,647/480
//all variables used by interrupt ISR function should be global and volatile as per the website https://www.arduino.cc/reference/en/language/functions/external-interrupts/attachinterrupt/
volatile int lEncPos = 0;  //note: "int" here stores a 16-bit or 2-byte value, which yields -32,768 to 32,767 (-2^15 to (2^15 - 1)). For 480cpr, this is approx 68.3 wheel revolutions = 32,767/480
volatile bool lEncAStateCurrent;
volatile bool lEncAStateLast;
volatile bool lEncBStateCurrent;
volatile bool lEncBStateLast;

volatile int rEncPos = 0;  //note: "int" here stores a 16-bit or 2-byte value, which yields -32768 to 32,767 (-2^15 to (2^15 - 1)). For 480cpr, this is approx 68.26 revolutions = 32,767/480
volatile bool rEncAStateCurrent;
volatile bool rEncAStateLast;
volatile bool rEncBStateCurrent;
volatile bool rEncBStateLast;


//ROS VARIABLES & FUNCTIONS
std_msgs::Int16 lCount_msg;
std_msgs::Int16 rCount_msg;

std_msgs::Float32 leftTime_Current_msg;
std_msgs::Float32 leftOutput_msg;
std_msgs::Float32 leftSpeedActual_msg;

//create ROS node called "nh"; although in rqt_graph, it's listed as "/serial_node"
ros::NodeHandle nh; 

//create a publisher called "lCount_publisher", for the node nh, that publishes a "lCount_msg" message of type Int16 variable to the "lCount" topic
ros::Publisher lCount_publisher("lCount", &lCount_msg); 
//create a publisher called "rCount_publisher", for the node nh, that publishes a "rCount_msg" message of type Int16 variable to the "rlCount" topic
ros::Publisher rCount_publisher("rCount", &rCount_msg); 

ros::Publisher leftTime_Current_publisher("leftTime_Current", &leftTime_Current_msg);
ros::Publisher leftOutput_publisher("leftOutput", &leftOutput_msg);
ros::Publisher leftSpeedActual_publisher("leftSpeedActual", &leftSpeedActual_msg); 

//function called by motor_subscriber
void process_speeds( const geometry_msgs::Twist& cmd_vel_msg ){
  speed_ang = cmd_vel_msg.angular.z;  //speed_ang is the yaw speed (rotational speed about the z-axis) of the robot about its center of rotation in rad/s
  speed_lin = cmd_vel_msg.linear.x; //speed_lin is the linear speed of the robot in m/s; it is along an axis that always points straight ahead of the robot and moves with the robot
  speed_left = (2.0*speed_lin - (speed_ang*WHEEL_SEP)) / (2.0*WHEEL_RAD); //speed_left is the rotational speed of the left wheel/motor in rad/s
  speed_right = (2.0*speed_lin + (speed_ang*WHEEL_SEP)) / (2.0*WHEEL_RAD);  //speed_right is the rotational speed of the right wheel/motor in rad/s
  //fyi: the value "(speed_lin -/+ (speed_ang*WHEEL_SEP/2))" represents the linear speed of the center of the left/right wheel, respectively, in m/s
}
//create a subscriber called "motor_subscriber" for the node nh, which subscribes to the 
//"cmd_vel" topic and calls the function above "process_speeds" to process the Twist message 
//read on the "cmd_vel" topic
ros::Subscriber<geometry_msgs::Twist> motor_subscriber( "cmd_vel", &process_speeds );


void process_Kp( const std_msgs::Float32& Kp_msg ){
  Kp = Kp_msg.data;
}
void process_Ki( const std_msgs::Float32& Ki_msg ){
  Ki = Ki_msg.data;
}
void process_Kd( const std_msgs::Float32& Kd_msg ){
  Kd = Kd_msg.data;
}
ros::Subscriber<std_msgs::Float32> Kp_subscriber( "Kp", &process_Kp );
ros::Subscriber<std_msgs::Float32> Ki_subscriber( "Ki", &process_Ki );
ros::Subscriber<std_msgs::Float32> Kd_subscriber( "Kd", &process_Kd );



//arduino FUNCTION DECLARATIONS for MOTORS
void motors_init();
void motorLeft(float lspeed);
void motorRight(float rspeed);

//arduino FUNCTION DECLARATIONS for ENCODERS
//void serialReset();  //for testing/debugging
//void serialPrintBothPos(int lEncPos, int rEncPos);  //for testing/debugging
void encoders_init();
void lEncA_ISR();
void lEncB_ISR();
void rEncA_ISR();
void rEncB_ISR();


//arduino FUNCTION DEFINITIONS for MOTORS
void motors_init(){
  pinMode(EN_LEFT_PIN, OUTPUT);
  pinMode(PWM_LEFT_PIN, OUTPUT);
  pinMode(EN_RIGHT_PIN, OUTPUT);
  pinMode(PWM_RIGHT_PIN, OUTPUT);
  
  digitalWrite(EN_LEFT_PIN, LOW);
  digitalWrite(PWM_LEFT_PIN, LOW);
  digitalWrite(EN_RIGHT_PIN, LOW);
  digitalWrite(PWM_RIGHT_PIN, LOW);

  //leftTime_Last = nh.now().toSec();
  leftTime_Last = millis();
}

void motorLeft(float lspeed){ //lspeed is the PWM value (0-255) to give to the left motor driver unless its value is >= the HIGH_SPEED_LIMIT PWM value
  //leftTime_Current = nh.now().toSec();
  leftTime_Current = millis();
  left_dt = leftTime_Current - leftTime_Last;
  leftActual = ( MULTIPLIER_STRAIGHT_LEFTWHEEL * PI * (lEncPos - leftEncPos_Last) ) / (240 * left_dt );

  leftTime_Current_msg.data = leftTime_Current;
  //leftSpeedActual = ( WHEEL_RAD * PI * (lEncPos - leftEncPos_Last) ) / (240 * left_dt );
  leftSpeedActual_msg.data = WHEEL_RAD * leftActual / MULTIPLIER_STRAIGHT_LEFTWHEEL;

  leftError = lspeed - leftActual;
  leftIntegral += leftError * left_dt;
  leftDerivative = (leftError - leftError_Last) / left_dt;
  
  leftOutput = (tKp * leftError) + (tKi * leftIntegral) + (tKd * leftDerivative) + leftBias;
  leftOutput_msg.data = leftOutput;
  
  if (leftOutput >= 0){
    digitalWrite(EN_LEFT_PIN, HIGH);
    
    if(leftOutput < HIGH_SPEED_LIMIT){
      analogWrite(PWM_LEFT_PIN, leftOutput);
    }
    else{
      analogWrite(PWM_LEFT_PIN, HIGH_SPEED_LIMIT);
    }
  }
  else{
    digitalWrite(EN_LEFT_PIN, LOW);
  
    if(abs(leftOutput) < HIGH_SPEED_LIMIT){
      analogWrite(PWM_LEFT_PIN, abs(leftOutput));
    }
    else{
      analogWrite(PWM_LEFT_PIN, HIGH_SPEED_LIMIT);
    }
  }

  leftEncPos_Last = lEncPos;
  leftError_Last = leftError;
  leftTime_Last = leftTime_Current;
}

void motorRight(float rspeed){ //rspeed is the PWM value (0-255) to give to the right motor driver unless its value is >= the HIGH_SPEED_LIMIT PWM value
 if (rspeed >= 0){
   digitalWrite(EN_RIGHT_PIN, HIGH);

   if(rspeed < HIGH_SPEED_LIMIT){
    analogWrite(PWM_RIGHT_PIN, rspeed); 
   }
   else{
    analogWrite(PWM_RIGHT_PIN, HIGH_SPEED_LIMIT);
   }
 }
 else{
   digitalWrite(EN_RIGHT_PIN, LOW);

   if(abs(rspeed) < HIGH_SPEED_LIMIT){
    analogWrite(PWM_RIGHT_PIN, abs(rspeed));
   }
   else{
    analogWrite(PWM_RIGHT_PIN, HIGH_SPEED_LIMIT);
   }
 }
}


//arduino FUNCTION DEFINITIONS for ENCODERS
//***encoder interrupt function definitions at very bottom of code
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

  attachInterrupt(digitalPinToInterrupt(ENCODER_LEFT_A_PIN), lEncA_ISR, CHANGE); //setting up interrupt for left encoder A (interrupt 0 (pin 2))
  attachInterrupt(digitalPinToInterrupt(ENCODER_LEFT_B_PIN), lEncB_ISR, CHANGE); //setting up interrupt for left encoder B (interrupt 1 (pin 3))
  attachInterrupt(digitalPinToInterrupt(ENCODER_RIGHT_A_PIN), rEncA_ISR, CHANGE); //setting up interrupt for right encoder A (interrupt 2 (pin 21))
  attachInterrupt(digitalPinToInterrupt(ENCODER_RIGHT_B_PIN), rEncB_ISR, CHANGE); //setting up interrupt for right encoder B (interrupt 3 (pin 20))

  lEncAStateLast = digitalRead(ENCODER_LEFT_A_PIN);
  lEncBStateLast = digitalRead(ENCODER_LEFT_B_PIN);
  rEncAStateLast = digitalRead(ENCODER_RIGHT_A_PIN);
  rEncBStateLast = digitalRead(ENCODER_RIGHT_B_PIN);
}


void setup(){
  motors_init();  //initialize the needed motor pins
  
  nh.initNode();  //initialize node nh created way above
  
  nh.advertise(lCount_publisher); //publish the left encoder count to the topic "lCount"
  nh.advertise(rCount_publisher); //publish the right encoder count to the topic "rCount"

  nh.advertise(leftTime_Current_publisher);
  nh.advertise(leftOutput_publisher);
  nh.advertise(leftSpeedActual_publisher);
  
  nh.subscribe(motor_subscriber); //get the node nh to start subscribing to the "cmd_vel" topic
  nh.subscribe(Kp_subscriber);
  nh.subscribe(Ki_subscriber);
  nh.subscribe(Kd_subscriber);
  
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

  motorLeft(speed_left*MULTIPLIER_STRAIGHT_LEFTWHEEL);
  motorRight(speed_right*MULTIPLIER_STRAIGHT_RIGHTWHEEL);

  /*
  Serial.print("current time: ");
  Serial.print(leftTime_Current);
  Serial.print(",   right enc: ");
  Serial.println(rEncPos);
  delay(1000);
  */

  nh.spinOnce(); //check for incoming ROS messages
}



//Interrupt Service Routines (ISRs) for encoders:

//following ISR function called whenever there is a CHANGE: HIGH-->LOW or LOW-->HIGH transition of LEFT wheel encoder A signal
void lEncA_ISR(){
  lEncAStateCurrent = digitalRead(ENCODER_LEFT_A_PIN);
    
  //compare LEFT wheel A and B signals; if they are DIFFERENT then LEFT wheel has moved FORWARD (CCW looking at left wheel shaft)
  if (lEncAStateCurrent != lEncBStateCurrent) {
    lEncPos++;
  } 
  else {
    lEncPos--;
  }
  lCount_msg.data = lEncPos;
  lCount_publisher.publish(&lCount_msg);
      
  lEncAStateLast = lEncAStateCurrent;
}

//following ISR function called whenever there is a CHANGE: HIGH-->LOW or LOW-->HIGH transition of LEFT wheel encoder B signal
void lEncB_ISR(){
  lEncBStateCurrent = digitalRead(ENCODER_LEFT_B_PIN);
  
  //compare LEFT wheel A and B signals; if they are the SAME then LEFT wheel has moved FORWARD (CCW looking at left wheel shaft)
  if (lEncAStateCurrent == lEncBStateCurrent) {
    lEncPos++;
  } 
  else {
    lEncPos--;
  }
  //commenting out the 2 lines below because the serial connection at 57600 baud to my slow red laptop
  //with a publish buffer size of 512 bytes is too slow so I only publish on the A signal change, not the B signal change
  //NOTE TO SELF: try with faster laptop and faster serial connection baud rate
  //lCount_msg.data = lEncPos;
  //lCount_publisher.publish(&lCount_msg);
  
  lEncBStateLast = lEncBStateCurrent; 
}

//following ISR function called whenever there is a CHANGE: HIGH-->LOW or LOW-->HIGH transition of RIGHT wheel encoder A signal
void rEncA_ISR(){
  rEncAStateCurrent = digitalRead(ENCODER_RIGHT_A_PIN);

  //compare RIGHT wheel A and B signals; if they are DIFFERENT then RIGHT wheel has moved BACKWARD (CCW looking at left wheel shaft)
  if (rEncAStateCurrent != rEncBStateCurrent) {
    rEncPos--;
  } 
  else {
    rEncPos++;
  }
  rCount_msg.data = rEncPos;
  rCount_publisher.publish(&rCount_msg);

  rEncAStateLast = rEncAStateCurrent;
}

//following ISR function called whenever there is a CHANGE: HIGH-->LOW or LOW-->HIGH transition of RIGHT wheel encoder B signal
void rEncB_ISR(){
  rEncBStateCurrent = digitalRead(ENCODER_RIGHT_B_PIN);
  
  //compare RIGHT wheel A and B signals; if they are the SAME then RIGHT wheel has moved BACKWARD (CCW looking at left wheel shaft)
  if (rEncAStateCurrent == rEncBStateCurrent) {
    rEncPos--;
  } 
  else {
    rEncPos++;
  }
  //commenting out the 2 lines below because the serial connection at 57600 baud to my slow red laptop
  //with a publish buffer size of 512 bytes is too slow so I only publish on the A signal change, not the B signal change
  //NOTE TO SELF: try with faster laptop and faster serial connection baud rate
  //rCount_msg.data = rEncPos;
  //rCount_publisher.publish(&rCount_msg);

  rEncBStateLast = rEncBStateCurrent;
}
