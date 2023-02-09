//see "Hardware Setup" section in https://maker.pro/arduino/tutorial/how-to-use-arduino-with-robot-operating-system-ros
//this is a program to test the arduino board as a ROS node (publisher and subscriber)

#include <ros.h>  //include ROS library roslib
#include <std_msgs/String.h>
#include <std_msgs/UInt16.h>

#define BUTTON 8
#define LED 13

//create a node called "node_handle" in ROS; although in rqt_graph, it's listed as "/serial_node"
ros::NodeHandle node_handle;

std_msgs::String button_msg;
std_msgs::UInt16 led_msg;

//function called every time a message is received by the node subscriber, which subscribes to the "toggle_led" 
//topic. When this happens, it reads "led_msg" message of type UInt16 from the "toggle_led" topic and 
//then turns on or off an LED based on the value of led_msg
void subscriberCallback(const std_msgs::UInt16& led_msg) {
  if (led_msg.data  == 1) {
    digitalWrite(LED, HIGH); 
  } else {
    digitalWrite(LED, LOW);
  }
}

//create a publisher called "button_publisher" that publishes a message of type String (stored in the "button_msg" 
//variable) to the "/button_press" topic
ros::Publisher button_publisher("button_press", &button_msg); 

//create a subscriber called "led_subscriber" that subscribes a message of type UInt16 (stored in the "led_msg"
//variable passed to the function subscriberCallback) from the "/toggle_led" topic
ros::Subscriber<std_msgs::UInt16> led_subscriber("toggle_led", &subscriberCallback);

void setup()
{
  pinMode(LED, OUTPUT);
  pinMode(BUTTON, INPUT);
  
  node_handle.initNode(); //initialize the node "/node_handle"; although in rqt_graph, it's listed as "/serial_node"
  node_handle.advertise(button_publisher);  //get the node "/node_handle" to publish using "button_publisher" publisher
  node_handle.subscribe(led_subscriber);  //get the node "/node_handle" to start subscribing uing "led_subscriber" subscriber
}

void loop()
{ 
  //this if-else statement continually reads the di//create a publisher calledgital input where the pushbutton is connected and changes
  //the String "button_msg" accordingly
  if (digitalRead(BUTTON) == HIGH) {
    button_msg.data = "Pressed";
  } else {
    button_msg.data = "NOT pressed";
  }
checks if there are any callbacks/services to control the update rate of ROS.
  //publish this "button_msg" using the "button_publisher" publisher
  button_publisher.publish( &button_msg );
  node_handle.spinOnce(); //checks if there are any callbacks/services to control the update rate of ROS.
  
  delay(100);
}
