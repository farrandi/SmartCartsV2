//This is a program to test the prototype smart cart (two-wheeled balancing on its side) via
//its motors. To do this, we need to output PWM signals to the motor controllers of the cart, 
//and power the motor controllers with 24VDC from power supply.
//see websites: https://www.arduino.cc/reference/en/language/functions/analog-io/analogwrite/
//and: https://www.arduino.cc/en/tutorial/potentiometer

//open serial monitor to view console output (Tools-->Serial Monitor or ctrl+shift+M)

//if using Arduino IDE in linux for first time, you may need to do the following to upload an
//arduino sketch to the arduino board (see: https://www.arduino.cc/en/guide/linux):
//1) ls -l /dev/ttyACM*    or    ls -l /dev/ttyUSB0
//2) sudo usermod -a -G dialout fizzer   (where fizzer is your <username>)
//3) log out and log in again for the changes to take effect

int leftForward = 2; //digital pin 2 will be left direction control (1=left forward, 0=left reverse) 
int rightForward = 3; //digital pin 3 will be right direction control (1=right forward, 0=right reverse)

int PWMLeftPin = 10;  //digital pin 10 will be PWM left output
int PWMRightPin = 11; //digital pin 11 will be PWM right output

int potLeftPin = 0;   //analog pin 0 will be potentiometer left input
int potRightPin = 1;  //analog pin 1 will be potentiometer right input
//FYI: wire 5V from arduino 5V output to outside pin of potentiometer, wire COM from arduino GND to
//other outside pin of pot, wire analog 0/1 to middle pin of pot.

int leftVal = 0;    //value read from analog pin 0
int rightVal = 0;   //value read from anaolog pin 1


void setup() {
  // put your setup code here, to run once:

  // set up serial at 9600 baud
  Serial.begin(9600);

  pinMode(leftForward, OUTPUT);
  pinMode(rightForward, OUTPUT);
  digitalWrite(leftForward, HIGH);  //start left wheel forward
  digitalWrite(rightForward, HIGH); //start right wheel forward
  
  pinMode(PWMLeftPin, OUTPUT);  //set PWMLeftPin as an output
  pinMode(PWMRightPin, OUTPUT); //set PWMRightPin as an output
  
}

void loop() {
  // put your main code here, to run repeatedly:

  leftVal = analogRead(potLeftPin); //read left pot; will be value from 0 to 1023
  analogWrite(PWMLeftPin, leftVal/4); //divide by 4 to write value from 0 to 255

  rightVal = analogRead(potRightPin); //read right pot; will be value from 0 to 1023
  analogWrite(PWMRightPin, rightVal/4); //divide by 4 to write value from 0 to 255

  if (Serial.available() > 0)
  {
      // read a single character over serial
      int inByte = Serial.read();

      // do something different for each character
      switch (inByte)
      {
          case 'q':
              digitalWrite(leftForward, HIGH);  //left wheel only forward
              break;
          case 'a':
              digitalWrite(leftForward, LOW);   //left wheel only reverse
              break;
          case 'w':
              digitalWrite(rightForward, HIGH); //right wheel only forward
              break;
          case 's':
              digitalWrite(rightForward, LOW);  //right wheel only reverse
              break;
          case 'i':
              digitalWrite(leftForward, HIGH);  //(BOTH wheels) cart forward
              digitalWrite(rightForward, HIGH);
              break;
          case 'k':
              digitalWrite(leftForward, LOW);   //(BOTH wheels) cart reverse
              digitalWrite(rightForward, LOW);
              break;
          case 'j':
              digitalWrite(leftForward, LOW);   //cart turn left
              digitalWrite(rightForward, HIGH);
              break;
          case 'l':
              digitalWrite(leftForward, HIGH);   //cart turn right
              digitalWrite(rightForward, LOW);
              break;
          default:
              //digitalWrite(leftForward, HIGH);
              //digitalWrite(rightForward, HIGH);
              break;
      }
      
      //Serial.print("I received: ");
      //Serial.println(inByte, DEC);
      //analogWrite(PWMLeftPin, inByte);
  }

  //print values to console of what was written to PWM outputs
  Serial.print("Left: ");
  Serial.print(leftVal/4, DEC);
  Serial.print(",     Right: ");
  Serial.println(rightVal/4, DEC);
}
