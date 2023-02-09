//the following code is to read encoder pulses from both the left and right wheel encoders
//with quadrature. Not using Interrupts.Ensure tires are properly inflated to minimize
//slippage between tire and hub. Arduino Uno only has 2 external interrupt pins:
//digital pins 2 and 3. Maybe can't use interrupts properly since I would need 4 pins:
//leftA, leftB, rightA, rightB

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

#define ENCODER_LEFT_A_PIN 2  //use blue wire from Left Encoder Hall Sensor A Vout
#define ENCODER_LEFT_B_PIN 3  //use purple wire from Left Encoder Hall Sensor B Vout

#define ENCODER_RIGHT_A_PIN 4 //use blue wire from Right Encoder Hall Sensor A Vout
#define ENCODER_RIGHT_B_PIN 5 //use purple wire from Right Encoder Hall Sensor B Vout


bool serialEnable = HIGH;  //set to HIGH for testing/debugging. set to LOW under normal operation to speed up program (any serial communication really slows down the program) 

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


void serialReset();  //for testing/debugging
void serialPrintBothPos(int lEncPos, int rEncPos);  //for testing/debugging
void encoders_init();
    

void serialReset(){
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
}

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


void setup() {
  encoders_init();

  if(serialEnable == HIGH){
    Serial.begin(9600);
  }
}



void loop() {
  if(serialEnable == HIGH){
    if (Serial.available() > 0) {
      serialReset();  
    }
  }

  
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

    if(serialEnable == HIGH){
      //only print left wheel encoder count if it is a multiple of 120 (1 wheel revolution should be 480)
      //(I want to check how much the Serial.print() function slows down the program)
      if(lEncPos%120 == 0){serialPrintBothPos(lEncPos, rEncPos);}   
      
      //serialPrintBothPos(lEncPos, rEncPos);
    }
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
    
    if(serialEnable == HIGH){
      //only print left wheel encoder count if it is a multiple of 120 (1 wheel revolution should be 480)
      //(I want to check how much the Serial.print() function slows down the program)
      if(lEncPos%120 == 0){serialPrintBothPos(lEncPos, rEncPos);} 
      
      //serialPrintBothPos(lEncPos, rEncPos);
    }
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
    
    if(serialEnable == HIGH){
      //only print right wheel encoder count if it is a multiple of 120 (1 wheel revolution should be 480)
      //(I want to check how much the Serial.print() function slows down the program)
      if(rEncPos%120 == 0){serialPrintBothPos(lEncPos, rEncPos);} 
      
      //serialPrintBothPos(lEncPos, rEncPos);
    }

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

    if(serialEnable == HIGH){
      //only print right wheel encoder count if it is a multiple of 120 (1 wheel revolution should be 480)
      //(I want to check how much the Serial.print() function slows down the program)
      if(rEncPos%120 == 0){serialPrintBothPos(lEncPos, rEncPos);} 
      
      //serialPrintBothPos(lEncPos, rEncPos);
    }
  }
  rEncBStateLast = rEncBStateCurrent;
}
