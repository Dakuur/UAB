/*
Programa 1
Passa de senyal analògica del potenciòmetre a un valor digital de voltatge
*/

int led = 13;
const int bits_resolucio=13;

void setup() {                
  pinMode(A1, INPUT);     
  Serial.begin(9600);
  analogReadResolution(bits_resolucio);
}

void loop() {
  int x = analogRead(A1);
  delay(500);
  float res = x*3.3/(pow(2,13));
  Serial.print(res);
  Serial.print("\n");
  
}




/*
Programa 2
La brillantor del LED és propocional al voltatge generat pel potenciòmetre
*/
int led = 13;
const int bits_resolucio=13;
const int ledPin = 22 ; 
int x;
float duty_cycle;
int ButtonState;
int ButtonPin = 3;

void setup() {                
  pinMode(A1, INPUT);
  pinMode(ButtonPin, INPUT);     
  Serial.begin(9600);
  analogReadResolution(bits_resolucio);
  pinMode (ledPin, OUTPUT );
  analogWriteFrequency(ledPin,10000);
}

void loop() {
  int ButtonState = digitalRead(ButtonPin);
  if (ButtonState == HIGH) {
    int x = analogRead(A1);
    duty_cycle = x*255/pow(2, 13);
    analogWrite (ledPin, duty_cycle);
    while (ButtonState == HIGH) { 
      ButtonState = digitalRead(ButtonPin);
    }
  }
}






/*
Programa 3
Fa una mitjana de 25 valors analògics per a tenir un valor més aproximat al real, i després aplicar aquest al PWM
*/
const int bits_resolucio=13;
int x;
const int ledPin = 22;
int contador, sumatori;
int duty_cycle;
float mitja;

void setup() {                
  pinMode(A1, INPUT);
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
  analogReadResolution(bits_resolucio);
  analogWriteFrequency(ledPin,10000);
}

void loop() {
  int contador = 0;
  int sumatori = 0;
  while (contador < 25){
    int x = analogRead(A1);
    Serial.print(x);
    Serial.print("\n");
    duty_cycle = x*255/pow(2, 13);
    sumatori = sumatori + duty_cycle;
    contador = contador + 1;
  }
  mitja = sumatori/contador;
  analogWrite (ledPin, mitja);
  Serial.print("Mitjana: ");
  Serial.print(mitja);
  Serial.print("\n");
}
