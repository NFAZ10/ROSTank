// Tracked-robot drive firmware: Raspberry Pi Pico + L298N (Arduino-Pico core).
//
// Serial protocol over USB CDC, newline-terminated ASCII:
//   host -> pico   "V <left_rad_s> <right_rad_s>"   target sprocket speeds
//                  "S"                              stop now
//                  "P"                              debug: reply "P <lA> <lB> <rA> <rB> <gpio hex>" raw levels
//                  "T <gpio> <0|1>"                 debug: hold one motor pin (EN/IN) at a level, control
//                                                   loop paused until the next V/S or the command timeout
//   pico -> host   "E <left_rad> <right_rad>"       cumulative sprocket angle, every LOOP_MS
//                  "B <millivolts>"                 battery divider at GP26 (ADC0), every BATT_MS;
//                                                   the host scales it (pico_bridge battery_divider)
// Without encoders the reported angle is integrated from the command (open loop).
// Motors stop if no V command arrives within CMD_TIMEOUT_MS.
//
// Wiring (remove the ENA/ENB jumpers on the L298N; tie L298N GND to Pico GND):
//   Left  track: ENA=GP4  IN1=GP10 IN2=GP8     encoder A=GP20 B=GP21
//   Right track: ENB=GP1  IN3=GP3  IN4=GP2     encoder A=GP14 B=GP15
//   Battery: 6.8k from battery + to GP26, 1.5k from GP26 to GND

#include <Arduino.h>
#include <initializer_list>

// ---------------- config ----------------
#define HAS_ENCODERS 1                  // hall encoders: left GP20/21, right GP14/15 (wired 2026-10-27)
const float TICKS_PER_REV = 748.0f;     // per sprocket rev, x2 decoding: 11 PPR x ~34:1 x 2 (hand-measured 739-752)
const float FREE_RAD_S = 14.7f;         // sprocket speed at 100% PWM, tracks off, 12 V (measured 2026-10-27)
const float MAX_RAD_S = 13.0f;          // command limit, below FREE_RAD_S so the PI keeps headroom
const float MIN_DUTY = 0.18f;           // L298N drops ~2V; below this the motors just hum
const float KP = 0.05f, KI = 0.4f;      // speed PI, only used with encoders
const bool  INVERT_LEFT = true, INVERT_RIGHT = true;   // motor wiring: forward command must make the encoder count up
const int   ENC_SIGN_LEFT = -1, ENC_SIGN_RIGHT = -1;   // make forward count up (checked by hand)
const uint32_t CMD_TIMEOUT_MS = 500;
const uint32_t LOOP_MS = 20;            // 50 Hz control + odometry report
const int PWM_FREQ = 2000;
const int BATT_PIN = 26;                // ADC0, battery divider
const uint32_t BATT_MS = 500;

const int L_EN = 4, L_IN1 = 10, L_IN2 = 8;  // as wired 2026-10-27 (IN1/IN4 crossed vs. the plan)
const int R_EN = 1, R_IN1 = 3, R_IN2 = 2;
const int L_A = 20, L_B = 21, R_A = 14, R_B = 15;

// ---------------- state ----------------
volatile int32_t encL = 0, encR = 0;
float targetL = 0, targetR = 0;
float integL = 0, integR = 0;
double angleL = 0, angleR = 0;          // reported cumulative angle [rad]
int32_t lastEncL = 0, lastEncR = 0;
uint32_t lastCmdMs = 0, lastLoopMs = 0, lastBattMs = 0;
bool pinTest = false;                   // "T" debug mode owns the motor pins
char line[64];
uint8_t lineLen = 0;

void isrL() { encL += ((digitalRead(L_A) == digitalRead(L_B)) ? -1 : 1) * ENC_SIGN_LEFT; }
void isrR() { encR += ((digitalRead(R_A) == digitalRead(R_B)) ? -1 : 1) * ENC_SIGN_RIGHT; }

void setMotor(int en, int in1, int in2, float duty, bool invert) {
  if (invert) duty = -duty;
  duty = constrain(duty, -1.0f, 1.0f);
  float mag = fabsf(duty);
  if (mag < 0.01f) {                    // coast
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    analogWrite(en, 0);
    return;
  }
  mag = MIN_DUTY + (1.0f - MIN_DUTY) * mag;
  digitalWrite(in1, duty > 0 ? HIGH : LOW);
  digitalWrite(in2, duty > 0 ? LOW : HIGH);
  analogWrite(en, (int)(mag * 255));
}

float control(float target, float measured, float &integ, float dt) {
  float ff = target / FREE_RAD_S;
  if (!HAS_ENCODERS) return ff;
  if (fabsf(target) < 1e-3f) { integ = 0; return 0; }
  float err = target - measured;
  integ = constrain(integ + err * dt, -1.0f / KI, 1.0f / KI);
  return ff + KP * err + KI * integ;
}

void handleLine(char *s) {
  if (s[0] == 'V' || s[0] == 'S') pinTest = false;
  if (s[0] == 'V') {
    float l, r;
    if (sscanf(s + 1, "%f %f", &l, &r) == 2) {
      targetL = constrain(l, -MAX_RAD_S, MAX_RAD_S);
      targetR = constrain(r, -MAX_RAD_S, MAX_RAD_S);
      lastCmdMs = millis();
    }
  } else if (s[0] == 'S') {
    targetL = targetR = 0;
  } else if (s[0] == 'T') {
    int pin, level;
    if (sscanf(s + 1, "%d %d", &pin, &level) == 2 && (pin == L_EN || pin == L_IN1 || pin == L_IN2 || pin == R_EN || pin == R_IN1 || pin == R_IN2)) {
      pinTest = true;
      lastCmdMs = millis();
      if (pin == L_EN || pin == R_EN) analogWrite(pin, level ? 255 : 0);
      else digitalWrite(pin, level ? HIGH : LOW);
    }
  } else if (s[0] == 'P') {
    Serial.printf("P %d %d %d %d %08lx\n", digitalRead(L_A), digitalRead(L_B), digitalRead(R_A), digitalRead(R_B),
                  (unsigned long)gpio_get_all());
  }
}

void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineLen) { line[lineLen] = 0; handleLine(line); lineLen = 0; }
    } else if (lineLen < sizeof(line) - 1) {
      line[lineLen++] = c;
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  for (int p : {L_EN, L_IN1, L_IN2, R_EN, R_IN1, R_IN2}) { pinMode(p, OUTPUT); digitalWrite(p, LOW); }
  // Unused header pins pulled up, so a mis-plugged sensor shows up in the P debug reply.
  // Not GP26: it reads the battery divider, which a pull-up would skew.
  for (int p : {0, 5, 6, 7, 9, 11, 12, 13, 16, 17, 18, 19, 22, 27, 28}) pinMode(p, INPUT_PULLUP);
  analogReadResolution(12);
  analogWriteFreq(PWM_FREQ);
  analogWriteRange(255);
#if HAS_ENCODERS
  for (int p : {L_A, L_B, R_A, R_B}) pinMode(p, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(L_A), isrL, CHANGE);
  attachInterrupt(digitalPinToInterrupt(R_A), isrR, CHANGE);
#endif
  lastLoopMs = millis();
}

void loop() {
  readSerial();
  uint32_t now = millis();
  if (now - lastCmdMs > CMD_TIMEOUT_MS) {
    targetL = targetR = 0;
    pinTest = false;
  }
  if (now - lastLoopMs < LOOP_MS) return;
  float dt = (now - lastLoopMs) * 1e-3f;
  lastLoopMs = now;

  float velL, velR;
#if HAS_ENCODERS
  noInterrupts();
  int32_t eL = encL, eR = encR;
  interrupts();
  float radPerTick = 2.0f * PI / TICKS_PER_REV;
  velL = (eL - lastEncL) * radPerTick / dt;
  velR = (eR - lastEncR) * radPerTick / dt;
  lastEncL = eL; lastEncR = eR;
  angleL = eL * (double)radPerTick;
  angleR = eR * (double)radPerTick;
#else
  velL = targetL; velR = targetR;
  angleL += velL * dt;
  angleR += velR * dt;
#endif

  float dutyL = control(targetL, velL, integL, dt), dutyR = control(targetR, velR, integR, dt);
  if (!pinTest) {
    setMotor(L_EN, L_IN1, L_IN2, dutyL, INVERT_LEFT);
    setMotor(R_EN, R_IN1, R_IN2, dutyR, INVERT_RIGHT);
  }
  digitalWrite(LED_BUILTIN, (targetL != 0 || targetR != 0) ? HIGH : LOW);

  Serial.printf("E %.5f %.5f\n", angleL, angleR);

  if (now - lastBattMs >= BATT_MS) {
    lastBattMs = now;
    uint32_t sum = 0;
    for (int i = 0; i < 16; i++) sum += analogRead(BATT_PIN);
    Serial.printf("B %lu\n", (unsigned long)(sum * 3300UL / (16UL * 4095UL)));  // ADC ref = 3.3 V
  }
}
