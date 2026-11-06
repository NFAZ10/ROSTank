// Sensor + LED board firmware: Raspberry Pi Pico (Arduino-Pico core), the second Pico (/dev/pico_io).
// ROS side: robot_bringup/pico_io.py. Flash with: robot flash io
//
// Serial protocol over USB CDC, newline-terminated ASCII:
//   host -> pico   "C <group> <count> <type>"      set up an LED group: count 0-60, colour order
//                                                  type 0 = GRB (WS2812B), 1 = RGB, 2 = RGBW, 3 = GRBW
//                  "L <group> <first> <n> <effect> <r> <g> <b> <w> <period_ms>"
//                                                  light LEDs first..first+n-1 of a group;
//                                                  effect 0 off, 1 solid, 2 blink, 3 breathe, 4 chase,
//                                                  5 wig-wag (first half, then second half),
//                                                  6 two-colour wig-wag: halves swap between r,g,b and
//                                                  the white w (on RGB-only parts w is mixed white)
//                  "K"                             keepalive (any line counts)
//   pico -> host   "N pico_io <version>"           identity, every 2 s
//                  "R <mm> <ok>"                   VL53L0X range, ~30 Hz; ok 0 = no target in range,
//                                                  "R -1 0" every 1 s while the sensor is missing
//                  "D <temp_c> <humidity_%>"       DHT11 every 2 s; "D nan nan" when it doesn't answer
//                  "I <gx> <gy> <gz> <ax> <ay> <az>"  MPU6050 raw counts, chip axes, 50 Hz:
//                                                  gyro 65.5 per deg/s (+-500), accel 4096 per g (+-8 g)
//                  "M <who_am_i>"                  IMU chip id in decimal, every 2 s; -1 while it's missing
//                  "Q <bus_recoveries>"            I2C controller resets so far (glitches), every 2 s
//                  "V <mV>"                        charge port sense at the GP26 pin (before the divider
//                                                  is scaled back up), 16-sample average, every 200 ms
// Channels are 0-255. 255 is steady on (no PWM flicker for the cameras); on a group without a
// white chip, w is mixed into r, g and b. Effects use the shared millis() clock, so blinks line up
// across groups. No line from the host for HOST_TIMEOUT_MS: everything off but a dim red
// breathe on the back strip (robot powered, stack not running).
//
// Wiring (LED data through a 74AHCT125 at 5 V, 330 R in series at each strip, 10k pull-down on
// each data pin so the LEDs stay dark while the Pico boots):
//   LED groups: 0 headlights GP18 (two 7-LED RGBW discs), 1 body strips GP20 (three 8-LED RGBW
//   strips chained right -> back -> left, 24 LEDs), 2 spare GP16, 3 front bar GP19 (8 RGB, GRB)
//   VL53L0X: SDA GP4, SCL GP5, XSHUT GP7 (high = on)      DHT11: data GP6 (10k pull-up to 3.3 V)
//   MPU6050: same I2C bus, AD0 low (0x68); its INT is wired to GP8 but isn't used (left an input):
//   data-ready is read from the chip. A sensor that stops answering gets the I2C controller reset
//   and the bus clocked free (busRecover) before the retry: after an electrical glitch (LED
//   switching) the controller could stay stuck until the Pico rebooted.
//   Charge port sense: port + (charger side of the 1N5822) - 6.8k - GP26 - 1.5k - AGND
// Sensors run on core 1 (the DHT11 read blocks ~25 ms with interrupts off, so it costs the IMU a
// sample every 2 s); LEDs and serial on core 0.

#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include <DHT.h>
#include <VL53L0X.h>
#include <Wire.h>

#define VERSION 8
const int GROUPS = 4, MAX_LEDS = 60;
const int GROUP_PIN[GROUPS] = {18, 20, 16, 19};
const int BODY_GROUP = 1, BACK_FIRST = 8, BACK_N = 8;  // the back strip, for the link-lost glow
const int SDA_PIN = 4, SCL_PIN = 5, XSHUT_PIN = 7, DHT_PIN = 6, CHARGE_SENSE_PIN = 26;
const uint32_t HOST_TIMEOUT_MS = 2000;
const uint32_t FRAME_MS = 20;           // 50 Hz animation
const uint32_t IDENT_MS = 2000, DHT_MS = 2000, SENSOR_RETRY_MS = 1000;
const uint32_t TOF_TIMEOUT_MS = 200, IMU_TIMEOUT_MS = 200, IMU_PERIOD_MS = 18;  // just under the chip's 20 ms
const uint32_t IMU_RETRY_MS = 2000;  // a reset blocks the bus ~100 ms: don't starve the ToF
const uint16_t TYPES[4] = {NEO_GRB + NEO_KHZ800, NEO_RGB + NEO_KHZ800, NEO_RGBW + NEO_KHZ800, NEO_GRBW + NEO_KHZ800};

enum Effect : uint8_t { OFF = 0, SOLID, BLINK, BREATHE, CHASE, WIGWAG, WIGWAG2 };
struct Led {
  uint8_t effect, r, g, b, w;
  uint16_t period;
  uint8_t segFirst, segN;   // the L command's range, for chase
};

Adafruit_NeoPixel strip[GROUPS] = {
  Adafruit_NeoPixel(0, GROUP_PIN[0], TYPES[0]), Adafruit_NeoPixel(0, GROUP_PIN[1], TYPES[0]),
  Adafruit_NeoPixel(0, GROUP_PIN[2], TYPES[0]), Adafruit_NeoPixel(0, GROUP_PIN[3], TYPES[0]),
};
Led leds[GROUPS][MAX_LEDS];
int groupCount[GROUPS] = {0, 0, 0, 0};
bool groupRGBW[GROUPS] = {false, false, false, false};
uint32_t lastHostMs = 0, lastFrameMs = 0, lastIdentMs = 0, lastSenseMs = 0;
const uint32_t SENSE_MS = 200;
bool hostSeen = false;
char line[96];
uint8_t lineLen = 0;

// Written by core 1, read by core 0 (aligned 32-bit values: single loads/stores).
volatile int32_t rangeMm = -1;
volatile uint32_t rangeOk = 0, rangeSeq = 0, rangeMissingSeq = 0;
volatile float tempC = NAN, humidity = NAN;
volatile uint32_t dhtSeq = 0;
volatile int16_t imuSample[6];        // gx gy gz ax ay az
volatile uint32_t imuSeq = 0;         // odd while core 1 is writing imuSample
volatile int32_t imuId = -1;
volatile uint32_t busRecoveries = 0;

// ---------------- LEDs (core 0) ----------------
void configureGroup(int g, int count, int type) {
  if (g < 0 || g >= GROUPS || count < 0 || count > MAX_LEDS || type < 0 || type > 3) return;
  if (groupCount[g]) {  // blank the old length first (show() before begin() would drive an unclaimed PIO)
    strip[g].clear();
    strip[g].show();
  }
  strip[g].updateType(TYPES[type]);
  strip[g].updateLength(count);
  strip[g].begin();
  groupCount[g] = count;
  groupRGBW[g] = type >= 2;
  for (int i = 0; i < MAX_LEDS; i++) leds[g][i] = Led{OFF, 0, 0, 0, 0, 1000, 0, 1};
}

void setLeds(int g, int first, int n, int effect, int r, int gr, int b, int w, int period) {
  if (g < 0 || g >= GROUPS || first < 0 || n < 1 || first + n > MAX_LEDS || effect < 0 || effect > WIGWAG2) return;
  for (int i = first; i < first + n; i++)
    leds[g][i] = Led{(uint8_t)effect, (uint8_t)constrain(r, 0, 255), (uint8_t)constrain(gr, 0, 255),
                     (uint8_t)constrain(b, 0, 255), (uint8_t)constrain(w, 0, 255),
                     (uint16_t)constrain(period, 50, 60000), (uint8_t)first, (uint8_t)n};
}

// Brightness 0-1 of one LED right now.
float level(const Led &l, int index, uint32_t now) {
  float phase = (now % l.period) / (float)l.period;
  switch (l.effect) {
    case SOLID:
    case WIGWAG2: return 1.0f;
    case BLINK: return phase < 0.5f ? 1.0f : 0.0f;
    case BREATHE: return 0.5f - 0.5f * cosf(2.0f * PI * phase);
    case CHASE: {  // a short comet running first -> last
      float head = phase * l.segN, d = head - (index - l.segFirst);
      if (d < 0) d += l.segN;
      return d < 3.0f ? 1.0f - d / 3.0f : 0.0f;
    }
    case WIGWAG: return ((index - l.segFirst) * 2 < l.segN) == (phase < 0.5f) ? 1.0f : 0.0f;
    default: return 0.0f;
  }
}

void render(uint32_t now) {
  bool linkLost = !hostSeen || now - lastHostMs > HOST_TIMEOUT_MS;
  for (int g = 0; g < GROUPS; g++) {
    if (!groupCount[g]) continue;
    for (int i = 0; i < groupCount[g]; i++) {
      Led l = leds[g][i];
      if (linkLost) {
        bool back = g == BODY_GROUP && i >= BACK_FIRST && i < BACK_FIRST + BACK_N;
        l = back ? Led{BREATHE, 20, 0, 0, 0, 4000, 0, 1} : Led{OFF, 0, 0, 0, 0, 1000, 0, 1};
      }
      float k = level(l, i, now);
      int r = l.r * k, gr = l.g * k, b = l.b * k, w = l.w * k;
      if (l.effect == WIGWAG2) {  // halves swap between the colour and the white
        bool colour = ((i - l.segFirst) * 2 < l.segN) == ((now % l.period) * 2 < l.period);
        if (colour) w = 0; else r = gr = b = 0;
      }
      if (!groupRGBW[g]) { r = min(255, r + w); gr = min(255, gr + w); b = min(255, b + w); w = 0; }
      strip[g].setPixelColor(i, r, gr, b, w);
    }
    strip[g].show();
  }
}

// ---------------- serial (core 0) ----------------
void handleLine(char *s) {
  lastHostMs = millis();
  hostSeen = true;
  int a[9];
  if (s[0] == 'C' && sscanf(s + 1, "%d %d %d", &a[0], &a[1], &a[2]) == 3) {
    configureGroup(a[0], a[1], a[2]);
  } else if (s[0] == 'L' && sscanf(s + 1, "%d %d %d %d %d %d %d %d %d", &a[0], &a[1], &a[2], &a[3], &a[4], &a[5],
                                   &a[6], &a[7], &a[8]) == 9) {
    setLeds(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
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
  analogReadResolution(12);
  for (int g = 0; g < GROUPS; g++) configureGroup(g, 0, 0);
}

void loop() {
  readSerial();
  uint32_t now = millis();
  static uint32_t sentRange = 0, sentMissing = 0, sentDht = 0, sentImu = 0;
  if (rangeSeq != sentRange) {
    sentRange = rangeSeq;
    Serial.printf("R %ld %lu\n", (long)rangeMm, (unsigned long)rangeOk);
  }
  if (rangeMissingSeq != sentMissing) {
    sentMissing = rangeMissingSeq;
    Serial.printf("R -1 0\n");
  }
  if (dhtSeq != sentDht) {
    sentDht = dhtSeq;
    float t = tempC, h = humidity;
    if (isnan(t) || isnan(h)) Serial.printf("D nan nan\n");
    else Serial.printf("D %.1f %.1f\n", t, h);
  }
  uint32_t seq = imuSeq;
  if (seq != sentImu && !(seq & 1)) {
    int16_t v[6];
    for (int i = 0; i < 6; i++) v[i] = imuSample[i];
    if (imuSeq == seq) {  // otherwise core 1 wrote over it mid-copy: take the next one
      sentImu = seq;
      Serial.printf("I %d %d %d %d %d %d\n", v[0], v[1], v[2], v[3], v[4], v[5]);
    }
  }
  if (now - lastSenseMs >= SENSE_MS) {
    lastSenseMs = now;
    uint32_t sum = 0;
    for (int i = 0; i < 16; i++) sum += analogRead(CHARGE_SENSE_PIN);
    Serial.printf("V %lu\n", (unsigned long)(sum * 3300UL / (16UL * 4095UL)));
  }
  if (now - lastIdentMs >= IDENT_MS) {
    lastIdentMs = now;
    Serial.printf("N pico_io %d\n", VERSION);
    Serial.printf("M %ld\n", (long)imuId);
    Serial.printf("Q %lu\n", (unsigned long)busRecoveries);
  }
  if (now - lastFrameMs >= FRAME_MS) {
    lastFrameMs = now;
    render(now);
    digitalWrite(LED_BUILTIN, hostSeen && now - lastHostMs <= HOST_TIMEOUT_MS);  // on = host connected
  }
}

// ---------------- sensors (core 1) ----------------
VL53L0X tof;
DHT dht(DHT_PIN, DHT11);
bool tofUp = false, imuUp = false;
uint32_t lastTofTry = 0, lastDhtMs = 0, lastImuTry = 0, lastTofMs = 0, lastImuMs = 0;

const uint8_t MPU_ADDR = 0x68;
const uint8_t MPU_SMPLRT_DIV = 0x19, MPU_CONFIG = 0x1A, MPU_GYRO_CONFIG = 0x1B, MPU_ACCEL_CONFIG = 0x1C,
              MPU_INT_ENABLE = 0x38, MPU_INT_STATUS = 0x3A, MPU_ACCEL_XOUT_H = 0x3B, MPU_PWR_MGMT_1 = 0x6B,
              MPU_WHO_AM_I = 0x75, MPU6500_ACCEL_CONFIG2 = 0x1D, MPU6050_ID = 0x68;

bool mpuWrite(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool mpuRead(uint8_t reg, uint8_t *out, size_t n) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, n) != n) return false;
  for (size_t i = 0; i < n; i++) out[i] = Wire.read();
  return true;
}

bool startImu() {
  uint8_t id;
  if (!mpuRead(MPU_WHO_AM_I, &id, 1)) return false;
  if (!mpuWrite(MPU_PWR_MGMT_1, 0x80)) return false;  // reset
  delay(100);
  bool ok = mpuWrite(MPU_PWR_MGMT_1, 0x01)            // awake, clocked from the gyro
         && mpuWrite(MPU_CONFIG, 4)                   // 20 Hz low-pass, 1 kHz internal rate
         && mpuWrite(MPU_SMPLRT_DIV, 19)              // 1 kHz / 20 = 50 Hz
         && mpuWrite(MPU_GYRO_CONFIG, 1 << 3)         // +-500 deg/s
         && mpuWrite(MPU_ACCEL_CONFIG, 2 << 3)        // +-8 g
         && mpuWrite(MPU_INT_ENABLE, 0x01);           // data-ready flag
  if (id != MPU6050_ID) ok = ok && mpuWrite(MPU6500_ACCEL_CONFIG2, 4);  // clones (6500 family) filter accel separately: 21 Hz
  if (ok) imuId = id;
  return ok;
}

// True while the chip is answering; publishes a sample when one is ready.
bool pollImu(uint32_t now) {
  uint8_t status;
  if (!mpuRead(MPU_INT_STATUS, &status, 1)) return false;
  if (!(status & 0x01)) return now - lastImuMs < IMU_TIMEOUT_MS;  // a chip that reset itself stops flagging data
  uint8_t b[14];  // accel xyz, temperature, gyro xyz: high byte first
  if (!mpuRead(MPU_ACCEL_XOUT_H, b, sizeof(b))) return false;
  lastImuMs = now;
  imuSeq = imuSeq + 1;
  for (int i = 0; i < 3; i++) {
    imuSample[i] = (int16_t)(b[8 + 2 * i] << 8 | b[9 + 2 * i]);
    imuSample[3 + i] = (int16_t)(b[2 * i] << 8 | b[1 + 2 * i]);
  }
  imuSeq = imuSeq + 1;
  return true;
}

bool startTof() {
  digitalWrite(XSHUT_PIN, LOW);  // hard reset: also recovers a sensor that hung mid-read
  delay(5);
  digitalWrite(XSHUT_PIN, HIGH);
  delay(5);
  tof.setTimeout(200);
  if (!tof.init()) return false;
  tof.setMeasurementTimingBudget(33000);  // ~30 Hz, the default accuracy
  tof.startContinuous();
  lastTofMs = millis();
  return true;
}

void startBus() {
  Wire.setSDA(SDA_PIN);
  Wire.setSCL(SCL_PIN);
  Wire.begin();
  Wire.setClock(100000);  // 400 kHz kept losing the IMU (marginal pull-ups); 100 kHz is plenty for 50 Hz + the ToF
  Wire.setTimeout(50);
}

// Reset the I2C controller and free a slave holding SDA low (9 clocks, then a STOP).
void busRecover() {
  Wire.end();
  pinMode(SDA_PIN, INPUT_PULLUP);
  pinMode(SCL_PIN, OUTPUT);
  for (int i = 0; i < 9 && !digitalRead(SDA_PIN); i++) {
    digitalWrite(SCL_PIN, LOW); delayMicroseconds(5);
    digitalWrite(SCL_PIN, HIGH); delayMicroseconds(5);
  }
  pinMode(SDA_PIN, OUTPUT);  // STOP: SDA low -> high while SCL is high
  digitalWrite(SDA_PIN, LOW); delayMicroseconds(5);
  digitalWrite(SCL_PIN, HIGH); delayMicroseconds(5);
  digitalWrite(SDA_PIN, HIGH); delayMicroseconds(5);
  pinMode(SDA_PIN, INPUT);
  pinMode(SCL_PIN, INPUT);
  startBus();
  busRecoveries = busRecoveries + 1;
}

void setup1() {
  pinMode(XSHUT_PIN, OUTPUT);
  startBus();
  dht.begin();
}

void loop1() {
  uint32_t now = millis();
  if (!tofUp && now - lastTofTry >= SENSOR_RETRY_MS) {
    lastTofTry = now;
    tofUp = startTof();
    if (!tofUp) rangeMissingSeq++;
    now = millis();  // that took a while, and the timeouts below compare against times it set
  }
  if (tofUp) {
    if (tof.readReg(VL53L0X::RESULT_INTERRUPT_STATUS) & 0x07) {  // a sample every ~33 ms: don't wait for it, the IMU can't
      uint16_t mm = tof.readRangeContinuousMillimeters();
      lastTofMs = now;
      rangeOk = mm < 8000;  // 8190/65535 = nothing in range
      rangeMm = mm;
      rangeSeq++;
    } else if (now - lastTofMs > TOF_TIMEOUT_MS) {
      tofUp = false;
      rangeMissingSeq++;
    }
  }
  if (!imuUp && now - lastImuTry >= (lastImuTry ? IMU_RETRY_MS : 0)) {
    if (lastImuTry) {  // it was up, or the last try failed: start from a clean bus
      busRecover();  // the ToF keeps ranging through this (its continuous mode lives in the sensor)
    }
    lastImuTry = now;
    imuUp = startImu();
    if (!imuUp) imuId = -1;
    now = millis();
    lastImuMs = now;
  }
  if (imuUp && now - lastImuMs >= IMU_PERIOD_MS && !pollImu(now)) {
    imuUp = false;
    imuId = -1;
  }
  if (now - lastDhtMs >= DHT_MS) {
    lastDhtMs = now;
    float t = dht.readTemperature(), h = dht.readHumidity();
    tempC = t;
    humidity = h;
    dhtSeq++;
  }
  delay(1);
}
