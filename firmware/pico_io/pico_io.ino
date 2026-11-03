// Sensor + LED board firmware: Raspberry Pi Pico (Arduino-Pico core), the second Pico (/dev/pico_io).
// ROS side: robot_bringup/pico_io.py. Flash with: robot flash io
//
// Serial protocol over USB CDC, newline-terminated ASCII:
//   host -> pico   "C <group> <count> <type>"      set up an LED group: count 0-60, colour order
//                                                  type 0 = GRB (WS2812B), 1 = RGB, 2 = RGBW, 3 = GRBW
//                  "L <group> <first> <n> <effect> <r> <g> <b> <w> <period_ms>"
//                                                  light LEDs first..first+n-1 of a group;
//                                                  effect 0 off, 1 solid, 2 blink, 3 breathe, 4 chase,
//                                                  5 wig-wag (first half, then second half)
//                  "K"                             keepalive (any line counts)
//   pico -> host   "N pico_io <version>"           identity, every 2 s
//                  "R <mm> <ok>"                   VL53L0X range, ~30 Hz; ok 0 = no target in range,
//                                                  "R -1 0" every 1 s while the sensor is missing
//                  "D <temp_c> <humidity_%>"       DHT11 every 2 s; "D nan nan" when it doesn't answer
// Channels are 0-255. 255 is steady on (no PWM flicker for the cameras); on a group without a
// white chip, w is mixed into r, g and b. Effects use the shared millis() clock, so blinks line up
// across groups. No line from the host for HOST_TIMEOUT_MS: everything off but a dim red
// breathe on the tail group (robot powered, stack not running).
//
// Wiring (LED data through a 74AHCT125 at 5 V, 330 R in series at each strip, 10k pull-down on
// each data pin so the LEDs stay dark while the Pico boots):
//   LED groups: 0 headlights GP16, 1 top strips GP17, 2 tail lights GP18, 3 spare GP19
//   VL53L0X: SDA GP4, SCL GP5, XSHUT GP7 (high = on)      DHT11: data GP6 (10k pull-up to 3.3 V)
// Sensors run on core 1 (the DHT11 read blocks ~25 ms with interrupts off); LEDs and serial on core 0.

#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include <DHT.h>
#include <VL53L0X.h>
#include <Wire.h>

#define VERSION 2
const int GROUPS = 4, MAX_LEDS = 60;
const int GROUP_PIN[GROUPS] = {16, 17, 18, 19};
const int TAIL_GROUP = 2;
const int SDA_PIN = 4, SCL_PIN = 5, XSHUT_PIN = 7, DHT_PIN = 6;
const uint32_t HOST_TIMEOUT_MS = 2000;
const uint32_t FRAME_MS = 20;           // 50 Hz animation
const uint32_t IDENT_MS = 2000, DHT_MS = 2000, SENSOR_RETRY_MS = 1000;
const uint16_t TYPES[4] = {NEO_GRB + NEO_KHZ800, NEO_RGB + NEO_KHZ800, NEO_RGBW + NEO_KHZ800, NEO_GRBW + NEO_KHZ800};

enum Effect : uint8_t { OFF = 0, SOLID, BLINK, BREATHE, CHASE, WIGWAG };
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
uint32_t lastHostMs = 0, lastFrameMs = 0, lastIdentMs = 0;
bool hostSeen = false;
char line[96];
uint8_t lineLen = 0;

// Written by core 1, read by core 0 (aligned 32-bit values: single loads/stores).
volatile int32_t rangeMm = -1;
volatile uint32_t rangeOk = 0, rangeSeq = 0, rangeMissingSeq = 0;
volatile float tempC = NAN, humidity = NAN;
volatile uint32_t dhtSeq = 0;

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
  if (g < 0 || g >= GROUPS || first < 0 || n < 1 || first + n > MAX_LEDS || effect < 0 || effect > WIGWAG) return;
  for (int i = first; i < first + n; i++)
    leds[g][i] = Led{(uint8_t)effect, (uint8_t)constrain(r, 0, 255), (uint8_t)constrain(gr, 0, 255),
                     (uint8_t)constrain(b, 0, 255), (uint8_t)constrain(w, 0, 255),
                     (uint16_t)constrain(period, 50, 60000), (uint8_t)first, (uint8_t)n};
}

// Brightness 0-1 of one LED right now.
float level(const Led &l, int index, uint32_t now) {
  float phase = (now % l.period) / (float)l.period;
  switch (l.effect) {
    case SOLID: return 1.0f;
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
      if (linkLost) l = (g == TAIL_GROUP) ? Led{BREATHE, 20, 0, 0, 0, 4000, 0, 1} : Led{OFF, 0, 0, 0, 0, 1000, 0, 1};
      float k = level(l, i, now);
      int r = l.r * k, gr = l.g * k, b = l.b * k, w = l.w * k;
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
  for (int g = 0; g < GROUPS; g++) configureGroup(g, 0, 0);
}

void loop() {
  readSerial();
  uint32_t now = millis();
  static uint32_t sentRange = 0, sentMissing = 0, sentDht = 0;
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
  if (now - lastIdentMs >= IDENT_MS) {
    lastIdentMs = now;
    Serial.printf("N pico_io %d\n", VERSION);
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
bool tofUp = false;
uint32_t lastTofTry = 0, lastDhtMs = 0;

bool startTof() {
  digitalWrite(XSHUT_PIN, LOW);  // hard reset: also recovers a sensor that hung mid-read
  delay(5);
  digitalWrite(XSHUT_PIN, HIGH);
  delay(5);
  tof.setTimeout(200);
  if (!tof.init()) return false;
  tof.setMeasurementTimingBudget(33000);  // ~30 Hz, the default accuracy
  tof.startContinuous();
  return true;
}

void setup1() {
  pinMode(XSHUT_PIN, OUTPUT);
  Wire.setSDA(SDA_PIN);
  Wire.setSCL(SCL_PIN);
  Wire.begin();
  Wire.setClock(400000);
  Wire.setTimeout(50);
  dht.begin();
}

void loop1() {
  uint32_t now = millis();
  if (!tofUp && now - lastTofTry >= SENSOR_RETRY_MS) {
    lastTofTry = now;
    tofUp = startTof();
    if (!tofUp) rangeMissingSeq++;
  }
  if (tofUp) {
    uint16_t mm = tof.readRangeContinuousMillimeters();  // waits for the next sample (~33 ms)
    if (tof.timeoutOccurred()) {
      tofUp = false;
      rangeMissingSeq++;
    } else {
      rangeOk = mm < 8000;  // 8190/65535 = nothing in range
      rangeMm = mm;
      rangeSeq++;
    }
  } else {
    delay(20);
  }
  if (now - lastDhtMs >= DHT_MS) {
    lastDhtMs = now;
    float t = dht.readTemperature(), h = dht.readHumidity();
    tempC = t;
    humidity = h;
    dhtSeq++;
  }
}
