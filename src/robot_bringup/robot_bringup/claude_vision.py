"""Ask Claude what the front camera shows: after a crash, or when asked.

A switchable service (Services menu: 'Claude vision'), off unless switched on.
  in:  /crash/event         std_msgs/String JSON (crash_monitor.py): a forward hit -> look
       /vision/describe     std_srvs/Trigger: look now
  out: /vision/report       latched std_msgs/String JSON:
                            {time, trigger, status: asking|ok|error, text, image}
       /vision/status       latched std_msgs/String: 'connected' | 'checking' | 'no key' |
                            'bad key' | 'offline' | 'error: ...'; the API is checked at start and
                            every check_s with a model lookup (no tokens), and after each look
Each look grabs one left-camera frame (the camera starts on demand, ~20 s from cold), sends it
to the Claude API with the front range, and saves the JPEG + report under log_dir.

The API key is read from key_file (one line, chmod 600, never in the repo), or the usual
ANTHROPIC_API_KEY / `ant auth login` credentials if that file is missing. At most one look per
min_interval_s, so a run of bumps can't run up a bill.
"""
import base64
import json
import math
import os
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Range
from std_msgs.msg import String
from std_srvs.srv import Trigger

SYSTEM = (
    "You are the eyes of a small indoor tracked robot, about 30 cm long. The photo is from its front "
    "camera, 7 cm above the floor and tilted about 20 degrees up, so the floor just ahead is at the "
    "bottom of the picture and close objects look large. Someone is reading your answer on a phone "
    "to decide what to do with the robot. Answer in at most three short sentences of plain text: "
    "what is directly in front of the robot, and whether something is blocking its way (what it is, "
    "roughly how far). If the picture is too dark, blurry or blocked to tell, say so."
)


def prompt_for(trigger, front_range):
    if trigger == 'crash':
        ask = "The robot just bumped into something while driving forward. What did it most likely hit?"
    else:
        ask = "What does the robot see ahead?"
    if front_range is not None:
        ask += (f" Its front distance sensor reads {front_range * 100:.0f} cm." if math.isfinite(front_range)
                else " Its front distance sensor sees nothing within 2 m.")
    return ask


def read_key(path):
    try:
        with open(os.path.expanduser(path)) as f:
            return f.read().strip() or None
    except OSError:
        return None


class ClaudeVision(Node):
    def __init__(self):
        super().__init__('claude_vision')
        p = lambda name, default: self.declare_parameter(name, default).value
        self.model = p('model', 'claude-opus-5-5')
        self.effort = p('effort', 'low')  # a quick look, not a hard problem
        self.key_file = p('key_file', '~/.config/robot/anthropic_api_key')
        self.log_dir = os.path.expanduser(p('log_dir', '~/robot_ws/vision_log'))
        self.min_interval = p('min_interval_s', 20.0)
        self.camera_wait = p('camera_wait_s', 40.0)
        self.settle_s = p('settle_s', 1.0)  # frames after the camera starts, for exposure
        self.check_s = p('check_s', 300.0)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.report_pub = self.create_publisher(String, '/vision/report', latched)
        self.status_pub = self.create_publisher(String, '/vision/status', latched)
        self.status = None
        self.create_subscription(String, '/crash/event', self.on_crash, 10)
        self.create_service(Trigger, '/vision/describe', self.on_describe)
        self.front_range = (None, 0.0)
        self.create_subscription(Range, '/range/front', self.on_range, qos_profile_sensor_data)

        self.lock = threading.Lock()
        self.busy, self.last_ask = False, 0.0
        self.image_sub, self.frame, self.first_frame = None, None, 0.0
        self.client = None
        self.camera_wanted = False
        self.create_timer(0.2, self.manage_camera)  # subscriptions only change on the executor thread
        self.set_status('checking')
        self.checking = False
        self.check_soon()
        self.create_timer(self.check_s, self.check_soon)

    # ---------- connection ----------
    def set_status(self, status):
        if status != self.status:
            self.status = status
            self.status_pub.publish(String(data=status))
            self.get_logger().info(f'Claude API: {status}')

    def check_soon(self):
        if not self.checking and not self.busy:
            self.checking = True
            threading.Thread(target=self.check, daemon=True).start()

    def get_client(self):
        import anthropic  # here, so the node still starts (and says why) without the SDK

        if self.client is None:
            key = read_key(self.key_file)
            # A key in key_file wins; without one the SDK looks for its usual credentials.
            self.client = anthropic.Anthropic(api_key=key, timeout=60.0) if key else anthropic.Anthropic(timeout=60.0)
        return self.client

    def check(self):
        """Is the key good and the API reachable? A model lookup: free, no tokens."""
        try:
            import anthropic
        except ImportError:
            self.checking = False
            return self.set_status('error: the anthropic Python package is missing (pip3 install --user anthropic)')
        try:
            self.get_client().models.retrieve(self.model, timeout=15.0)
            self.set_status('connected')
        except anthropic.AuthenticationError:
            self.client = None
            self.set_status('bad key')
        except anthropic.PermissionDeniedError:
            self.set_status(f'error: the key may not use {self.model}')
        except anthropic.NotFoundError:
            self.set_status(f'error: model {self.model} not found')
        except anthropic.APIConnectionError:
            self.set_status('offline')
        except anthropic.APIStatusError as e:
            self.set_status('offline' if e.status_code >= 500 else f'error: {e.status_code}')
        except Exception as e:  # noqa: BLE001 - e.g. no credentials found at all
            self.client = None
            self.set_status('no key' if 'api_key' in str(e) or 'credentials' in str(e).lower() else f'error: {e}')
        finally:
            self.checking = False

    # ---------- inputs ----------
    def on_range(self, msg):
        self.front_range = (msg.range, time.monotonic())

    def on_crash(self, msg):
        try:
            event = json.loads(msg.data)
        except ValueError:
            return
        if event.get('direction', 1) < 0:
            self.publish('crash', 'ok', 'Hit while reversing: the camera faces forward, so it can\'t see what was hit.')
            return
        ok, why = self.start('crash')
        if not ok:
            self.get_logger().info(f'crash: not asking Claude ({why})')

    def on_describe(self, request, response):
        response.success, response.message = self.start('asked')
        return response

    # ---------- one look ----------
    def start(self, trigger):
        with self.lock:
            now = time.monotonic()
            if self.busy:
                return False, 'already looking'
            if now - self.last_ask < self.min_interval:
                return False, f'asked {now - self.last_ask:.0f} s ago: wait {self.min_interval:.0f} s between looks'
            self.busy, self.last_ask = True, now
        self.publish(trigger, 'asking', 'Looking…')
        self.frame, self.first_frame = None, 0.0
        self.camera_wanted = True
        self.manage_camera()
        threading.Thread(target=self.look, args=(trigger,), daemon=True).start()
        return True, 'asking Claude what the camera sees'

    def on_frame(self, msg):
        if self.first_frame == 0.0:
            self.first_frame = time.monotonic()
        self.frame = msg

    def look(self, trigger):
        image_path = ''
        try:
            start = time.monotonic()
            while not (self.frame is not None and time.monotonic() - self.first_frame >= self.settle_s):
                if time.monotonic() - start > self.camera_wait:
                    return self.publish(trigger, 'error', "The camera didn't start (is it switched off?).")
                time.sleep(0.1)
            jpeg = bytes(self.frame.data)
            self.stop_camera()
            rng, rng_t = self.front_range
            front = rng if rng is not None and time.monotonic() - rng_t < 2.0 else None
            image_path = self.save_image(jpeg)
            text = self.ask_claude(jpeg, prompt_for(trigger, front))
            self.publish(trigger, 'ok', text, image_path)
            self.set_status('connected')
        except Exception as e:  # noqa: BLE001 - report it, don't kill the node
            self.get_logger().error(f'look failed: {e!r}')
            self.publish(trigger, 'error', str(e), image_path)
        finally:
            self.stop_camera()
            self.busy = False
            if self.status != 'connected':
                self.check_soon()

    def stop_camera(self):
        self.camera_wanted = False  # manage_camera drops the subscription

    def manage_camera(self):
        if self.camera_wanted and self.image_sub is None:  # the camera starts on demand (camera_manager.py)
            self.image_sub = self.create_subscription(CompressedImage, '/left/preview/image/compressed',
                                                      self.on_frame, qos_profile_sensor_data)
        elif not self.camera_wanted and self.image_sub is not None:
            self.destroy_subscription(self.image_sub)
            self.image_sub = None

    def ask_claude(self, jpeg, question):
        import anthropic

        try:
            response = self.get_client().beta.messages.create(
                model=self.model,
                max_tokens=4000,
                output_config={'effort': self.effort},
                betas=['server-side-fallback-2026-07-01'],
                fallbacks='default',  # a declined request is re-run on Anthropic's pick of fallback model
                system=SYSTEM,
                messages=[{'role': 'user', 'content': [
                    {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                                 'data': base64.standard_b64encode(jpeg).decode()}},
                    {'type': 'text', 'text': question},
                ]}],
            )
        except anthropic.AuthenticationError:
            self.client = None
            raise RuntimeError(f'No valid Anthropic API key: put one in {self.key_file}') from None
        except anthropic.PermissionDeniedError as e:
            raise RuntimeError(f'The API key may not use {self.model}: {e.message}') from None
        except anthropic.RateLimitError:
            raise RuntimeError('Claude API rate limit: try again in a minute') from None
        except anthropic.APIStatusError as e:
            raise RuntimeError(f'Claude API error {e.status_code}: {e.message}') from None
        except anthropic.APIConnectionError:
            raise RuntimeError("Can't reach the Claude API (no internet?)") from None
        except Exception as e:  # noqa: BLE001 - e.g. no credentials found at all
            if 'api_key' in str(e) or 'credentials' in str(e).lower():
                self.client = None
                raise RuntimeError(f'No Anthropic API key: put one in {self.key_file}') from None
            raise
        if response.stop_reason == 'refusal':
            return 'Claude declined to describe this picture.'
        text = ' '.join(b.text for b in response.content if b.type == 'text').strip()
        u = response.usage
        self.get_logger().info(f'Claude ({self.model}): {u.input_tokens} in / {u.output_tokens} out tokens')
        return text or '(no answer)'

    # ---------- output ----------
    def save_image(self, jpeg):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, time.strftime('%Y%m%d_%H%M%S') + '.jpg')
        with open(path, 'wb') as f:
            f.write(jpeg)
        return path

    def publish(self, trigger, status, text, image=''):
        report = {'time': time.time(), 'trigger': trigger, 'status': status, 'text': text, 'image': image}
        self.report_pub.publish(String(data=json.dumps(report)))
        if status != 'asking':
            self.get_logger().info(f'{trigger}: {text}')
        if image and status in ('ok', 'error'):
            with open(os.path.splitext(image)[0] + '.json', 'w') as f:
                json.dump(report, f)


def main():
    rclpy.init()
    node = ClaudeVision()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
