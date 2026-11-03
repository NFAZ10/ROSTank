"""Frontier exploration: drives Nav2 to the edges of the map being built until none are left.

Run by nav_manager (/robot/start_exploring) next to SLAM + Nav2. Each round:
  1. frontiers = free cells of /map next to unknown ones, clustered; small ones and cells
     too close to walls for the robot are skipped
  2. best = biggest and nearest (size_weight * size - distance)
  3. NavigateToPose to it; switch target if it got explored on the way; blacklist spots
     where a goal fails or the robot stops making progress
Done when no frontiers are left: drives back to where it started and exits 0.
Status (latched): /explore/status.
"""
import math
import time

import cv2
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener


class Explorer(Node):
    def __init__(self):
        super().__init__('explorer')
        p = lambda name, default: self.declare_parameter(name, default).value
        self.min_frontier = p('min_frontier_size', 0.5)   # m of frontier edge worth visiting
        # Goal this far from any wall: beyond Nav2's inflation (0.25 m), or the planner refuses it.
        self.clearance = p('robot_clearance', 0.4)
        self.size_weight = p('size_weight', 0.5)         # metres of detour one metre of frontier is worth
        self.commit_time = p('commit_time', 15.0)        # s before switching away from a target
        self.blacklist_radius = p('blacklist_radius', 0.6)
        self.progress_timeout = p('progress_timeout', 30.0)  # s without moving 0.2 m -> give up on goal
        self.return_home = p('return_home', True)
        self.empty_rounds_needed = p('empty_rounds', 3)  # map must stay frontier-free this many checks

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.map = None
        self.create_subscription(OccupancyGrid, '/map', lambda m: setattr(self, 'map', m), latched)
        self.status_pub = self.create_publisher(String, '/explore/status', latched)
        self.nav = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.tf = Buffer()
        TransformListener(self.tf, self)

        self.blacklist = []
        self.goal = None          # (x, y) being driven to
        self.goal_handle = None
        self.goal_result = None
        self.progress = None      # (time, x, y) of the last real movement
        self.home = None
        self.empty_rounds = 0
        self.visited = 0
        self.state = 'starting'
        self.say('starting: waiting for the map and navigation')
        self.create_timer(1.0, self.tick)

    def say(self, text):
        self.status_pub.publish(String(data=text))
        self.get_logger().info(text)

    def robot_xy(self):
        try:
            t = self.tf.lookup_transform('map', 'base_link', Time()).transform.translation
            return t.x, t.y
        except Exception:  # noqa: BLE001 - not localized yet
            return None

    # ---------- frontiers ----------
    def frontiers(self):
        m = self.map
        w, h, res = m.info.width, m.info.height, m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        grid = np.array(m.data, np.int16).reshape(h, w)
        free, unknown, occupied = grid == 0, grid < 0, grid > 50
        kernel = np.ones((3, 3), np.uint8)
        edge = free & (cv2.dilate(unknown.astype(np.uint8), kernel) > 0)
        # Room for the robot: distance to the nearest occupied cell, in metres.
        clear = cv2.distanceTransform((~occupied).astype(np.uint8), cv2.DIST_L2, 5) * res
        n, labels, stats, _ = cv2.connectedComponentsWithStats(edge.astype(np.uint8), connectivity=8)
        out = []
        for i in range(1, n):
            size_m = stats[i, cv2.CC_STAT_AREA] * res
            if size_m < self.min_frontier:
                continue
            ys, xs = np.nonzero((labels == i) & (clear >= self.clearance))
            if len(xs) == 0:
                continue
            cx, cy = xs.mean(), ys.mean()
            k = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))  # reachable cell nearest the middle
            x, y = ox + (xs[k] + 0.5) * res, oy + (ys[k] + 0.5) * res
            if any(math.hypot(x - bx, y - by) < self.blacklist_radius for bx, by in self.blacklist):
                continue
            out.append((x, y, size_m))
        return out

    def still_frontier(self, goal):
        # Generous: on a real map the frontier edge shifts a little with every scan.
        return any(math.hypot(goal[0] - x, goal[1] - y) < 0.8 for x, y, _ in self.frontiers())

    # ---------- goals ----------
    def send(self, x, y, label):
        here = self.robot_xy()
        yaw = math.atan2(y - here[1], x - here[0]) if here else 0.0
        g = NavigateToPose.Goal()
        g.pose.header.frame_id = 'map'
        g.pose.header.stamp = self.get_clock().now().to_msg()
        g.pose.pose.position.x, g.pose.pose.position.y = x, y
        g.pose.pose.orientation.z, g.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.goal, self.goal_result, self.goal_handle = (x, y), None, None
        self.goal_time = time.monotonic()
        self.progress = (time.monotonic(), *(here or (x, y)))
        self.nav.send_goal_async(g).add_done_callback(self.on_accepted)
        self.say(label)

    def on_accepted(self, future):
        handle = future.result()
        if not handle.accepted:
            self.goal_result = GoalStatus.STATUS_ABORTED
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(
            lambda f, h=handle: setattr(self, 'goal_result', f.result().status) if h is self.goal_handle else None)

    def cancel(self):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        self.goal, self.goal_handle, self.goal_result = None, None, None

    # ---------- main loop ----------
    def tick(self):
        here = self.robot_xy()
        if self.map is None or here is None or not self.nav.server_is_ready():
            return
        if self.home is None:
            self.home = here
            self.state = 'exploring'

        if self.goal is not None:
            if self.goal_result is not None:  # finished, one way or another
                if self.state == 'returning':
                    self.finish('done: map complete, back at the start. Save it now.')
                    return
                if self.goal_result == GoalStatus.STATUS_SUCCEEDED:
                    self.visited += 1
                else:
                    self.blacklist.append(self.goal)
                    self.get_logger().warn(f'could not reach {self.goal}: skipping that spot')
                self.goal = None
            else:
                t0, px, py = self.progress
                if math.hypot(here[0] - px, here[1] - py) > 0.2:
                    self.progress = (time.monotonic(), *here)
                elif time.monotonic() - t0 > self.progress_timeout:
                    self.get_logger().warn(f'no progress towards {self.goal}: skipping that spot')
                    self.blacklist.append(self.goal)
                    self.cancel()
                elif (self.state == 'exploring' and time.monotonic() - self.goal_time > self.commit_time
                      and int(time.monotonic()) % 5 == 0 and not self.still_frontier(self.goal)):
                    self.cancel()  # mapped on the way there: pick the next one
                return
        if self.state == 'returning':
            return

        found = self.frontiers()
        if not found:
            self.empty_rounds += 1
            if self.empty_rounds < self.empty_rounds_needed:
                self.say('checking for anything left to explore…')
                return
            if self.return_home and math.hypot(here[0] - self.home[0], here[1] - self.home[1]) > 0.3:
                self.state = 'returning'
                self.send(*self.home, 'done exploring: driving back to the start')
            else:
                self.finish('done: map complete. Save it now.')
            return
        self.empty_rounds = 0
        x, y, size = min(found, key=lambda f: math.hypot(f[0] - here[0], f[1] - here[1])
                         - self.size_weight * f[2])
        self.send(x, y, f'exploring: {len(found)} frontier{"s" * (len(found) != 1)} left, '
                        f'heading to ({x:.1f}, {y:.1f}), {self.visited} reached so far')

    def finish(self, text):
        self.say(text)
        self.state = 'done'
        raise SystemExit(0)


def main():
    rclpy.init()
    node = Explorer()
    code = 0
    try:
        rclpy.spin(node)
    except SystemExit as e:
        code = e.code or 0
    except KeyboardInterrupt:
        try:
            node.cancel()
            node.say('stopped')
        except Exception:  # noqa: BLE001 - ROS may already be shutting down (Ctrl-C reaches both)
            pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    raise SystemExit(code)


if __name__ == '__main__':
    main()
