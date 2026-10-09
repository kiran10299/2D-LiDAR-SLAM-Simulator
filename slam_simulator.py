import pygame
import math
import numpy as np
import random
import sys

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
PANE_WIDTH = 640
PANE_HEIGHT = 650
HUD_HEIGHT = 70

# Colors
COLOR_BG_SIM = (15, 23, 42)         # Deep slate (Gazebo world)
COLOR_BG_SLAM = (30, 41, 59)        # Dark navy (RViz SLAM map)
COLOR_OBSTACLE = (51, 65, 85)       # Obstacle color
COLOR_OBSTACLE_BORDER = (148, 163, 184)
COLOR_ROBOT = (56, 189, 248)        # Electric cyan
COLOR_ROBOT_HEADING = (244, 114, 182)
COLOR_RAY = (34, 211, 238, 45)      # Faint laser beam
COLOR_HIT = (239, 68, 68)           # Red hit marker
COLOR_TRAJECTORY = (34, 197, 94)    # Green path trail
COLOR_TEXT = (241, 245, 249)
COLOR_ACCENT = (167, 139, 250)

# SLAM Map Constants
GRID_RES = 4                        # Pixels per grid cell
GRID_W = PANE_WIDTH // GRID_RES
GRID_H = PANE_HEIGHT // GRID_RES
L_FREE = -0.4                       # Log-odds update for free cells
L_OCC = 0.85                        # Log-odds update for occupied cells
L_MIN = -5.0
L_MAX = 5.0

class Environment:
    """Simulated 2D Physical World (Obstacles, Walls)"""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.obstacles = []
        self._build_warehouse_map()

    def _build_warehouse_map(self):
        # Outer boundary walls
        border_thick = 15
        self.obstacles.append(pygame.Rect(0, 0, self.width, border_thick))
        self.obstacles.append(pygame.Rect(0, self.height - border_thick, self.width, border_thick))
        self.obstacles.append(pygame.Rect(0, 0, border_thick, self.height))
        self.obstacles.append(pygame.Rect(self.width - border_thick, 0, border_thick, self.height))

        # Warehouse interior obstacles & rooms
        rooms = [
            # Central pillar cluster
            pygame.Rect(200, 150, 40, 160),
            pygame.Rect(400, 150, 40, 160),
            pygame.Rect(200, 310, 240, 35),
            
            # Storage crates & pallets
            pygame.Rect(70, 70, 80, 50),
            pygame.Rect(490, 70, 80, 80),
            pygame.Rect(70, 480, 110, 80),
            pygame.Rect(460, 460, 100, 100),
            
            # Corridors / dividing barriers
            pygame.Rect(240, 440, 160, 30),
            pygame.Rect(300, 70, 35, 100),
        ]
        self.obstacles.extend(rooms)

    def is_collision(self, x, y, radius):
        robot_rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
        for obs in self.obstacles:
            if obs.colliderect(robot_rect):
                return True
        return False

    def raycast(self, start_x, start_y, angle, max_range):
        """Ray-AABB fast intersection testing"""
        dx = math.cos(angle)
        dy = math.sin(angle)
        closest_dist = max_range
        hit_pos = (start_x + dx * max_range, start_y + dy * max_range)

        # Step along ray
        step = 4
        dist = 0
        while dist < max_range:
            curr_x = start_x + dx * dist
            curr_y = start_y + dy * dist

            # Check boundary limits
            if curr_x <= 0 or curr_x >= self.width or curr_y <= 0 or curr_y >= self.height:
                closest_dist = dist
                hit_pos = (curr_x, curr_y)
                break

            # Check obstacle bounding boxes
            hit = False
            for obs in self.obstacles:
                if obs.collidepoint(curr_x, curr_y):
                    closest_dist = dist
                    hit_pos = (curr_x, curr_y)
                    hit = True
                    break
            if hit:
                break
            dist += step

        return closest_dist, hit_pos

class LidarSensor:
    """Simulated 360-Degree Planar LiDAR Sensor"""
    def __init__(self, num_beams=120, max_range=240, noise_std=1.2):
        self.num_beams = num_beams
        self.max_range = max_range
        self.noise_std = noise_std
        self.angles = [2 * math.pi * i / num_beams for i in range(num_beams)]

    def scan(self, env, robot_x, robot_y, robot_theta):
        scan_data = []
        for rel_angle in self.angles:
            global_angle = robot_theta + rel_angle
            dist, hit_pt = env.raycast(robot_x, robot_y, global_angle, self.max_range)
            # Add Gaussian noise to emulate real sensor measurements
            noisy_dist = max(0, dist + random.gauss(0, self.noise_std))
            noisy_hit_x = robot_x + math.cos(global_angle) * noisy_dist
            noisy_hit_y = robot_y + math.sin(global_angle) * noisy_dist
            
            scan_data.append({
                'angle': global_angle,
                'dist': noisy_dist,
                'max_range': dist >= self.max_range,
                'hit_pos': (noisy_hit_x, noisy_hit_y)
            })
        return scan_data

class OccupancyGridMap:
    """Log-Odds Occupancy Grid Mapping SLAM Component"""
    def __init__(self, grid_w, grid_h, res):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.res = res
        # Log-odds representation: 0 = prior unknown (p=0.5)
        self.log_odds = np.zeros((grid_w, grid_h), dtype=np.float32)
        self.surface = pygame.Surface((grid_w * res, grid_h * res))

    def update_scan(self, robot_x, robot_y, scan_data):
        rx = int(robot_x / self.res)
        ry = int(robot_y / self.res)

        for beam in scan_data:
            hx = int(beam['hit_pos'][0] / self.res)
            hy = int(beam['hit_pos'][1] / self.res)

            # Bresenham's line algorithm for free cells
            cells = self._bresenham_line(rx, ry, hx, hy)
            for cx, cy in cells[:-1]:
                if 0 <= cx < self.grid_w and 0 <= cy < self.grid_h:
                    self.log_odds[cx, cy] = max(L_MIN, self.log_odds[cx, cy] + L_FREE)

            # Endpoint is occupied (if within range)
            if not beam['max_range'] and 0 <= hx < self.grid_w and 0 <= hy < self.grid_h:
                self.log_odds[hx, hy] = min(L_MAX, self.log_odds[hx, hy] + L_OCC)

    def _bresenham_line(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    def render_to_surface(self):
        # Convert log odds to probability: p = 1 - 1 / (1 + exp(l))
        probs = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))
        # Map probability to grayscale color:
        # Unknown (~0.5) -> Gray (100)
        # Free (<0.5)    -> Light White/Cyan (215)
        # Occupied (>0.5)-> Dark Obsidian (20)
        img_array = np.zeros((self.grid_w, self.grid_h, 3), dtype=np.uint8)
        
        # Color mapping
        free_mask = probs < 0.4
        occ_mask = probs > 0.65
        unknown_mask = ~free_mask & ~occ_mask

        img_array[unknown_mask] = [40, 50, 70]     # Unknown
        img_array[free_mask] = [210, 225, 240]      # Free space
        img_array[occ_mask] = [15, 20, 30]         # Occupied obstacle

        surf = pygame.surfarray.make_surface(img_array)
        scaled_surf = pygame.transform.scale(surf, (self.grid_w * self.res, self.grid_h * self.res))
        return scaled_surf

    def get_explored_percentage(self):
        total = self.grid_w * self.grid_h
        known = np.sum((self.log_odds < -0.2) | (self.log_odds > 0.5))
        return (known / total) * 100.0

class MobileRobot:
    """Differential Drive Mobile Robot with Odometry & Autonomous Navigation"""
    def __init__(self, x=320, y=360, radius=12):
        self.x = x
        self.y = y
        self.radius = radius
        self.theta = 0.0          # Heading angle in radians
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.path = [(x, y)]
        self.auto_explore = False
        self.wander_cooldown = 0
        self.wander_turn = 0.0

    def update(self, env):
        # Autonomous Frontier / Obstacle-Avoidance Exploration Logic
        if self.auto_explore:
            self._autonomous_wander(env)

        # Kinematic Motion Update
        self.theta += self.angular_speed
        new_x = self.x + math.cos(self.theta) * self.linear_speed
        new_y = self.y + math.sin(self.theta) * self.linear_speed

        # Collision verification against environment boundaries & obstacles
        if not env.is_collision(new_x, new_y, self.radius):
            self.x = new_x
            self.y = new_y
            if len(self.path) == 0 or math.hypot(self.x - self.path[-1][0], self.y - self.path[-1][1]) > 5:
                self.path.append((self.x, self.y))
        else:
            # Rebound slightly if colliding in autonomous mode
            if self.auto_explore:
                self.theta += math.pi / 2.0
                self.linear_speed = 0.5

    def _autonomous_wander(self, env):
        # Raycast 3 forward probes: left (45 deg), center (0 deg), right (-45 deg)
        d_center, _ = env.raycast(self.x, self.y, self.theta, 70)
        d_left, _ = env.raycast(self.x, self.y, self.theta + math.pi / 4, 60)
        d_right, _ = env.raycast(self.x, self.y, self.theta - math.pi / 4, 60)

        safe_dist = 45.0
        if d_center < safe_dist or d_left < safe_dist or d_right < safe_dist:
            self.linear_speed = 0.6
            if d_left < d_right:
                self.angular_speed = -0.08  # Turn right
            else:
                self.angular_speed = 0.08   # Turn left
        else:
            self.linear_speed = 2.4
            # Gentle random meandering to explore new frontiers
            self.wander_cooldown -= 1
            if self.wander_cooldown <= 0:
                self.wander_turn = random.uniform(-0.025, 0.025)
                self.wander_cooldown = random.randint(30, 70)
            self.angular_speed = self.wander_turn

class SlamSimulatorApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("2D LiDAR SLAM & Autonomous Navigation Simulator | Kiran Shivakumar")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 14)
        self.font_bold = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 18, bold=True)

        self.env = Environment(PANE_WIDTH, PANE_HEIGHT)
        self.lidar = LidarSensor(num_beams=140, max_range=230)
        self.slam_map = OccupancyGridMap(GRID_W, GRID_H, GRID_RES)
        self.robot = MobileRobot(x=140, y=260)
        self.running = True
        self.show_rays = True

    def run(self):
        while self.running:
            self.clock.tick(45) # 45 FPS simulation rate
            self._handle_events()
            
            # Step simulation
            self.robot.update(self.env)
            scan = self.lidar.scan(self.env, self.robot.x, self.robot.y, self.robot.theta)
            self.slam_map.update_scan(self.robot.x, self.robot.y, scan)

            # Draw everything
            self._render(scan)
            pygame.display.flip()

        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.robot.auto_explore = not self.robot.auto_explore
                    self.robot.linear_speed = 0
                    self.robot.angular_speed = 0
                elif event.key == pygame.K_r:
                    # Reset
                    self.slam_map = OccupancyGridMap(GRID_W, GRID_H, GRID_RES)
                    self.robot = MobileRobot(x=140, y=260)
                elif event.key == pygame.K_l:
                    self.show_rays = not self.show_rays
                elif event.key == pygame.K_s:
                    # Save SLAM Map to disk
                    saved_surface = self.slam_map.render_to_surface()
                    pygame.image.save(saved_surface, "slam_map_export.png")
                    print("SLAM Occupancy Map saved to slam_map_export.png")

        # Manual control keys if not in auto-explore mode
        if not self.robot.auto_explore:
            keys = pygame.key.get_pressed()
            # Linear speed
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                self.robot.linear_speed = 2.5
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                self.robot.linear_speed = -1.8
            else:
                self.robot.linear_speed = 0.0

            # Angular speed
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.robot.angular_speed = -0.06
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.robot.angular_speed = 0.06
            else:
                self.robot.angular_speed = 0.0

    def _render(self, scan):
        self.screen.fill((10, 15, 26))

        # -----------------------------------------------
        # 1. TOP HEADER / HUD BAR
        # -----------------------------------------------
        hud_rect = pygame.Rect(0, 0, WINDOW_WIDTH, HUD_HEIGHT)
        pygame.draw.rect(self.screen, (15, 23, 42), hud_rect)
        pygame.draw.line(self.screen, (30, 41, 59), (0, HUD_HEIGHT), (WINDOW_WIDTH, HUD_HEIGHT), 2)

        # Title
        t1 = self.font_title.render("2D LIDAR SLAM & MOBILE ROBOT SIMULATOR", True, COLOR_TEXT)
        self.screen.blit(t1, (20, 12))

        # Status indicators
        mode_text = "[AUTO EXPLORATION: ON]" if self.robot.auto_explore else "[MANUAL DRIVE: ON]"
        mode_color = (34, 197, 94) if self.robot.auto_explore else (56, 189, 248)
        m_surf = self.font_bold.render(mode_text, True, mode_color)
        self.screen.blit(m_surf, (20, 40))

        # Telemetry metrics
        fps = int(self.clock.get_fps())
        exp_pct = self.slam_map.get_explored_percentage()
        info_text = f"Explored: {exp_pct:.1f}% | Beams: {self.lidar.num_beams} | Range: {self.lidar.max_range}px | FPS: {fps}"
        info_surf = self.font.render(info_text, True, (148, 163, 184))
        self.screen.blit(info_surf, (280, 42))

        # Controls instructions
        ctrl_text = "Controls: [Arrows/WASD] Drive | [SPACE] Toggle Auto | [L] Toggle Rays | [S] Save Map | [R] Reset"
        c_surf = self.font.render(ctrl_text, True, COLOR_ACCENT)
        self.screen.blit(c_surf, (WINDOW_WIDTH - c_surf.get_width() - 20, 16))

        # -----------------------------------------------
        # 2. LEFT PANE: SIMULATED ENVIRONMENT (Gazebo-style)
        # -----------------------------------------------
        sim_surface = pygame.Surface((PANE_WIDTH, PANE_HEIGHT))
        sim_surface.fill(COLOR_BG_SIM)

        # Draw grid lines
        for x in range(0, PANE_WIDTH, 40):
            pygame.draw.line(sim_surface, (23, 33, 56), (x, 0), (x, PANE_HEIGHT))
        for y in range(0, PANE_HEIGHT, 40):
            pygame.draw.line(sim_surface, (23, 33, 56), (0, y), (PANE_WIDTH, y))

        # Draw Obstacles
        for obs in self.env.obstacles:
            pygame.draw.rect(sim_surface, COLOR_OBSTACLE, obs)
            pygame.draw.rect(sim_surface, COLOR_OBSTACLE_BORDER, obs, 1)

        # Draw LiDAR Laser Beams (Optional toggle)
        if self.show_rays:
            ray_surface = pygame.Surface((PANE_WIDTH, PANE_HEIGHT), pygame.SRCALPHA)
            for beam in scan:
                pygame.draw.line(ray_surface, (34, 211, 238, 25), (self.robot.x, self.robot.y), beam['hit_pos'], 1)
                if not beam['max_range']:
                    pygame.draw.circle(ray_surface, (239, 68, 68, 160), (int(beam['hit_pos'][0]), int(beam['hit_pos'][1])), 2)
            sim_surface.blit(ray_surface, (0, 0))

        # Draw Robot Path Trajectory
        if len(self.robot.path) > 1:
            pygame.draw.lines(sim_surface, COLOR_TRAJECTORY, False, self.robot.path, 2)

        # Draw Robot Body
        rx, ry = int(self.robot.x), int(self.robot.y)
        pygame.draw.circle(sim_surface, COLOR_ROBOT, (rx, ry), self.robot.radius)
        pygame.draw.circle(sim_surface, (255, 255, 255), (rx, ry), self.robot.radius, 2)

        # Heading vector
        hx = rx + math.cos(self.robot.theta) * (self.robot.radius + 7)
        hy = ry + math.sin(self.robot.theta) * (self.robot.radius + 7)
        pygame.draw.line(sim_surface, COLOR_ROBOT_HEADING, (rx, ry), (hx, hy), 3)

        # Label Left Pane
        lbl_left = self.font_bold.render("Physical World (Simulated Gazebo)", True, (255, 255, 255))
        sim_surface.blit(lbl_left, (15, 15))

        self.screen.blit(sim_surface, (0, HUD_HEIGHT))

        # -----------------------------------------------
        # 3. RIGHT PANE: SLAM OCCUPANCY GRID (RViz-style)
        # -----------------------------------------------
        slam_surface = pygame.Surface((PANE_WIDTH, PANE_HEIGHT))
        slam_surface.fill(COLOR_BG_SLAM)

        # Blit probability occupancy map
        occupancy_map_surf = self.slam_map.render_to_surface()
        slam_surface.blit(occupancy_map_surf, (0, 0))

        # Overlay robot path in SLAM view
        if len(self.robot.path) > 1:
            pygame.draw.lines(slam_surface, (34, 197, 94), False, self.robot.path, 2)

        # Draw Robot Pose in SLAM view
        pygame.draw.circle(slam_surface, COLOR_ROBOT, (rx, ry), self.robot.radius - 2)
        pygame.draw.circle(slam_surface, (255, 255, 255), (rx, ry), self.robot.radius - 2, 2)
        pygame.draw.line(slam_surface, COLOR_ROBOT_HEADING, (rx, ry), (hx, hy), 2)

        # Label Right Pane
        lbl_right = self.font_bold.render("Occupancy Grid SLAM (Estimated Map)", True, (255, 255, 255))
        slam_surface.blit(lbl_right, (15, 15))

        # Legend
        leg_y = PANE_HEIGHT - 35
        pygame.draw.rect(slam_surface, (40, 50, 70), (15, leg_y, 14, 14))
        l1 = self.font.render("Unknown", True, (200, 210, 225))
        slam_surface.blit(l1, (35, leg_y))

        pygame.draw.rect(slam_surface, (210, 225, 240), (120, leg_y, 14, 14))
        l2 = self.font.render("Free", True, (200, 210, 225))
        slam_surface.blit(l2, (140, leg_y))

        pygame.draw.rect(slam_surface, (15, 20, 30), (200, leg_y, 14, 14))
        pygame.draw.rect(slam_surface, (100, 115, 130), (200, leg_y, 14, 14), 1)
        l3 = self.font.render("Occupied", True, (200, 210, 225))
        slam_surface.blit(l3, (220, leg_y))

        self.screen.blit(slam_surface, (PANE_WIDTH, HUD_HEIGHT))

        # Center dividing line
        pygame.draw.line(self.screen, (56, 189, 248), (PANE_WIDTH, HUD_HEIGHT), (PANE_WIDTH, WINDOW_HEIGHT), 2)

if __name__ == "__main__":
    app = SlamSimulatorApp()
    app.run()
