<h1 align="center">🤖 2D LiDAR SLAM & Autonomous Navigation Simulator</h1>

<p align="center">
  A pure-Python 2D Mobile Robot & LiDAR SLAM simulator with real-time probabilistic Occupancy Grid Mapping, Differential Drive Kinematics, and Autonomous Frontier Exploration.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img alt="Robotics" src="https://img.shields.io/badge/Robotics-SLAM%20%7C%20ROS%20Concepts-red?style=for-the-badge&logo=ros&logoColor=white"/>
  <img alt="LiDAR" src="https://img.shields.io/badge/Sensor-360%C2%B0%20LiDAR-22D3EE?style=for-the-badge"/>
  <img alt="Pygame" src="https://img.shields.io/badge/GUI-Pygame-green?style=for-the-badge"/>
</p>

---

## 🚀 Overview

This project simulates the core mathematical components of **ROS / RViz SLAM** (Simultaneous Localization and Mapping) in a lightweight, pure-Python desktop application without requiring a heavy Ubuntu/ROS setup. 

The simulator features a dual-screen visualization:
* **Left Screen (Gazebo View):** The physical ground-truth world with obstacles, walls, moving differential-drive robot, and 140 simulated laser rays with Gaussian noise.
* **Right Screen (RViz View):** The estimated SLAM map built dynamically using **Log-Odds Bayesian updates** and **Bresenham raycasting**, tracking explored free space and obstacles.

---

## 🌟 Key Features

* **360° Planar LiDAR Simulation:** 140 raycast beams measuring obstacle ranges with configurable Gaussian measurement noise ($\sigma = 1.2$).
* **Log-Odds Occupancy Grid Mapping:** Probabilistic cell updates:
  $$\ell_{t}(m_i) = \ell_{t-1}(m_i) + \text{inverse\_sensor\_model}(z_t)$$
  Free cells along the ray are decremented, while the obstacle endpoint is marked occupied.
* **Autonomous Frontier Exploration:** Built-in reactive obstacle-avoidance and wall-following algorithms enabling the robot to map unknown rooms independently.
* **Manual Teleoperation:** Drive the robot manually using Keyboard Arrow keys or WASD.
* **Map Export:** Save the generated occupancy grid directly to an image (`slam_map_export.png`) with one keystroke.
* **Real-Time Telemetry HUD:** Displays explored territory percentage, beam counts, sensor range, and FPS.

---

## 🎮 Controls

| Key | Action |
| :--- | :--- |
| **Arrow Keys / W, A, S, D** | Manual Robot Teleoperation (Forward, Reverse, Turn) |
| **SPACE** | Toggle Autonomous Exploration Mode (Self-driving) |
| **L** | Toggle LiDAR Laser Beam Visualization |
| **S** | Export current SLAM Occupancy Map to PNG file |
| **R** | Reset robot position and wipe occupancy map |
| **ESC** | Exit Simulator |

---

## 🛠️ Installation & Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kiran10299/2D-LiDAR-SLAM-Simulator.git
   cd 2D-LiDAR-SLAM-Simulator
   ```

2. **Install dependencies:**
   ```bash
   pip install pygame numpy
   ```

3. **Launch the simulator:**
   ```bash
   python slam_simulator.py
   ```

---

## 📐 Mathematical Pipeline

1. **Forward Kinematics:**
   $$\theta_{t} = \theta_{t-1} + \omega \cdot \Delta t$$
   $$x_{t} = x_{t-1} + v \cdot \cos(\theta_t) \cdot \Delta t$$
   $$y_{t} = y_{t-1} + v \cdot \sin(\theta_t) \cdot \Delta t$$
2. **Raycasting & Spatial Discretization:**
   Ray-AABB fast stepping calculates distances to obstacles. Points are converted to grid indices $(cx, cy) = (\lfloor x / \text{res} \rfloor, \lfloor y / \text{res} \rfloor)$.
3. **Probability Recovery:**
   $$p(m_i) = 1 - \frac{1}{1 + e^{\ell_i}}$$

---

## 👨‍💻 Author

**Kiran Shivakumar**  
*Application Engineer | LabVIEW, Robotics (ROS) & Computer Vision*  
* 🌐 **Portfolio Website:** [kiran10299.github.io](https://kiran10299.github.io)  
* 💼 **LinkedIn:** [linkedin.com/in/kiranshivakumar](https://www.linkedin.com/in/kiranshivakumar)  
* 🐙 **GitHub:** [@kiran10299](https://github.com/kiran10299)
