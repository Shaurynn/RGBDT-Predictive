#!/bin/bash

# ==============================================================================
# UAV Multi-Spectral Sensor Master Boot Sequence
# ==============================================================================

# --- FAILSAFE: GRACEFUL SHUTDOWN ---
# This trap catches Ctrl+C (SIGINT) and cleanly kills all background processes
cleanup() {
    echo -e "\n[WARN] Termination signal received."
    echo "[INFO] Shutting down UAV Multi-Spectral Sensor Pipeline..."
    # kill 0 sends the kill signal to all processes in the current process group
    kill 0
    exit 0
}
trap cleanup SIGINT

echo "[INFO] Initiating 'One-Click' Master Boot Sequence..."

# --- 1. VISUAL PAYLOAD (RGB-D) ---
echo "[INFO] Starting HP-60C RGB-D Payload..."
# Wrapping in ( ) runs it in a subshell, ensuring the directory change doesn't break the main script
(
    cd ~/ascam_ros2_ws || exit
    source install/setup.bash
    ros2 launch ascamera ascamera_node.launch.py
) &

# Brief pause to allow the hardware buses to initialize
sleep 2

# --- 2. THERMAL PAYLOAD ---
echo "[INFO] Starting Thermal Payload..."
(
    cd ~/Code/thermalcam_ros2_ws || exit
    source install/setup.bash
    # Note: Ensure this matches your exact thermal run/launch command
    ros2 run thermal_driver thermal_node 
) &

# Brief pause to ensure topics are actively publishing before the lock engages
sleep 2

# --- 3. THE FUSION ENGINE ---
echo "[INFO] Engaging RGB-D-T Fusion Engine & Presentation UI..."
(
    cd ~/Code/thermalcam_ros2_ws || exit
    source install/setup.bash
    ros2 run uav_sensor_fusion rgbdt_production_node
) &

echo "======================================================================"
echo "[SUCCESS] ALL SYSTEMS GO. Triple-Lock engaged."
echo "[ACTION]  Press Ctrl+C in this terminal to safely terminate all nodes."
echo "======================================================================"

# Wait command pauses the script here, keeping the background processes alive until Ctrl+C is pressed
wait