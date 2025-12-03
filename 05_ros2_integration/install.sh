#!/bin/bash
# ROS2视觉系统一键安装脚本

set -e

echo "========================================"
echo "ROS2视觉系统安装脚本"
echo "========================================"

# 检测ROS2版本
if [ -z "$ROS_DISTRO" ]; then
    echo "错误: 未检测到ROS2环境"
    echo "请先安装ROS2并source setup.bash"
    exit 1
fi

echo "检测到ROS2版本: $ROS_DISTRO"

# 安装依赖
echo ""
echo "安装系统依赖..."
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-opencv \
    ros-$ROS_DISTRO-cv-bridge \
    ros-$ROS_DISTRO-vision-msgs \
    ros-$ROS_DISTRO-image-transport \
    ros-$ROS_DISTRO-rqt-image-view

# 安装Python依赖
echo ""
echo "安装Python依赖..."
pip3 install \
    ultralytics \
    opencv-python \
    numpy

# 构建工作空间
echo ""
echo "构建ROS2工作空间..."

cd "$(dirname "$0")"

# 创建工作空间(如果不存在)
if [ ! -d "src" ]; then
    echo "错误: 请在ROS2工作空间根目录运行此脚本"
    exit 1
fi

# 构建
colcon build --symlink-install

echo ""
echo "========================================"
echo "安装完成!"
echo "========================================"
echo ""
echo "使用方法:"
echo "  1. source install/setup.bash"
echo "  2. ros2 launch launch/grasp_system.launch.py"
echo ""
echo "测试相机:"
echo "  ros2 launch launch/test_camera.launch.py"
echo ""
echo "仅视觉系统:"
echo "  ros2 launch launch/vision_only.launch.py"
echo ""
