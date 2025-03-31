#!/bin/bash

# 更新包列表
echo "Updating package list..."
sudo apt update

# 安装 MoveIt 和相关包
echo "Installing MoveIt packages..."
sudo apt install -y ros-melodic-object-recognition-msgs 
sudo apt install -y ros-melodic-moveit-*

# 安装 Trac-IK
echo "Installing Trac-IK..."
sudo apt install -y ros-melodic-trac-ik

# 安装 Joint Trajectory Controller
echo "Installing Joint Trajectory Controller..."
sudo apt install -y ros-melodic-joint-trajectory-controller

# 安装 tf 和其他相关包
echo "Installing tf and other related packages..."
sudo apt-get install -y ros-melodic-tf-*

# 安装 Gazebo ROS 插件
echo "Installing Gazebo ROS..."
sudo apt-get install -y ros-melodic-gazebo-ros

# 安装 Robot State Publisher
echo "Installing Robot State Publisher..."
sudo apt-get install -y ros-melodic-robot-state-publisher

# 安装 RQT 相关包
echo "Installing RQT packages..."
sudo apt-get install -y ros-melodic-rqt-*

# 安装 Joint Trajectory Controller Debugging Symbols
echo "Installing Joint Trajectory Controller Debugging Symbols..."
sudo apt install -y ros-melodic-joint-trajectory-controller-dbgsym 

# 安装 ROS Control
echo "Installing ROS Control..."
sudo apt install -y ros-melodic-ros-control

echo "Installation completed!"
