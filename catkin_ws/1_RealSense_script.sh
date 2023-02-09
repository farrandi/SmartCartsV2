#!/bin/bash

echo 'starting RealSense Camera D435 package...'

cd ~/catkin_ws
source devel/setup.bash

roslaunch realsense2_camera rs_rgbd.launch align_depth:=true
