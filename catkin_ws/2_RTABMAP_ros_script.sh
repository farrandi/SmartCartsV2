#!/bin/bash

echo 'starting RTABMAP_ros package...'

roslaunch rtabmap_ros rtabmap.launch rviz:=false rtabmapviz:=true \
    depth_topic:=/camera/aligned_depth_to_color/image_raw \
    rgb_topic:=/camera/color/image_raw \
    camera_info_topic:=/camera/color/camera_info \
    approx_sync:=false
