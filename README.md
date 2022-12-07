# SmartCarts
This is the code base for the project 2054, SmartCarts. It contains code for both the simulation and the real robot. The simulation contains a follower and leader robot, where the leader robot follows a preset path and the follower robot tried to follow the path by visually detecting the position of the leader robot. The real robot is our physical implementation and interfaces with physical motor drivers, encoders and a stereovision camera mounted on a platform. This has been tested and works under limited conditions.

*For more information, please read the document "2054 - SmartCarts - Final Report.pdf"*

## Development Environment
This project uses Ubuntu 18.04, Bionic Beaver. The ROS version we use is ROS 1 - Melodic Morenia. The simulation code is mostly in Python 2.7/3.7 and some arduino code is in C++. 

The simulation environment uses Gazebo and Rviz.

We will now explain the usage of both the simulation and physical package separately. 

## How to use the simulation package <smartcarts></smartcarts>
The simulation package is called *smartcarts* and the files/folders in this repository make up the ROS package itself. See this tutorial for more information if you are unfamiliar with ROS: http://wiki.ros.org/ROS/Tutorials/CreatingPackage

First make a ROS Catkin Workspace (ROS Tutorial: http://wiki.ros.org/catkin/Tutorials/create_a_workspace)

The file structure should be like this:
```
/catkin_workspace
    .catkin_workspace
    /build
    /devel
    /src
        CMakeLists.txt
        /SmartCarts
            CMakeLists.txt    
            package.xml
            README.md
            /nodes
            /launch
            ...
```
Open up a termianl in your /catkin_workspace and type in the following command:
```
$ catkin_make
```
This will compile any new changes in your workspace. Feel free to name your workspace any other name other than catkin_workspace. 

In the same terminal, type 
```
$ echo "alias youralias='source ~/catkin_workspace/devel/setup.bash'" >> ~/.bashrc
```
This will modify your ~/.bashrc file to concatenate the echoded line. Once you have done this, in order to register your ROS package in each terminal, you simply navigate into the /SmartCarts folder and type into the terminal:
```
$ youralias
```
To see if this has worked, check with the command ```$ rospack find smartcarts```
To launch the simulation, make sure to have Gazebo and rviz on your computer. Use the following command:
```
$ roslaunch smartcarts my_launch.launch
```
What this is doing: my_launch.launch is the main launchfile that starts both the physics engine, Gazebo and the visualization tool, rviz. The follower and leader robot are also setup by starting up their respective nodes (nodes are essentially Python/C++ files in the /node folder that gives the robot various types of functions like target tracking, motion control, etc). You should see an rviz application window like so:
![](https://i.imgur.com/HxtvKs7.png)
Right now, the Gazebo simulation GUI is turned off. To turn it on, go to /SmartCarts/launch/my_launch.launch and change the argument for ```gui``` to ```true``
```
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
      <env name="GAZEBO_RESOURCE_PATH" value="$(find smartcarts)"/>
      <env name="GAZEBO_MODEL_PATH" value="$(find smartcarts)/models"/>
      <arg name="world_name" value="$(find smartcarts)/worlds/smartcarts.world"/>
      <arg name="gui" value="false"/>
  </include>
```

You should then be able to see something like this:
![](https://i.imgur.com/M8OjYEe.png)

Bear in mind that running both Gazebo and Rviz at the same time takes up alot of computing power so it is best to run only one of the two at any point in time. (Of course if your computer is very powerful then it's no problem!)

rviz shows a simplified visualization- only displaying the relevant tfs and Gazebo shows the entire Gazebo world, including the two robots we created. The robot "holding" the red ball is the leader and the gray robot is the follower. The rviz startup settings can be changed in /rviz/follower_pathview.rviz

### Files
```
/SmartCarts
    <!-- The following are Gazebo Simulation specific files -->
    /media
    /models
    /urdf
    /worlds
    <!--  The following are functional ROS files -->
    /launch
    /node
        /data
        /preset_poses
    /msg
```
If the ```ball_waypoint_listerner.py``` node is running, then after the simulation is shutdown, the data will be saved in the folder /node/data, presently as log.txt. You can also view pictures of the results of our test runs with different leader preset paths. 

The preset paths for the leader robot (there is a curved sin45 path, straight paths with heading changes) are stored as .csv files in /node/preset_poses. To load in paths, change the value for the argument ```waypoints_file```. This is the line as it is currently:

```
<launch>
  <arg name="waypoints_file" value="$(find smartcarts)/node/preset_poses/sine_L4A1.csv"/> <!-- load in preset poses for leader -->
  ...
```
You also have the option to set custom waypoints *in* the rviz GUI by clicking on the position in the screen. To use this option, simply change the line above to:
```
<arg name="waypoint_file" value=""/>
```
An important point to take note of are the custom messages that we have created for our specific use case. These are stored in /msg folder and are also added to CMakeLists.txt and the package.xml file is also modified to reflect custom messages being used. If you have compiled the package correctly, typing in the command in terminal ```$ rosmsg info Float32Stamped.msg``` (this is one of the custom messages) should yield the following result:
```
[smartcarts/Float32Stamped]:
std_msgs/Header header
  uint32 seq
  time stamp
  string frame_id
float32 data
```

---
Team 2054, ENPH 479, 2020-2021

---
Team 2252, ENPH 479

Code review: 
