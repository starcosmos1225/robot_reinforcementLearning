# robot_reinforcementLearning
## robottest 
This is a ros-package for husky.
<br>It runs under the gazebo simulation environment. The world randomly generates obstacles, husky starting point and target address. The purpose of training is to make husky move from starting point to ending point and avoid obstacles.
<br>I run in ros-kinetic. 
<br>husky install by:sudo apt-get install ros-indigo-husky*
<br>Tensorflow 1.15.0:pip install tensorflow=1.15.0
***
<br>HOW TO RUN:
<br>git clone https://github.com/starcosmos1225/robot_reinforcementLearning.git
<br>cd robot_reinforcementLearning/
<br>cp -r robottest/ <your_ros_ws>/src/
<br>cd <your_ros_ws>
<br>catkin_make -j4
<br>source devel/setup.bash
<br>roslaunch robottest RL.launch
<br>Note:If you want see how husky move in gazebo,change the line: "arg name="gui" default="false" " to "arg name="gui" default="true" " in the file <RL_Env.launch>
