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
***
<br>Since training is too slow in the gazebo environment, I created a gym environment to simplify the environment interaction required for robot training. This environment is in the gym_env folder with the name "puckWorld_v3". The python file for training it is A3C_DQN_IMAGE.py. Because it is consistent with the data format accepted by the robot's environment, it can run well under gazebo after training.
<br>To use this simplified environment, you need to put puckWorld_v3.py into envs/classic_control in the installed gym address, and add a line at the end of classic_control/__init__.py: "from gym.envs.classic_control.puckWorld_v3 import puckworldEnv_v3". Then add some codes in envs/__init__.py:
<br>register(
<br>     id='puckworld-v3',
<br>     entry_point='gym.envs.classic_control:puckworldEnv_v3'
<br>)
