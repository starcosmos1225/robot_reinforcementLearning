#! /usr/bin/env python
from geometry_msgs.msg import Twist
import numpy as np
import math
class Interact:

    def __init__(self, action_n=5, pubAction = None):
        self.action_n = action_n
        self.state = [0]*4
        self.action_list = [(0.5, 0),
                            (0.2, 0.2),
                            (0.2, -0.2),
                            (0, 0)]
        self.velocity = 0
        self.rad = 0
        self.oldDistance = 0
        self.oldV = 0
        self.oldR = 0
        self.distanceLimit = 1.0
        self.reward = 20
        self.collusion = False
        self.pubAction = pubAction

    def getState(self):
        return np.array(self.state)

    def setState(self, state, index):
        #print("get state{} index{}".format(state,index))
        self.state[index] = state

    def step(self, action):
        self.oldV = self.velocity
        self.oldR = self.rad
        self.oldDistance = self.state[0]
        self.velocity, self.rad = self.action_list[action]
        msg = Twist()
        msg.linear.x = self.velocity
        msg.angular.z = self.rad
        self.pubAction.publish(msg)

    def getStatus(self):
        done = self.state[0] < self.distanceLimit
        if not done:
            current_reward = 100 * (self.oldDistance - 0.00001 - self.state[0]) - 0.001 * (
                    math.fabs(self.velocity - self.oldV) + math.fabs(self.rad - self.oldR))
            if self.collusion:
                current_reward -= 100
        else:
            current_reward = self.reward
        if self.collusion:
            done = True
        return np.array(self.state), current_reward, done, {}

    def getAction(self):
        return self.action_n
    def setCollusion(self,IsCollusion):
        self.collusion = IsCollusion

    def setPublish(self,publish):
        self.pubAction = publish